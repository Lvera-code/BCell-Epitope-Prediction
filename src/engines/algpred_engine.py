"""Allergenicity via LOCAL AlgPred 2.0 (dedicated venv, pure subprocess).

A 100% local wrapper over ``algpred2.py`` (Raghava group, open source), the same
pattern as ``bepipred_engine.py``: it invokes the interpreter of a dedicated venv
(``Settings.ALGPRED_PYTHON_BIN``) against the installed script
(``Settings.ALGPRED_SCRIPT_PATH``), never the network.

Model (``Settings.ALGPRED_MODEL``, 2 by default): the
upstream script supports two modes with DIFFERENT output CSV formats,
which is why this wrapper parses each one separately instead of assuming a single
schema:

    Model 1 (pure ML): columns ``ID,Sequence,ML_Score,Prediction``, a
        Random Forest over amino acid composition (AAC) only, without
        any comparison against real allergens/IgE motifs.
    Model 2 (hybrid, default): columns
        ``Subject,ML Score,MERCI Score,BLAST Score,Hybrid Score,Prediction``,
        which combines the same RF with BLAST against a real database of IgE
        allergens and MERCI against documented IgE motifs (``Hybrid Score`` = the sum
        of the three). It does NOT carry a ``Sequence`` column: each row's sequence
        is reconstructed BY POSITION against the input list, having verified
        that the order is preserved end to end in the upstream script (the same
        ``seqid``/``seq`` list reused at each internal step).

Model 2 was chosen as the default after verifying that model 1 marks as
"Allergen" candidates with no real homology or IgE motif behind them, an
artifact of the amino acid composition, not biological
evidence. The result always exposes ``algpred_ml_score``/
``algpred_merci_score``/``algpred_blast_score`` separately (0 if model
1 did not compute them) so as to be able to distinguish a verdict backed by
real evidence from one that is only composition statistics.

A known bug of the upstream script (verified empirically, not assumed):
``algpred2.py`` fails with ``ValueError: Expected 2D array, got 1D array``
when the input FASTA has EXACTLY 1 sequence (its own sklearn code does not
reshape a batch of size 1). This is avoided by duplicating the single input
sequence when ``len(sequences) == 1`` and discarding the duplicated row
from the result, a workaround that is necessary because the installed script
cannot be patched without breaking the pip package's traceability.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import EngineExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

_OUTPUT_COLUMNS = [
    "sequence", "algpred_score", "algpred_veredicto",
    "algpred_ml_score", "algpred_merci_score", "algpred_blast_score",
]
_MODEL1_COLUMNS = {"Sequence", "ML_Score", "Prediction"}
_MODEL2_COLUMNS = {"Subject", "ML Score", "MERCI Score", "BLAST Score", "Hybrid Score", "Prediction"}


def _resolve_binary() -> None:
    """Valida que el interprete y el script de AlgPred2 existan."""
    python_bin = Path(Settings.ALGPRED_PYTHON_BIN)
    script = Path(Settings.ALGPRED_SCRIPT_PATH)
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de AlgPred2 en '{python_bin}'. "
            "Ver README (Seccion de Instalacion) o apunta ALGPRED_PYTHON_BIN a la ubicacion correcta."
        )
    if not script.is_file():
        raise EngineExecutionError(
            f"No se encontro el script 'algpred2.py' en '{script}'. "
            "Ver README (Seccion de Instalacion) o apunta ALGPRED_SCRIPT_PATH a la ubicacion correcta."
        )


def predict_allergenicity(
    sequences: List[str], output_dir: Path, filename_prefix: str = "",
    threshold: float = None, model: int = None,
) -> pd.DataFrame:
    """Evalua alergenicidad de cada secuencia/peptido con AlgPred 2.0 local.

    Args:
        sequences: Peptidos/secuencias candidatos a evaluar. Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir el CSV crudo devuelto por AlgPred2.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para el CSV crudo.
        threshold: Umbral ML_Score/Hybrid Score (por defecto ``Settings.ALGPRED_THRESHOLD``).
        model: 1 (ML puro) o 2/hibrido (por defecto ``Settings.ALGPRED_MODEL``). Ver
            docstring del modulo para la diferencia de esquema de salida entre ambos.

    Returns:
        DataFrame con columnas ``sequence``, ``algpred_score`` (Hybrid Score en
        modelo 2, ML_Score en modelo 1), ``algpred_veredicto`` (``'Allergen'`` /
        ``'Non-Allergen'``, texto crudo de AlgPred2), y el desglose
        ``algpred_ml_score``/``algpred_merci_score``/``algpred_blast_score``
        (estos dos ultimos siempre 0 en modelo 1, que no los calcula).

    Raises:
        EngineExecutionError: Si el venv/script no esta instalado, el
            subproceso falla/excede el timeout, o el CSV de salida no tiene
            el formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    _resolve_binary()
    threshold = threshold if threshold is not None else Settings.ALGPRED_THRESHOLD
    model = model if model is not None else Settings.ALGPRED_MODEL

    # Workaround del bug de tamano de batch == 1 (ver docstring del modulo).
    padded = len(sequences) == 1
    input_seqs = sequences + [sequences[0]] if padded else sequences

    output_dir.mkdir(parents=True, exist_ok=True)
    # Resuelto a absoluto: el subprocess de abajo corre con 'cwd' forzado a
    # la carpeta del script de AlgPred2 (algpred2.py necesita rutas propias
    # relativas a su instalacion), asi que un 'output_dir' relativo (default
    # de Settings.FASTA_OUTPUT_DIR) se resolveria mal si se le pasa tal cual:
    # 'OSError: Cannot save file into a non-existent directory' porque el
    # hijo interpreta la ruta relativa contra SU cwd, no el de pipeline.py.
    raw_csv_path = (output_dir / f"{filename_prefix}algpred_raw.csv").resolve()

    with tempfile.TemporaryDirectory(prefix="algpred_") as tmp:
        fasta_path = Path(tmp) / "candidates.fasta"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for i, seq in enumerate(input_seqs):
                fh.write(f">candidato_{i}\n{seq}\n")

        cmd = [
            Settings.ALGPRED_PYTHON_BIN, Settings.ALGPRED_SCRIPT_PATH,
            "-i", str(fasta_path), "-o", str(raw_csv_path),
            "-t", str(threshold), "-m", str(model), "-d", "2",
        ]
        logger.info("Ejecutando AlgPred2 local: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=Settings.ALGPRED_TIMEOUT_SECONDS,
                            cwd=Path(Settings.ALGPRED_SCRIPT_PATH).parent)
        except subprocess.CalledProcessError as exc:
            raise EngineExecutionError(
                f"AlgPred2 termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(f"AlgPred2 excedio el tiempo limite de {Settings.ALGPRED_TIMEOUT_SECONDS}s.") from exc

    if not raw_csv_path.is_file():
        raise EngineExecutionError(f"AlgPred2 termino sin error pero no genero el CSV esperado en '{raw_csv_path}'.")

    raw = pd.read_csv(raw_csv_path)
    cols = set(raw.columns)

    if _MODEL2_COLUMNS.issubset(cols):
        # Modelo hibrido: sin columna 'Sequence', se reconstruye por posicion
        # (ver docstring del modulo -- el orden de salida de algpred2.py
        # preserva el de entrada de punta a punta).
        if len(raw) != len(input_seqs):
            raise EngineExecutionError(
                f"AlgPred2 (modo hibrido) devolvio {len(raw)} filas para {len(input_seqs)} secuencias de "
                "entrada -- no se puede reconstruir la secuencia por posicion con seguridad."
            )
        result = pd.DataFrame(
            {
                "sequence": input_seqs,
                "algpred_score": raw["Hybrid Score"],
                "algpred_veredicto": raw["Prediction"],
                "algpred_ml_score": raw["ML Score"],
                "algpred_merci_score": raw["MERCI Score"],
                "algpred_blast_score": raw["BLAST Score"],
            }
        )
    elif _MODEL1_COLUMNS.issubset(cols):
        result = pd.DataFrame(
            {
                "sequence": raw["Sequence"],
                "algpred_score": raw["ML_Score"],
                "algpred_veredicto": raw["Prediction"],
                "algpred_ml_score": raw["ML_Score"],
                "algpred_merci_score": 0.0,
                "algpred_blast_score": 0.0,
            }
        )
    else:
        raise EngineExecutionError(
            f"El formato del CSV de AlgPred2 no coincide con lo esperado (modelo {model}). "
            f"Columnas encontradas: {list(raw.columns)}."
        )

    if padded:
        result = result.iloc[: len(sequences)]

    return result[_OUTPUT_COLUMNS].reset_index(drop=True)


def print_allergenicity_report(report_df: pd.DataFrame, show_sequence: bool = True) -> None:
    """Imprime el informe de alergenicidad: analogo a ``blast_engine.print_blast_report``.

    ``show_sequence``: en Fase 4b (por peptido, muchos candidatos) la
    secuencia es la unica forma de identificar cada fila -> True (default).
    En Fase 8 (constructo ensamblado, una sola fila de 200+ aa) mostrarla
    solo agrega ruido -> False (ver ``pipeline.py::fase_8_chequeo_constructo``).
    """
    if report_df.empty:
        print("No hay peptidos candidatos de la Fase 4 para evaluar alergenicidad.")
        return

    columns = []
    if show_sequence:
        seq_width = max(30, report_df["sequence"].str.len().max() + 2)
        columns.append(Column("Secuencia", lambda r: r.sequence, seq_width, "<"))
    columns.append(Column("Score", lambda r: f"{r.algpred_score:.4f}", 10, ">"))
    columns.append(Column("Veredicto", lambda r: r.algpred_veredicto, 16, ">"))
    # Evidencia real = BLAST y/o MERCI aportaron algo (!= 0), no solo el RF
    # de composicion de aminoacidos -- ver docstring del modulo. Siempre "-"
    # si se corrio en modelo 1 (esas columnas quedan a 0 por diseno).
    columns.append(Column(
        "Evidencia BLAST/MERCI",
        lambda r: "si" if (r.algpred_merci_score or r.algpred_blast_score) else "no",
        22, ">",
    ))
    print_fixed_width_table(report_df.itertuples(index=False), columns)

    n_allergen = int((report_df["algpred_veredicto"] == "Allergen").sum())
    n_non_allergen = int((report_df["algpred_veredicto"] == "Non-Allergen").sum())
    print(f"\nResumen alergenicidad: {n_non_allergen} no alergeno(s) / {n_allergen} alergeno(s) potencial(es).")
