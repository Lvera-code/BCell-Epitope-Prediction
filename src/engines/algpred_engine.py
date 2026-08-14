"""Alergenicidad via AlgPred 2.0 LOCAL (venv dedicado, subprocess puro).

Wrapper 100% local sobre ``algpred2.py`` (Raghava group, open source), mismo
patron que ``bepipred_engine.py``: invoca el interprete de un venv dedicado
(``Settings.ALGPRED_PYTHON_BIN``) contra el script instalado
(``Settings.ALGPRED_SCRIPT_PATH``), nunca red.

Modelo (``Settings.ALGPRED_MODEL``, por defecto 2 desde 2026-08-14): el
script upstream soporta dos modos con formatos de CSV de salida DISTINTOS,
por eso este wrapper parsea cada uno por separado en vez de asumir un unico
esquema:

    Modelo 1 (ML puro): columnas ``ID,Sequence,ML_Score,Prediction`` --
        Random Forest sobre composicion de aminoacidos (AAC) unicamente, sin
        ninguna comparacion contra alergenos/motivos IgE reales.
    Modelo 2 (hibrido, default): columnas
        ``Subject,ML Score,MERCI Score,BLAST Score,Hybrid Score,Prediction``
        -- combina el mismo RF con BLAST contra una base real de alergenos
        IgE y MERCI contra motivos IgE documentados (``Hybrid Score`` = suma
        de los tres). NO trae columna ``Sequence``: la secuencia de cada fila
        se reconstruye por POSICION contra la lista de entrada, verificado
        que el orden se preserva end-to-end en el script upstream (misma
        lista ``seqid``/``seq`` reutilizada en cada paso interno).

Se eligio el modelo 2 como default tras verificar (sesion 2026-08-14, ver
vault) que el modelo 1 marca "Allergen" candidatos sin ninguna homologia o
motivo IgE real detras -- artefacto de la composicion de aminoacidos, no
evidencia biologica. El resultado siempre expone ``algpred_ml_score``/
``algpred_merci_score``/``algpred_blast_score`` por separado (0 si el modelo
1 no los calculo) para poder distinguir un veredicto respaldado por
evidencia real de uno que es solo estadistica de composicion.

Bug conocido del script upstream (verificado empiricamente, no asumido):
``algpred2.py`` revienta con ``ValueError: Expected 2D array, got 1D array``
cuando el FASTA de entrada tiene EXACTAMENTE 1 secuencia (su propio codigo
sklearn no reshapea un batch de tamano 1). Se evita duplicando la unica
secuencia de entrada cuando ``len(sequences) == 1`` y descartando la fila
duplicada del resultado -- workaround necesario porque no se puede parchear
el script instalado sin romper la trazabilidad del paquete pip.
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
