"""Alergenicidad via AlgPred 2.0 LOCAL (venv dedicado, subprocess puro).

Wrapper 100% local sobre ``algpred2.py`` (Raghava group, open source), mismo
patron que ``bepipred_engine.py``: invoca el interprete de un venv dedicado
(``Settings.ALGPRED_PYTHON_BIN``) contra el script instalado
(``Settings.ALGPRED_SCRIPT_PATH``), nunca red.

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

_OUTPUT_COLUMNS = ["sequence", "algpred_score", "algpred_veredicto"]


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
    sequences: List[str], output_dir: Path, filename_prefix: str = "", threshold: float = None
) -> pd.DataFrame:
    """Evalua alergenicidad de cada secuencia/peptido con AlgPred 2.0 local.

    Args:
        sequences: Peptidos/secuencias candidatos a evaluar. Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir el CSV crudo devuelto por AlgPred2.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para el CSV crudo.
        threshold: Umbral ML_Score (por defecto ``Settings.ALGPRED_THRESHOLD``).

    Returns:
        DataFrame con columnas ``sequence``, ``algpred_score``, ``algpred_veredicto``
        (``'Allergen'`` / ``'Non-Allergen'``, texto crudo de AlgPred2).

    Raises:
        EngineExecutionError: Si el venv/script no esta instalado, el
            subproceso falla/excede el timeout, o el CSV de salida no tiene
            el formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    _resolve_binary()
    threshold = threshold if threshold is not None else Settings.ALGPRED_THRESHOLD

    # Workaround del bug de tamano de batch == 1 (ver docstring del modulo).
    padded = len(sequences) == 1
    input_seqs = sequences + [sequences[0]] if padded else sequences

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / f"{filename_prefix}algpred_raw.csv"

    with tempfile.TemporaryDirectory(prefix="algpred_") as tmp:
        fasta_path = Path(tmp) / "candidates.fasta"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for i, seq in enumerate(input_seqs):
                fh.write(f">candidato_{i}\n{seq}\n")

        cmd = [
            Settings.ALGPRED_PYTHON_BIN, Settings.ALGPRED_SCRIPT_PATH,
            "-i", str(fasta_path), "-o", str(raw_csv_path),
            "-t", str(threshold), "-d", "2",
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
    if not {"Sequence", "ML_Score", "Prediction"}.issubset(raw.columns):
        raise EngineExecutionError(
            f"El formato del CSV de AlgPred2 no coincide con lo esperado. Columnas encontradas: {list(raw.columns)}."
        )

    if padded:
        raw = raw.iloc[: len(sequences)]

    result = pd.DataFrame(
        {
            "sequence": raw["Sequence"],
            "algpred_score": raw["ML_Score"],
            "algpred_veredicto": raw["Prediction"],
        }
    )
    return result[_OUTPUT_COLUMNS]


def print_allergenicity_report(report_df: pd.DataFrame) -> None:
    """Imprime el informe de alergenicidad: analogo a ``blast_engine.print_blast_report``."""
    if report_df.empty:
        print("No hay peptidos candidatos de la Fase 4 para evaluar alergenicidad.")
        return

    seq_width = max(30, report_df["sequence"].str.len().max() + 2)
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("ML_Score", lambda r: f"{r.algpred_score:.4f}", 12, ">"),
        Column("Veredicto", lambda r: r.algpred_veredicto, 16, ">"),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns)

    n_allergen = int((report_df["algpred_veredicto"] == "Allergen").sum())
    n_non_allergen = int((report_df["algpred_veredicto"] == "Non-Allergen").sum())
    print(f"\nResumen Fase 4b: {n_non_allergen} no alergeno(s) / {n_allergen} alergeno(s) potencial(es).")
