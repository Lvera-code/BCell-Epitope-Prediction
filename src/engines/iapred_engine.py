"""Intrinsic antigenicity of the construct via LOCAL IApred (dedicated venv, pure subprocess).

A 100% local wrapper over ``IApred.py`` (Miles et al. 2025,
github.com/sebamiles/IApred, open source), the same pattern as
``netcleave_engine.py``. A replacement for VaxiJen: VaxiJen (the historical
reference tool for "whole-protein antigenicity") is NOT
open-source, has no downloadable standalone binary and no documented public
API for programmatic use, and was explicitly discarded. IApred is
the only open-source/local alternative published specifically to
fill that gap.

Unlike the Phase 2 engines (BepiPred/EpiDope/DiscoTope/ScanNet,
which score antigenicity PER RESIDUE in order to locate epitopes), IApred
scores the COMPLETE sequence in one pass (an SVM over aggregated
physicochemical features, without PyTorch/TensorFlow), which is exactly the question
needed in Phase 8 ("is this construct antigenic as a whole"), not
"where are the epitopes inside it".

Installation details (not documented in the repo itself):

1. ``models_folder = "models"`` in ``IApred.py`` is a RELATIVE path
   resolved against the process CWD, not against the script's location.
   The subprocess is therefore always invoked with ``cwd=Settings.IAPRED_HOME`` (the root
   of the clone), for the same reason as ``discotope_engine.py``/``netcleave_engine.py``.
2. The repo's ``requirements.txt`` is INCOMPLETE: it declares
   ``numpy``/``biopython``/``scikit-learn==1.5.2``/``joblib``/``scipy``
   (plus ``imbalanced-learn`` commented out) but ``functions.py`` also
   imports, without declaring them anywhere, ``imbalanced-learn``,
   ``matplotlib`` and ``seaborn`` at module top level, so without installing them
   manually the import fails with ``ModuleNotFoundError`` before reaching
   any execution.
3. ``main()`` in ``IApred.py`` calls ``check_and_install_dependencies()``
   at startup, which could in theory trigger a network ``pip install`` if
   ``numpy``/``Bio``/``sklearn``/``joblib`` are missing (note that this list does NOT include
   ``imbalanced-learn``/``matplotlib``/``seaborn``, see point 2). With this
   project's venv already complete (all 7 packages installed), that branch never
   executes, but if the venv is ever reinstalled without completing
   point 2, the real risk is a
   ``ModuleNotFoundError``, not a silent network call (the list
   checked by that function does not cover the 3 missing dependencies).
"""

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

_OUTPUT_COLUMNS = ["sequence", "iapred_score", "iapred_categoria"]


def _resolve_binary() -> Path:
    """Valida que el interprete, el script y la carpeta de modelos de IApred existan."""
    python_bin = Path(Settings.IAPRED_PYTHON_BIN)
    home = Path(Settings.IAPRED_HOME)
    script = home / Settings.IAPRED_SCRIPT_NAME
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de IApred en '{python_bin}'. "
            "Ver README.md (Seccion 14) o apunta IAPRED_PYTHON_BIN a la ubicacion correcta."
        )
    if not script.is_file():
        raise EngineExecutionError(
            f"No se encontro el script 'IApred.py' en '{script}'. "
            "Ver README.md (Seccion 14) o apunta IAPRED_HOME a la ubicacion correcta del clon."
        )
    if not (home / "models").is_dir():
        raise EngineExecutionError(
            f"No se encontro la carpeta 'models/' de IApred en '{home}'. "
            "Este clon debe traer los pesos bundled (ver 'git clone https://github.com/sebamiles/IApred.git')."
        )
    return script


def predict_intrinsic_antigenicity(sequences: List[str], output_dir: Path, filename_prefix: str = "") -> pd.DataFrame:
    """Evalua antigenicidad intrinseca de cada secuencia completa con IApred local.

    Args:
        sequences: Secuencias a evaluar (tipicamente un unico constructo
            ensamblado en Fase 8, pero acepta cualquier batch, sin el bug de
            tamano 1 que si tienen AlgPred2/ToxinPred2). Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir el CSV crudo devuelto por IApred.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para el CSV crudo.

    Returns:
        DataFrame con columnas ``sequence``, ``iapred_score`` (rango
        aproximado -3 a +3, ver el propio script) y ``iapred_categoria``
        (``'Low'``/``'Moderate'``/``'High'``, texto crudo de IApred).

    Raises:
        EngineExecutionError: Si el venv/script no esta instalado, el
            subproceso falla/excede el timeout, o el CSV de salida no tiene
            el formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    script = _resolve_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Resuelto a absoluto: el subprocess de abajo corre con
    # 'cwd=Settings.IAPRED_HOME', asi que un 'output_dir' relativo (default
    # de Settings.FASTA_OUTPUT_DIR) se resolveria contra ESA carpeta, no la
    # de pipeline.py -- mismo bug real confirmado en algpred_engine.py, ver
    # su docstring inline.
    raw_csv_path = (output_dir / f"{filename_prefix}iapred_raw.csv").resolve()

    with tempfile.TemporaryDirectory(prefix="iapred_") as tmp:
        fasta_path = Path(tmp) / "candidates.fasta"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for i, seq in enumerate(sequences):
                fh.write(f">candidato_{i}\n{seq}\n")

        cmd = [
            Settings.IAPRED_PYTHON_BIN, str(script),
            str(fasta_path), str(raw_csv_path), "-v",
        ]
        logger.info("Ejecutando IApred local: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=Settings.IAPRED_TIMEOUT_SECONDS, cwd=Settings.IAPRED_HOME)
        except subprocess.CalledProcessError as exc:
            raise EngineExecutionError(
                f"IApred termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(f"IApred excedio el tiempo limite de {Settings.IAPRED_TIMEOUT_SECONDS}s.") from exc

        if not raw_csv_path.is_file():
            raise EngineExecutionError(f"IApred termino sin error pero no genero el CSV esperado en '{raw_csv_path}'.")

        raw = pd.read_csv(raw_csv_path)

    expected_cols = {"Header", "Sequence_Length", "Intrinsic_Antigenicity_Score", "Antigenicity_Category"}
    if not expected_cols.issubset(raw.columns):
        raise EngineExecutionError(
            f"El formato del CSV de IApred no coincide con lo esperado. Columnas encontradas: {list(raw.columns)}."
        )
    if len(raw) != len(sequences):
        raise EngineExecutionError(
            f"IApred devolvio {len(raw)} fila(s), se esperaban {len(sequences)} (una por secuencia de entrada)."
        )

    # Bug/limitacion real de IApred verificada empiricamente (IApred.py linea
    # ~116): para secuencias de MENOS de 20 aa, no calcula nada -- escribe el
    # texto literal 'Sequence too short' en la columna de score (no un
    # numero) y 'N/A' en la de categoria. Sin esta coercion, un consumidor
    # que asuma 'iapred_score' siempre numerico (como
    # ``print_iapred_report``, que formatea con ``:.4f``) revienta. Se
    # detecta con ``pd.to_numeric(errors='coerce')`` -> NaN, y se reemplaza
    # la categoria perdida (pandas ya la leyo como NaN: 'N/A' es un token de
    # NA reconocido por defecto en ``pd.read_csv``) por un mensaje explicito,
    # en vez de dejar un NaN silencioso sin explicacion.
    scores = pd.to_numeric(raw["Intrinsic_Antigenicity_Score"], errors="coerce")
    categorias = raw["Antigenicity_Category"].tolist()
    too_short_mask = scores.isna()
    categorias = [
        "No evaluado (secuencia < 20 aa)" if is_short and pd.isna(cat) else cat
        for is_short, cat in zip(too_short_mask, categorias)
    ]

    result = pd.DataFrame(
        {
            "sequence": sequences,
            "iapred_score": scores.tolist(),
            "iapred_categoria": categorias,
        }
    )
    return result[_OUTPUT_COLUMNS]


def print_iapred_report(report_df: pd.DataFrame) -> None:
    """Imprime el informe de antigenicidad intrinseca del constructo."""
    if report_df.empty:
        print("No hay secuencias candidatas para evaluar antigenicidad.")
        return

    columns = [
        Column("Score", lambda r: f"{r.iapred_score:.4f}" if pd.notna(r.iapred_score) else "-", 10, ">"),
        Column("Categoria", lambda r: r.iapred_categoria, 30, ">"),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns)

    categoria = report_df.iloc[0]["iapred_categoria"]
    score = report_df.iloc[0]["iapred_score"]
    score_txt = f"{score:.4f}" if pd.notna(score) else "-"
    print(f"\nResumen antigenicidad: {categoria} (score {score_txt}).")
