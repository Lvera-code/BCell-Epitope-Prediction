"""Toxicidad del constructo via ToxinPred 2.0 LOCAL (venv dedicado, subprocess puro).

Wrapper 100% local sobre ``toxinpred2`` (Raghava group, open source,
``pip install toxinpred2``), mismo patron que ``algpred_engine.py``. A
diferencia de ToxinPred 3.0 (pensado para peptidos cortos), el propio grupo
que lo desarrolla recomienda ToxinPred2 para proteinas/constructos de
longitud completa -- exactamente el caso de uso de Fase 8 (chequeo del
constructo multi-epitopo ensamblado, no peptidos individuales).

El modelo (Random Forest, exportado a ONNX) y el binario ``blastp`` (usado
por el modo ``-m 2``, hibrido) vienen EMBEBIDOS en el paquete pip
(``toxinpred2/model/RF_model.onnx.zip``, ``toxinpred2/blast_binaries/``):
no hay ningun paso de descarga de pesos aparte, ni en setup ni en runtime.

Venv dedicado con Python 3.10 (no 3.13, el default del sistema) + pandas
``1.5.3`` + ``numpy<2`` pineados, por dos bugs reales verificados
empiricamente al instalar:

1. El script empaquetado escribe su FASTA intermedio con
   ``CM.to_csv("Sequence_1", header=False, index=None, sep="\\n")`` -- un
   separador de mas de 1 caracter, que ``pandas>=2`` rechaza de plano con
   ``ValueError: bad delimiter value`` (pandas exige ``sep`` de longitud 1).
   No hay ningun flag de CLI que evite este paso: se ejecuta siempre, antes
   de cualquier rama de ``Model``/``Display``. Unica salida viable sin
   parchear el paquete instalado: pinear una version de pandas anterior a
   esa validacion estricta.
2. ``pandas==1.5.3`` esta compilado contra la ABI de ``numpy<2``: con
   ``numpy>=2`` (lo que trae por defecto ``pip install toxinpred2`` sin
   pines) el import de pandas revienta con
   ``ValueError: numpy.dtype size changed, may indicate binary incompatibility``.

Bug de batch de tamano 1 (verificado empiricamente, mismo patron que
``algpred_engine.py``): el modelo ONNX espera un tensor de rank 2
(``(n_muestras, n_features)``); con una unica secuencia de entrada, el
pipeline de features del script produce un array de rank 1 y ONNX Runtime
revienta con ``INVALID_ARGUMENT: Invalid rank for input``. Esto NO es un
caso de borde marginal aqui: Fase 8 evalua un unico constructo ensamblado
por corrida, asi que este workaround (duplicar la secuencia, descartar la
fila extra del resultado) se activa en el camino normal, no solo en tests.
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

_OUTPUT_COLUMNS = ["sequence", "toxinpred_score", "toxinpred_veredicto"]


def _resolve_binary() -> Path:
    """Valida que el interprete y el script de consola de ToxinPred2 existan.

    ``toxinpred2`` se instala como entry point de consola (``pip install
    toxinpred2``, sin ``__main__.py`` propio: ``python -m toxinpred2`` NO
    funciona), asi que se invoca el ejecutable ``toxinpred2`` del ``bin/``
    del venv directamente, no el interprete + un path de script como en
    ``algpred_engine.py``/``netcleave_engine.py`` (esos si son scripts .py sueltos).
    """
    python_bin = Path(Settings.TOXINPRED2_PYTHON_BIN)
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de ToxinPred2 en '{python_bin}'. "
            "Ver STATUS.md o apunta TOXINPRED2_PYTHON_BIN a la ubicacion correcta."
        )
    binary = python_bin.parent / Settings.TOXINPRED2_BINARY_NAME
    if not binary.is_file():
        raise EngineExecutionError(
            f"No se encontro el ejecutable '{binary}'. Ver STATUS.md o reinstala ToxinPred2 "
            f"en el venv de TOXINPRED2_PYTHON_BIN ('pip install toxinpred2')."
        )
    return binary


def predict_toxicity(
    sequences: List[str], output_dir: Path, filename_prefix: str = "", threshold: float = None
) -> pd.DataFrame:
    """Evalua toxicidad de cada secuencia con ToxinPred 2.0 local.

    Args:
        sequences: Secuencias a evaluar (tipicamente un unico constructo
            ensamblado en Fase 8, pero acepta cualquier batch). Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir el CSV crudo devuelto por ToxinPred2.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para el CSV crudo.
        threshold: Umbral de score (por defecto ``Settings.TOXINPRED2_THRESHOLD``, 0.6).

    Returns:
        DataFrame con columnas ``sequence``, ``toxinpred_score``, ``toxinpred_veredicto``
        (``'Toxin'`` / ``'Non-Toxin'``, texto crudo de ToxinPred2).

    Raises:
        EngineExecutionError: Si el venv no esta instalado, el subproceso
            falla/excede el timeout, o el CSV de salida no tiene el formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    binary = _resolve_binary()
    threshold = threshold if threshold is not None else Settings.TOXINPRED2_THRESHOLD

    # Workaround del bug de tamano de batch == 1 (ver docstring del modulo).
    padded = len(sequences) == 1
    input_seqs = sequences + [sequences[0]] if padded else sequences

    output_dir.mkdir(parents=True, exist_ok=True)
    # Resuelto a absoluto: el subprocess de abajo corre con 'cwd=tmp' (el
    # directorio temporal del batch), asi que un 'output_dir' relativo
    # (default de Settings.FASTA_OUTPUT_DIR) se resolveria contra ESE
    # directorio, no el de pipeline.py -- mismo bug real confirmado en
    # algpred_engine.py, ver su docstring inline.
    raw_csv_path = (output_dir / f"{filename_prefix}toxinpred_raw.csv").resolve()

    with tempfile.TemporaryDirectory(prefix="toxinpred_") as tmp:
        fasta_path = Path(tmp) / "candidates.fasta"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for i, seq in enumerate(input_seqs):
                fh.write(f">candidato_{i}\n{seq}\n")

        cmd = [
            str(binary),
            "-i", str(fasta_path), "-o", str(raw_csv_path),
            "-t", str(threshold), "-d", "2",
        ]
        logger.info("Ejecutando ToxinPred2 local: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=Settings.TOXINPRED2_TIMEOUT_SECONDS, cwd=tmp)
        except subprocess.CalledProcessError as exc:
            raise EngineExecutionError(
                f"ToxinPred2 termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(f"ToxinPred2 excedio el tiempo limite de {Settings.TOXINPRED2_TIMEOUT_SECONDS}s.") from exc

    if not raw_csv_path.is_file():
        raise EngineExecutionError(f"ToxinPred2 termino sin error pero no genero el CSV esperado en '{raw_csv_path}'.")

    raw = pd.read_csv(raw_csv_path)
    if not {"Sequence", "ML_Score", "Prediction"}.issubset(raw.columns):
        raise EngineExecutionError(
            f"El formato del CSV de ToxinPred2 no coincide con lo esperado. Columnas encontradas: {list(raw.columns)}."
        )

    if padded:
        raw = raw.iloc[: len(sequences)]

    result = pd.DataFrame(
        {
            "sequence": raw["Sequence"],
            "toxinpred_score": raw["ML_Score"],
            "toxinpred_veredicto": raw["Prediction"],
        }
    )
    return result[_OUTPUT_COLUMNS]


def print_toxicity_report(report_df: pd.DataFrame) -> None:
    """Imprime el informe de toxicidad: analogo a ``algpred_engine.print_allergenicity_report``."""
    if report_df.empty:
        print("No hay secuencias candidatas para evaluar toxicidad.")
        return

    seq_width = max(30, report_df["sequence"].str.len().max() + 2)
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("ML_Score", lambda r: f"{r.toxinpred_score:.4f}", 12, ">"),
        Column("Veredicto", lambda r: r.toxinpred_veredicto, 16, ">"),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns)

    n_toxin = int((report_df["toxinpred_veredicto"] == "Toxin").sum())
    n_non_toxin = int((report_df["toxinpred_veredicto"] == "Non-Toxin").sum())
    print(f"\nResumen toxicidad: {n_non_toxin} no toxico(s) / {n_toxin} toxico(s) potencial(es).")
