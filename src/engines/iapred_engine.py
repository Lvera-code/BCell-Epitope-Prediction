"""Antigenicidad intrinseca del constructo via IApred LOCAL (venv dedicado, subprocess puro).

Wrapper 100% local sobre ``IApred.py`` (Miles et al. 2025,
github.com/sebamiles/IApred, open source), mismo patron que
``netcleave_engine.py``. Reemplazo de VaxiJen: VaxiJen (la herramienta de
referencia historica para "antigenicidad de proteina completa") NO es
open-source, no tiene binario standalone descargable ni API publica
documentada para uso programatico -- descartado explicitamente, ver
STATUS.md. IApred es, a la fecha, la unica alternativa open-source/local
publicada especificamente para llenar ese hueco (2025).

A diferencia de los motores de Fase 2 (BepiPred/EpiDope/DiscoTope/ScanNet,
que puntuan antigenicidad POR RESIDUO para localizar epitopos), IApred
puntua la secuencia COMPLETA de una sola vez (SVM sobre features
fisicoquimicas agregadas, sin PyTorch/TensorFlow) -- exactamente la pregunta
que hace falta en Fase 8 ("es antigenico este constructo como un todo"), no
"donde estan los epitopos dentro de el".

Detalles de instalacion verificados empiricamente (no documentados en el
propio repo):

1. ``models_folder = "models"`` en ``IApred.py`` es una ruta RELATIVA
   resuelta contra el CWD del proceso, no contra la ubicacion del script.
   El subprocess se invoca siempre con ``cwd=Settings.IAPRED_HOME`` (la raiz
   del clon), igual motivo que ``discotope_engine.py``/``netcleave_engine.py``.
2. ``requirements.txt`` del repo esta INCOMPLETO: declara
   ``numpy``/``biopython``/``scikit-learn==1.5.2``/``joblib``/``scipy``
   (mas ``imbalanced-learn`` comentado) pero ``functions.py`` tambien
   importa, sin declararlas en ningun lado, ``imbalanced-learn``,
   ``matplotlib`` y ``seaborn`` en el top-level del modulo -- sin instalarlas
   a mano el import revienta con ``ModuleNotFoundError`` antes de llegar a
   ejecutar nada.
3. ``main()`` en ``IApred.py`` llama a ``check_and_install_dependencies()``
   al arrancar, que en teoria podria disparar un ``pip install`` de red si
   falta ``numpy``/``Bio``/``sklearn``/``joblib`` (OJO: esa lista NO incluye
   ``imbalanced-learn``/``matplotlib``/``seaborn``, ver punto 2). Con el venv
   de este proyecto ya completo (los 7 paquetes instalados), esa rama nunca
   se ejecuta -- confirmado empiricamente, no asumido -- pero si alguna vez
   el venv se reinstala sin completar el punto 2, el riesgo real es un
   ``ModuleNotFoundError``, no una llamada de red silenciosa (la lista
   chequeada por esa funcion no cubre las 3 dependencias faltantes).
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
            "Ver STATUS.md o apunta IAPRED_PYTHON_BIN a la ubicacion correcta."
        )
    if not script.is_file():
        raise EngineExecutionError(
            f"No se encontro el script 'IApred.py' en '{script}'. "
            "Ver STATUS.md o apunta IAPRED_HOME a la ubicacion correcta del clon."
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
        print("No hay secuencias candidatas para evaluar antigenicidad intrinseca.")
        return

    seq_width = max(30, report_df["sequence"].str.len().max() + 2)
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("Score", lambda r: f"{r.iapred_score:.4f}" if pd.notna(r.iapred_score) else "-", 10, ">"),
        Column("Categoria", lambda r: r.iapred_categoria, 30, ">"),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns)
