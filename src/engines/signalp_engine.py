"""Peptido senal del constructo via SignalP-6.0 LOCAL (venv dedicado, subprocess puro).

Wrapper 100% local sobre ``signalp6`` (DTU Health Tech, licencia academica,
descarga manual obligatoria -- mismo patron que BepiPred-3.0/NetMHCIIpan-4.3/
NetMHCpan-4.2), mismo criterio de subprocess que el resto de motores.

Proposito en Fase 8: confirmar que el constructo multi-epitopo final
ensamblado NO tenga un peptido senal predicho en su extremo N-terminal. Un
constructo de fusion sintetico pensado para expresion recombinante estandar
no deberia tener uno -- si SignalP predice uno, es señal de que la union de
fragmentos (o el propio primer epitopo B-cell) genero por accidente un
motivo con esa forma, y vale la pena revisarlo antes de dar el constructo
por valido. Es puramente informativo aqui, no un filtro automatico.

Modo ``slow-sequential`` (~9.2GB de pesos): corre el modelo completo (no la
aproximacion rapida) de forma secuencial en vez de en paralelo, mismo
footprint de RAM que el modo ``fast`` pero ~6x mas lento -- pensado para
maquinas CPU-only con RAM limitada (el modo ``slow`` en paralelo requiere
>14GB libres).

Detalle de instalacion (no documentado en el README oficial de DTU): los
pesos se referencian por ``--model_dir`` apuntando DIRECTO a la carpeta
que ya contiene ``sequential_models_signalp6/`` (``Settings.SIGNALP_MODEL_DIR``),
en vez de copiarlos dentro del paquete instalado (paso "4" del README
oficial) -- evita duplicar ~9.2GB. Venv con Python 3.10 + ``torch>1.7,<2``
(pin del propio ``requirements.txt``) + ``numpy<2`` (ABI: sin este pin,
``pip install`` arrastra numpy 2.x via otra dependencia -matplotlib- que
rompe contra el torch 1.13 compilado con ABI de numpy 1.x, mismo tipo de
bug que ``toxinpred_engine.py``).
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

_OUTPUT_COLUMNS = ["sequence", "signalp_prediction", "signalp_prob_other", "signalp_prob_sp", "signalp_cs_position"]


def _resolve_binary() -> Path:
    """Valida que el interprete, el binario y los pesos de SignalP-6.0 existan."""
    python_bin = Path(Settings.SIGNALP_PYTHON_BIN)
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de SignalP-6.0 en '{python_bin}'. "
            "Ver README.md (Seccion 15) o apunta SIGNALP_PYTHON_BIN a la ubicacion correcta."
        )
    binary = python_bin.parent / Settings.SIGNALP_BINARY_NAME
    if not binary.is_file():
        raise EngineExecutionError(
            f"No se encontro el ejecutable '{binary}'. Ver README.md (Seccion 15) o reinstala SignalP-6.0 "
            "en el venv de SIGNALP_PYTHON_BIN ('pip install ./signalp-6.0')."
        )
    model_dir = Path(Settings.SIGNALP_MODEL_DIR)
    if not (model_dir / "sequential_models_signalp6").is_dir():
        raise EngineExecutionError(
            f"No se encontro 'sequential_models_signalp6/' dentro de '{model_dir}'. "
            "Por restricciones de licencia academica, DTU Health Tech no permite redistribuir "
            "los pesos: descargalos manualmente y apunta SIGNALP_MODEL_DIR a la carpeta que "
            "los contiene. Ver README.md - Seccion de Instalacion."
        )
    return binary


def predict_signal_peptide(sequences: List[str], output_dir: Path, filename_prefix: str = "") -> pd.DataFrame:
    """Evalua presencia de peptido senal N-terminal con SignalP-6.0 local (modo slow-sequential).

    Args:
        sequences: Secuencias a evaluar (tipicamente un unico constructo
            ensamblado en Fase 8). Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir la salida cruda de SignalP, para trazabilidad.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para la subcarpeta cruda.

    Returns:
        DataFrame con columnas ``sequence``, ``signalp_prediction`` (``'OTHER'``
        si NO se detecto peptido senal; ``'SP(Sec/SPI)'``/``'LIPO(Sec/SPII)'``/etc.
        si si), ``signalp_prob_other``, ``signalp_prob_sp`` (probabilidad del tipo
        Sec/SPI, el mas comun) y ``signalp_cs_position`` (posicion del sitio de
        corte predicho, vacio si ``prediction == 'OTHER'``).

    Raises:
        EngineExecutionError: Si el venv/pesos no estan instalados, el
            subproceso falla/excede el timeout, o la salida no tiene el
            formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    binary = _resolve_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="signalp_") as tmp:
        tmp_dir = Path(tmp)
        fasta_path = tmp_dir / "candidates.fasta"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for i, seq in enumerate(sequences):
                fh.write(f">candidato_{i}\n{seq}\n")

        run_output_dir = tmp_dir / "out"
        cmd = [
            str(binary),
            "--fastafile", str(fasta_path), "--output_dir", str(run_output_dir),
            "--format", "none", "--mode", "slow-sequential",
            "--organism", Settings.SIGNALP_ORGANISM,
            "--model_dir", Settings.SIGNALP_MODEL_DIR,
        ]
        logger.info("Ejecutando SignalP-6.0 local: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=Settings.SIGNALP_TIMEOUT_SECONDS)
        except subprocess.CalledProcessError as exc:
            raise EngineExecutionError(
                f"SignalP-6.0 termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(f"SignalP-6.0 excedio el tiempo limite de {Settings.SIGNALP_TIMEOUT_SECONDS}s.") from exc

        results_path = run_output_dir / "prediction_results.txt"
        if not results_path.is_file():
            raise EngineExecutionError(
                f"SignalP-6.0 termino sin error pero no genero '{results_path}'."
            )

        # 'prediction_results.txt' trae 2 lineas de comentario ('#...') antes
        # de los datos (cabecera '# SignalP-6.0 ...' + '# ID Prediction ...'),
        # numero variable segun version -- 'comment="#"' las salta todas sin
        # asumir cuantas hay, en vez de un 'skiprows' fijo (bug real: con
        # skiprows=1 la segunda linea de comentario se leia como fila de datos).
        raw = pd.read_csv(results_path, sep="\t", comment="#", header=None)
        raw.columns = ["ID", "Prediction", "OTHER", "SP", "LIPO", "TAT", "TATLIPO", "PILIN", "CS_Position"]

        raw_copy_path = output_dir / f"{filename_prefix}signalp_raw.txt"
        raw_copy_path.write_bytes(results_path.read_bytes())

    if len(raw) != len(sequences):
        raise EngineExecutionError(
            f"SignalP-6.0 devolvio {len(raw)} prediccion(es), se esperaban {len(sequences)} "
            "(una por secuencia de entrada)."
        )

    result = pd.DataFrame(
        {
            "sequence": sequences,
            "signalp_prediction": raw["Prediction"].tolist(),
            "signalp_prob_other": raw["OTHER"].tolist(),
            "signalp_prob_sp": raw["SP"].tolist(),
            "signalp_cs_position": raw["CS_Position"].tolist(),
        }
    )
    return result[_OUTPUT_COLUMNS]


def print_signalp_report(report_df: pd.DataFrame) -> None:
    """Imprime el informe de peptido senal del constructo."""
    if report_df.empty:
        print("No hay secuencias candidatas para evaluar peptido senal.")
        return

    columns = [
        Column("Prediccion", lambda r: r.signalp_prediction, 16, ">"),
        Column("Prob(OTHER)", lambda r: f"{r.signalp_prob_other:.4f}", 12, ">"),
        # prefix="  ": 'CS pos: 24-25. Pr: 0.9771' (texto crudo de SignalP)
        # suele exceder el 'width' -- sin este separador explicito, una
        # columna que supera su ancho minimo queda pegada a la anterior sin
        # ningun espacio (Column.width es un MINIMO, no trunca, ver table_format.py).
        Column("Sitio de corte", lambda r: str(r.signalp_cs_position) if pd.notna(r.signalp_cs_position) else "-", 20, "<", prefix="  "),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns)

    n_with_sp = int((report_df["signalp_prediction"] != "OTHER").sum())
    if n_with_sp:
        print(f"\n[AVISO] {n_with_sp} secuencia(s) con peptido senal predicho -- revisar antes de dar el constructo por valido.")
    else:
        print("\nNinguna secuencia tiene peptido senal predicho (resultado esperado para un constructo de fusion).")
