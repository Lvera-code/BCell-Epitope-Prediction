"""N-glicosilacion via StackGlyEmbed LOCAL (venv dedicado, subprocess puro).

Wrapper 100% local sobre ``StackGlyEmbed/prediction/predict_local.py`` (venv
propio ``.venv-stackglyembed`` con torch/xgboost/sklearn/transformers/
tensorflow + ProteinBERT), mismo patron que ``bepipred_engine.py``/
``algpred_engine.py``/``netcleave_engine.py``: invoca el interprete de un
venv dedicado contra un script, nunca red en runtime (los 3 embedders que
consume -- ProteinBERT, ESM-2 650M, ProtT5 -- cargan offline una vez
cacheados, ver docstring de ``predict_local.py``).

El repo original de StackGlyEmbed (github.com/GaryChan-lab/StackGlyEmbed) no
trae un scanner de secuones: espera que el usuario le pase manualmente las
posiciones candidatas en ``dataset.txt``. ``scan_sequons`` implementa la
regla estandar de N-glicosilacion (Asn-X-Ser/Thr, X != Prolina) para derivar
esas posiciones automaticamente de cada peptido/secuencia candidato.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import EngineExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

_OUTPUT_COLUMNS = ["sequence", "sequon_position", "stackglyembed_veredicto", "stackglyembed_score"]

# Secuon canonico de N-glicosilacion: Asn - cualquiera menos Prolina - Ser/Thr.
# Solapado: 'NPNSTPNST' debe reportar la N en la posicion 1 Y la de la
# posicion 7, asi que se escanea con una regex de lookahead en vez de
# 'finditer' plano (que saltaria de largo tras cada match, perdiendo
# secuones que empiezan dentro de la ventana del anterior).
_SEQUON_PATTERN = re.compile(r"(?=(N[^P][ST]))")


def scan_sequons(sequence: str) -> List[int]:
    """Escanea secuones N-X-[S/T] (X != Prolina) en ``sequence``.

    Returns:
        Posiciones 1-indexadas de la Asparagina (N) de cada secuon encontrado,
        incluyendo secuones solapados.
    """
    return [m.start() + 1 for m in _SEQUON_PATTERN.finditer(sequence)]


def _resolve_binary() -> None:
    """Valida que el interprete, el script propio y los pickles del clon externo existan."""
    python_bin = Path(Settings.STACKGLYEMBED_PYTHON_BIN)
    script = Path(Settings.STACKGLYEMBED_SCRIPT_PATH)
    models_dir = Path(Settings.STACKGLYEMBED_MODELS_DIR)
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de StackGlyEmbed en '{python_bin}'. "
            "Ver README.md (Seccion 11) o apunta STACKGLYEMBED_PYTHON_BIN a la ubicacion correcta."
        )
    if not script.is_file():
        raise EngineExecutionError(
            f"No se encontro el script 'stackglyembed_predict_local.py' en '{script}'. "
            "Ver README.md (Seccion 11) o apunta STACKGLYEMBED_SCRIPT_PATH a la ubicacion correcta."
        )
    if not (models_dir / "base_layer_pickle_files" / "SVM_meta_layer.sav").is_file():
        raise EngineExecutionError(
            f"No se encontraron los pickles del clasificador de StackGlyEmbed en '{models_dir}'. "
            "Este clon externo debe descargarse aparte (ver README.md, Seccion 11), apunta "
            "STACKGLYEMBED_MODELS_DIR a su carpeta 'prediction/' si esta en otra ubicacion."
        )


def predict_nglycosylation(sequences: List[str], output_dir: Path, filename_prefix: str = "") -> pd.DataFrame:
    """Evalua N-glicosilacion en cada secuon candidato de ``sequences`` con StackGlyEmbed local.

    Args:
        sequences: Peptidos/secuencias candidatos a evaluar. Los que no
            contienen ningun secuon N-X-[S/T] se omiten (no producen ninguna
            fila). Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir ``features.csv``/``predicted_values.csv``
            crudos, para trazabilidad.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para los
            archivos crudos persistidos en ``output_dir``.

    Returns:
        DataFrame con una fila por secuon candidato (columnas ``sequence``,
        ``sequon_position`` -1-indexada, posicion de la N-, ``stackglyembed_veredicto``
        -``'Glicosilado'``/``'No glicosilado'``- y ``stackglyembed_score``
        -probabilidad cruda del meta-clasificador SVM-).

    Raises:
        EngineExecutionError: Si el venv/script no esta instalado, el
            subproceso falla/excede el timeout, o la salida no tiene el
            formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    _resolve_binary()

    entries: List[Tuple[str, List[int]]] = [
        (seq, scan_sequons(seq)) for seq in sequences if scan_sequons(seq)
    ]
    if not entries:
        logger.info("Ningun peptido de entrada contiene un secuon N-X-[S/T] (X != Prolina): nada que evaluar.")
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(Settings.STACKGLYEMBED_SCRIPT_PATH)

    with tempfile.TemporaryDirectory(prefix="stackglyembed_") as tmp:
        tmp_dir = Path(tmp)
        dataset_path = tmp_dir / "dataset.txt"
        with dataset_path.open("w", encoding="utf-8") as fh:
            for i, (seq, positions) in enumerate(entries):
                fh.write(f"candidato_{i}," + ",".join(str(p) for p in positions) + "\n")
                fh.write(seq + "\n")

        run_output_dir = tmp_dir / "out"
        cmd = [
            Settings.STACKGLYEMBED_PYTHON_BIN, str(script_path),
            "--dataset", str(dataset_path), "--output-dir", str(run_output_dir),
            "--models-dir", Settings.STACKGLYEMBED_MODELS_DIR,
            "--t5-model-path", Settings.STACKGLYEMBED_T5_MODEL_PATH,
            "--esm-model-name", Settings.STACKGLYEMBED_ESM_MODEL_NAME,
        ]
        logger.info("Ejecutando StackGlyEmbed local: %s", " ".join(cmd))
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=Settings.STACKGLYEMBED_TIMEOUT_SECONDS, cwd=script_path.parent, env=env)
        except subprocess.CalledProcessError as exc:
            raise EngineExecutionError(
                f"StackGlyEmbed termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(
                f"StackGlyEmbed excedio el tiempo limite de {Settings.STACKGLYEMBED_TIMEOUT_SECONDS}s."
            ) from exc

        predicted_path = run_output_dir / "predicted_values.csv"
        if not predicted_path.is_file():
            raise EngineExecutionError(
                f"StackGlyEmbed termino sin error pero no genero el archivo esperado en '{predicted_path}'."
            )

        predicted = pd.read_csv(predicted_path)
        n_sites = sum(len(positions) for _, positions in entries)
        if len(predicted) != n_sites:
            raise EngineExecutionError(
                f"StackGlyEmbed devolvio {len(predicted)} prediccion(es), se esperaban {n_sites} "
                f"(una por secuon candidato)."
            )

        features_src = run_output_dir / "features.csv"
        if features_src.is_file():
            (output_dir / f"{filename_prefix}stackglyembed_features.csv").write_bytes(features_src.read_bytes())
        (output_dir / f"{filename_prefix}stackglyembed_raw.csv").write_bytes(predicted_path.read_bytes())

        rows = []
        idx = 0
        for seq, positions in entries:
            for pos in positions:
                pred_row = predicted.iloc[idx]
                rows.append(
                    {
                        "sequence": seq,
                        "sequon_position": pos,
                        "stackglyembed_veredicto": "Glicosilado" if int(pred_row["prediction"]) == 1 else "No glicosilado",
                        "stackglyembed_score": float(pred_row["probability"]),
                    }
                )
                idx += 1

    return pd.DataFrame(rows)[_OUTPUT_COLUMNS]


# Resaltado ANSI (negrita + amarillo) del sequon N-X-[S/T] dentro de la
# columna Secuencia, mismo codigo de color que usa
# ``netmhciipan_engine.print_traceback_table`` para el nucleo de union MHC
# (consistencia visual entre fases, sin acoplar ambos modulos).
_SEQUON_ANSI_START = "\033[1;33m"
_SEQUON_ANSI_END = "\033[0m"


def _highlight_sequon(line: str, row: Any) -> str:
    """Inyecta el resaltado del sequon en ``line`` (fila ya formateada con padding).

    ``sequon_position`` es 1-indexado y marca la 'N' del motivo (ver
    ``scan_sequons``): el motivo completo son los 3 residuos
    ``[sequon_position-1 : sequon_position+2]`` de ``row.sequence``. Se busca
    primero donde cae ``row.sequence`` dentro de la linea ya formateada -mismo
    motivo que en ``print_traceback_table``: insertar ANSI antes del padding
    descoloca la columna, porque los codigos de color cuentan para ``len()``
    aunque no ocupen espacio visible en la terminal.
    """
    seq_start = line.find(row.sequence)
    if seq_start == -1:
        return line
    motif_start = seq_start + (row.sequon_position - 1)
    motif_end = motif_start + 3
    return f"{line[:motif_start]}{_SEQUON_ANSI_START}{line[motif_start:motif_end]}{_SEQUON_ANSI_END}{line[motif_end:]}"


def print_glycosylation_report(report_df: pd.DataFrame) -> None:
    """Imprime el informe de N-glicosilacion: analogo a ``algpred_engine.print_allergenicity_report``.

    El sequon N-X-[S/T] evaluado se resalta en amarillo dentro de la columna
    Secuencia (ver ``_highlight_sequon``), igual que el nucleo de union MHC en
    ``netmhciipan_engine.print_traceback_table``.
    """
    if report_df.empty:
        print("Ningun peptido 'Seguro' contiene un sequon N-X-[S/T] (X != Prolina): nada que reportar.")
        return

    seq_width = max(30, report_df["sequence"].str.len().max() + 2)
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("Sequon (pos)", lambda r: str(r.sequon_position), 12, ">"),
        Column("Score", lambda r: f"{r.stackglyembed_score:.4f}", 10, ">"),
        Column("Veredicto", lambda r: r.stackglyembed_veredicto, 16, ">"),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns, line_formatter=_highlight_sequon)

    n_glyco = int((report_df["stackglyembed_veredicto"] == "Glicosilado").sum())
    n_non_glyco = int((report_df["stackglyembed_veredicto"] == "No glicosilado").sum())
    print(f"\nResumen Fase 4c: {n_non_glyco} no glicosilado(s) / {n_glyco} glicosilado(s) predicho(s).")
