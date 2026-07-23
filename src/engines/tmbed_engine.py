"""Enmascarado transmembrana/peptido senal via TMbed LOCAL (venv dedicado, subprocess puro).

Wrapper 100% local sobre el CLI de ``tmbed`` (Bernhofer & Rost, 2022,
Apache-2.0), mismo patron de subprocess que el resto de motores. Reusa el
mismo venv/pesos ya instalados para el plugin Scipion ``scipion-chem-tmbed``
(repo hermano, ``TMBED_PYTHON_BIN``/``TMBED_MODEL_DIR``), pero SIN importar
ningun modulo de ese plugin (depende de ``pwchem``, no instalado en el venv
principal de este pipeline) -- el parseo del formato de salida de TMbed
(``extract_masking_regions``/``parse_predictions`` de
``scipion-chem-tmbed/tmbed/utils/tmbed.py``) se reimplementa aqui en forma
pura de pandas/stdlib, misma logica verificada por los tests de ese plugin.

Proposito en Fase 3b (ver ``pipeline.py``): correr sobre la secuencia
COMPLETA de cada accession (no por peptido candidato, a diferencia de
Fase 4b/4c) y descartar de la union anotada de Fase 3 cualquier region que
caiga dentro de una helice/tira transmembrana o del peptido senal
N-terminal, ANTES de BLASTp (Fase 4) -- esos residuos no son accesibles a
anticuerpos en la proteina madura/anclada a membrana (o, en el caso del
peptido senal, se escinden y no forman parte de la proteina madura).
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import EngineExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

_REGIONS_COLUMNS = ["accession", "start", "end", "type"]

# Letras de clase de 'tmbed predict --out-format 1' que se convierten en
# region de enmascarado. 'i'/'o' (residuo no-membrana, adentro/afuera) se
# dejan sin tocar -- ya se presumen accesibles al solvente.
_MASKED_CLASS_TYPES = {
    "B": "TM_beta_strand",
    "H": "TM_alpha_helix",
    "S": "signal_peptide",
}


def _resolve_binary() -> Path:
    """Valida que el interprete, el binario y los pesos de TMbed existan."""
    python_bin = Path(Settings.TMBED_PYTHON_BIN)
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de TMbed en '{python_bin}'. "
            "Ver scipion-chem-tmbed/README.rst o apunta TMBED_PYTHON_BIN a la ubicacion correcta."
        )
    binary = python_bin.parent / Settings.TMBED_BINARY_NAME
    if not binary.is_file():
        raise EngineExecutionError(
            f"No se encontro el ejecutable '{binary}'. Ver scipion-chem-tmbed/README.rst o "
            "reinstala TMbed en el venv de TMBED_PYTHON_BIN ('pip install tmbed')."
        )
    model_dir = Path(Settings.TMBED_MODEL_DIR)
    if not (model_dir / "config.json").is_file():
        raise EngineExecutionError(
            f"No se encontro 'config.json' dentro de '{model_dir}'. Los pesos ProtT5-XL-U50 "
            "(~2.4 GB) no se descargan en tiempo de ejecucion (politica local-only de este "
            "proyecto): descargalos manualmente y apunta TMBED_MODEL_DIR a la carpeta que los "
            "contiene. Ver scipion-chem-tmbed/README.rst."
        )
    return binary


def _parse_predictions(pred_path: Path) -> Dict[str, str]:
    """Parsea el formato de 3 lineas por proteina de 'tmbed predict --out-format 1'.

    Cada proteina ocupa 3 lineas no vacias: cabecera FASTA, secuencia y una
    cadena de igual longitud con una letra de clase por residuo. Mismo
    contrato verificado empiricamente por los tests de
    ``scipion-chem-tmbed/tmbed/utils/tmbed.py::parse_predictions``.
    """
    lines = [line.rstrip("\n") for line in pred_path.read_text().splitlines() if line.strip()]

    if not lines or len(lines) % 3 != 0:
        raise EngineExecutionError(
            f"La salida de TMbed en '{pred_path}' no tiene el formato esperado de 3 lineas por "
            f"proteina (cabecera/secuencia/prediccion): se encontraron {len(lines)} linea(s) no vacia(s)."
        )

    classes_by_header: Dict[str, str] = {}
    for i in range(0, len(lines), 3):
        header, sequence, classes = lines[i], lines[i + 1], lines[i + 2]
        if not header.startswith(">"):
            raise EngineExecutionError(
                f"Se esperaba una cabecera FASTA en la linea {i + 1} de '{pred_path}', se obtuvo: '{header}'."
            )
        if len(sequence) != len(classes):
            raise EngineExecutionError(
                f"Desfase secuencia/prediccion para '{header}' en '{pred_path}': "
                f"{len(sequence)} residuo(s) vs {len(classes)} letra(s) de clase."
            )
        classes_by_header[header[1:].strip()] = classes

    return classes_by_header


def _extract_regions(classes: str, min_length: int) -> List[Tuple[int, int, str]]:
    """Colapsa una cadena de clase por residuo en regiones de enmascarado contiguas (1-indexado, inclusivo)."""
    regions: List[Tuple[int, int, str]] = []
    cur_type = None
    start = None
    n = len(classes)
    for i in range(n):
        roi_type = _MASKED_CLASS_TYPES.get(classes[i])
        pos = i + 1
        if roi_type != cur_type:
            if cur_type is not None and (pos - start) >= min_length:
                regions.append((start, pos - 1, cur_type))
            cur_type = roi_type
            start = pos if roi_type is not None else None
    if cur_type is not None and (n + 1 - start) >= min_length:
        regions.append((start, n, cur_type))
    return regions


def predict_tm_signal_regions(
    sequence_by_accession: Dict[str, str], output_dir: Path, filename_prefix: str = ""
) -> pd.DataFrame:
    """Evalua transmembrana/peptido senal con TMbed local sobre la secuencia COMPLETA de cada accession.

    Args:
        sequence_by_accession: ``accession -> secuencia completa``. Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir la salida cruda de TMbed, para trazabilidad.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para el archivo crudo.

    Returns:
        DataFrame con columnas ``accession``, ``start``, ``end`` (1-indexado,
        inclusivo), ``type`` (``'TM_beta_strand'``, ``'TM_alpha_helix'`` o
        ``'signal_peptide'``) -- una fila por region contigua detectada.
        Vacio si TMbed no detecto ninguna region en ninguna accession.

    Raises:
        EngineExecutionError: Si el venv/pesos no estan instalados, el
            subproceso falla/excede el timeout, o la salida no tiene el
            formato esperado.
    """
    if not sequence_by_accession:
        return pd.DataFrame(columns=_REGIONS_COLUMNS)

    binary = _resolve_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tmbed_") as tmp:
        tmp_dir = Path(tmp)
        fasta_path = tmp_dir / "input.fasta"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for accession, sequence in sequence_by_accession.items():
                fh.write(f">{accession}\n{sequence}\n")

        pred_path = tmp_dir / "predictions.pred"
        cmd = [
            str(binary), "predict",
            "--fasta", str(fasta_path), "--predictions", str(pred_path),
            "--out-format", Settings.TMBED_OUT_FORMAT,
            "--model-dir", Settings.TMBED_MODEL_DIR,
            "--threads", str(Settings.TMBED_THREADS),
            "--use-gpu" if Settings.TMBED_USE_GPU else "--no-use-gpu",
        ]
        logger.info("Ejecutando TMbed local: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=Settings.TMBED_TIMEOUT_SECONDS)
        except subprocess.CalledProcessError as exc:
            raise EngineExecutionError(
                f"TMbed termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(f"TMbed excedio el tiempo limite de {Settings.TMBED_TIMEOUT_SECONDS}s.") from exc

        if not pred_path.is_file():
            raise EngineExecutionError(f"TMbed termino sin error pero no genero '{pred_path}'.")

        classes_by_header = _parse_predictions(pred_path)

        raw_copy_path = output_dir / f"{filename_prefix}tmbed_raw.pred"
        raw_copy_path.write_bytes(pred_path.read_bytes())

    rows = [
        {"accession": accession, "start": start, "end": end, "type": region_type}
        for accession, classes in classes_by_header.items()
        for start, end, region_type in _extract_regions(classes, Settings.TMBED_MIN_REGION_LENGTH)
    ]
    return pd.DataFrame(rows, columns=_REGIONS_COLUMNS)


def filter_overlapping_regions(union_df: pd.DataFrame, regions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Descarta de ``union_df`` las filas cuyo rango ``[start, end]`` se solapa con una region TMbed de la misma accession.

    Args:
        union_df: Union anotada de Fase 3 (columnas ``accession``/``start``/``end``, 1-indexado inclusivo).
        regions_df: Salida de :func:`predict_tm_signal_regions` (mismo esquema de coordenadas).

    Returns:
        Tupla ``(kept_df, discarded_df)``. ``kept_df`` conserva todas las
        columnas originales de ``union_df``, con el mismo orden de filas
        restantes. ``discarded_df`` tiene columnas ``accession``/``start``/
        ``end``/``type`` -- una fila por cada region TMbed con la que se
        solapo la fila descartada de ``union_df`` (una misma fila descartada
        puede solaparse con mas de una region TMbed, p. ej. un peptido que
        cubre tanto el final del signal_peptide como el inicio de la
        siguiente TM_alpha_helix, y aparece entonces una vez por cada una).
    """
    if union_df.empty or regions_df.empty:
        return union_df, pd.DataFrame(columns=["accession", "start", "end", "type"])

    def _overlapping_regions(row) -> pd.DataFrame:
        acc_regions = regions_df[regions_df["accession"] == row["accession"]]
        return acc_regions[(acc_regions["start"] <= row["end"]) & (acc_regions["end"] >= row["start"])]

    discarded_frames = []
    keep_mask = []
    for _, row in union_df.iterrows():
        overlaps = _overlapping_regions(row)
        keep_mask.append(overlaps.empty)
        if not overlaps.empty:
            discarded_frames.append(pd.DataFrame({
                "accession": row["accession"], "start": row["start"], "end": row["end"],
                "type": overlaps["type"].tolist(),
            }))

    kept = union_df[keep_mask].reset_index(drop=True)
    discarded_df = (
        pd.concat(discarded_frames, ignore_index=True) if discarded_frames
        else pd.DataFrame(columns=["accession", "start", "end", "type"])
    )
    return kept, discarded_df


def print_tmbed_regions_report(regions_df: pd.DataFrame) -> None:
    """Imprime las regiones transmembrana/peptido senal detectadas por TMbed."""
    if regions_df.empty:
        print("TMbed no detecto ninguna region transmembrana/peptido senal.")
        return

    columns = [
        Column("accession", lambda r: r.accession, 28, "<"),
        Column("start", lambda r: str(r.start), 7, ">"),
        Column("end", lambda r: str(r.end), 7, ">"),
        # prefix="  ": "end" es right-aligned (">"), sin este separador
        # explicito queda pegada a "tipo" (mismo caso que signalp_engine.py).
        Column("tipo", lambda r: r.type, 18, "<", prefix="  "),
    ]
    display_df = regions_df.sort_values(["accession", "start"]).reset_index(drop=True)
    print_fixed_width_table(display_df.itertuples(index=False), columns, group_by=lambda r: r.accession)
    print(f"\nTotal: {len(regions_df)} region(es) transmembrana/peptido senal en {regions_df['accession'].nunique()} accession(es).")


def print_discarded_regions_report(discarded_df: pd.DataFrame) -> None:
    """Imprime las regiones de la union anotada descartadas en Fase 3b por solaparse con una region TM/senal.

    ``discarded_df`` es la salida de :func:`filter_overlapping_regions`: una
    fila por cada (region descartada, region TMbed con la que se solapo) --
    ``tipo`` explica el motivo puntual del descarte, no solo el conteo.
    """
    if discarded_df.empty:
        return

    columns = [
        Column("accession", lambda r: r.accession, 28, "<"),
        Column("start", lambda r: str(r.start), 7, ">"),
        Column("end", lambda r: str(r.end), 7, ">"),
        Column("tipo", lambda r: r.type, 18, "<", prefix="  "),
    ]
    display_df = discarded_df.sort_values(["accession", "start"]).reset_index(drop=True)
    print_fixed_width_table(display_df.itertuples(index=False), columns, group_by=lambda r: r.accession)
