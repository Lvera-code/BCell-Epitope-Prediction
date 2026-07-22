"""Prediccion de cleavage MHC-I (NetCleave LOCAL, venv dedicado, subprocess puro).

Wrapper 100% local sobre ``NetCleave.py`` (Amengual-Rigo & Guallar, MIT),
mismo patron que ``bepipred_engine.py``. Usa el modelo pre-entrenado
localmente ya incluido en el repo (``data/models/I_mass-spectrometry_HLA/``,
entrenado sobre datos de IEDB/UniProt/UniParc descargados una sola vez, ver
``scipion-chem-netcleave/netcleave_src/data/databases/``): NUNCA se
reentrena en runtime del pipeline, solo se usa ``--score_fasta`` contra el
modelo ya existente en disco.

Detalle no obvio verificado empiricamente: ``NetCleave.py`` resuelve la ruta
del modelo a cargar como ``data/models/{mhc_class}_{technique}_{mhc_family}``,
RELATIVA al directorio de trabajo del proceso (no a ``package_dir``), asi que
el subprocess se invoca siempre con ``cwd=<carpeta de NetCleave.py>``. El
modelo bundled localmente fue entrenado con ``--mhc_family HLA`` (generico,
no ``HLA-A*02:01`` que es el default del script): hay que pasar
``--mhc_family HLA`` explicitamente o el path resuelto no encuentra el
modelo bundled.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import EngineExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

_OUTPUT_COLUMNS = [
    "sequence_window", "cleavage_position", "cleavage_residue", "cleavage_score", "source_sequence",
]

# Modelo pre-entrenado local ya presente en el repo (mass-spectrometry, MHC-I,
# familia HLA generica). Debe coincidir EXACTO con el nombre de carpeta bajo
# 'data/models/' o NetCleave no lo encuentra (ver docstring del modulo).
_MHC_CLASS = "I"
_TECHNIQUE = "mass_spectrometry"
_MHC_FAMILY = "HLA"


def _resolve_binary() -> None:
    """Valida que el interprete y el script de NetCleave existan."""
    python_bin = Path(Settings.NETCLEAVE_PYTHON_BIN)
    script = Path(Settings.NETCLEAVE_SCRIPT_PATH)
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de NetCleave en '{python_bin}'. "
            "Ver README (Seccion de Instalacion) o apunta NETCLEAVE_PYTHON_BIN a la ubicacion correcta."
        )
    if not script.is_file():
        raise EngineExecutionError(
            f"No se encontro el script 'NetCleave.py' en '{script}'. "
            "Ver README (Seccion de Instalacion) o apunta NETCLEAVE_SCRIPT_PATH a la ubicacion correcta."
        )
    model_dir = script.parent / "data" / "models" / f"{_MHC_CLASS}_{_TECHNIQUE.replace('_', '-')}_{_MHC_FAMILY}"
    if not model_dir.is_dir():
        raise EngineExecutionError(
            f"No se encontro el modelo pre-entrenado local de NetCleave en '{model_dir}'. "
            "Este modelo viene bundled con el repo (no requiere reentrenamiento); si falta, "
            "revisa que 'scipion-chem-netcleave/netcleave_src/data/models/' este completo."
        )


def predict_cleavage(sequences: List[str], output_dir: Path, filename_prefix: str = "") -> pd.DataFrame:
    """Predice sitios de cleavage MHC-I (C-terminal) con NetCleave local.

    Args:
        sequences: Secuencias/fragmentos a evaluar (idealmente >= 7 aa: NetCleave
            necesita una ventana de 4+3 residuos alrededor de cada corte
            candidato, fragmentos mas cortos no producen ninguna fila). Vacio
            -> DataFrame vacio.
        output_dir: Carpeta donde persistir el .xlsx crudo devuelto por NetCleave.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para el .xlsx crudo.

    Returns:
        DataFrame con columnas ``sequence_window`` (ventana de 7 residuos
        alrededor del corte, formato ``XXXX|XXX``), ``cleavage_position``
        (1-indexado, residuo INMEDIATAMENTE DESPUES del corte),
        ``cleavage_residue``, ``cleavage_score`` (0-1, score crudo de
        NetCleave, mayor = corte mas probable).

    Raises:
        EngineExecutionError: Si el venv/script/modelo no esta instalado, el
            subproceso falla/excede el timeout, o la salida no tiene el
            formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    _resolve_binary()
    script_path = Path(Settings.NETCLEAVE_SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_frames = []
    with tempfile.TemporaryDirectory(prefix="netcleave_") as tmp:
        tmp_dir = Path(tmp)
        for i, seq in enumerate(sequences):
            fasta_path = tmp_dir / f"seq_{i}.fasta"
            fasta_path.write_text(f">candidato_{i}\n{seq}\n", encoding="utf-8")

            cmd = [
                Settings.NETCLEAVE_PYTHON_BIN, str(script_path),
                "--mhc_class", _MHC_CLASS, "--technique", _TECHNIQUE, "--mhc_family", _MHC_FAMILY,
                "--score_fasta", str(fasta_path),
            ]
            logger.info("Ejecutando NetCleave local (candidato %d/%d): %s", i + 1, len(sequences), " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True,
                                timeout=Settings.NETCLEAVE_TIMEOUT_SECONDS, cwd=script_path.parent)
            except subprocess.CalledProcessError as exc:
                raise EngineExecutionError(
                    f"NetCleave termino con exit code {exc.returncode} en el candidato {i}: "
                    f"{(exc.stderr or '<sin stderr>')[:2000]}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise EngineExecutionError(
                    f"NetCleave excedio el tiempo limite de {Settings.NETCLEAVE_TIMEOUT_SECONDS}s en el candidato {i}."
                ) from exc

            # NetCleave nombra el .xlsx de salida como
            # '<fasta_stem>_<primer_token_del_header_fasta>_NetCleave.xlsx'
            # (ver predict_csv_or_fasta.score_set: usa el nombre del registro
            # FASTA, no solo el stem del archivo) -- verificado leyendo el
            # codigo fuente, no documentado en --help. Se busca por glob en
            # vez de reconstruir el nombre exacto, para no depender de ese
            # detalle interno si cambia entre versiones.
            xlsx_matches = list(tmp_dir.glob(f"seq_{i}_*_NetCleave.xlsx"))
            if not xlsx_matches:
                raise EngineExecutionError(
                    f"NetCleave termino sin error pero no genero ningun .xlsx esperado "
                    f"('seq_{i}_*_NetCleave.xlsx') en '{tmp_dir}' (candidato {i})."
                )
            xlsx_path = xlsx_matches[0]
            raw = pd.read_excel(xlsx_path)
            expected_cols = {"Cleavage site", "Cleavage site after position", "Cleavage site after residue",
                              "Cleavage site prediction score"}
            if not expected_cols.issubset(raw.columns):
                raise EngineExecutionError(
                    f"El formato del .xlsx de NetCleave no coincide con lo esperado (candidato {i}). "
                    f"Columnas encontradas: {list(raw.columns)}."
                )
            frame = pd.DataFrame(
                {
                    "sequence_window": raw["Cleavage site"],
                    "cleavage_position": raw["Cleavage site after position"],
                    "cleavage_residue": raw["Cleavage site after residue"],
                    "cleavage_score": raw["Cleavage site prediction score"],
                    "source_sequence": seq,
                }
            )
            result_frames.append(frame)

        if result_frames:
            combined = pd.concat(result_frames, ignore_index=True)
            combined.to_csv(output_dir / f"{filename_prefix}netcleave_raw.csv", index=False)
            return combined[_OUTPUT_COLUMNS]

    return pd.DataFrame(columns=_OUTPUT_COLUMNS)


def annotate_cterm_cleavage(traceback_df: pd.DataFrame, cleavage_df: pd.DataFrame) -> pd.DataFrame:
    """Anota un traceback de Fase 5/5b con evidencia de corte proteasomal en el C-terminal del candidato.

    Senal complementaria, no un filtro: un peptido puede bindear MHC-I/II con
    fuerza pero nunca generarse via procesamiento antigenico real si el
    proteasoma no corta exactamente donde termina el nucleo de union. Requiere
    que ``cleavage_df`` (salida de ``predict_cleavage``) se haya corrido sobre
    el/los peptido(s) ORIGINALES de ``safe_df`` (no solo el nucleo aceptado),
    para conservar el contexto de flanco que NetCleave necesita alrededor de
    cada corte candidato.

    Coincidencia exigida (no solo "hay algun corte en la region"): dado un
    candidato cuya secuencia evaluada (``traceback_df['sequence_f5']``) es
    substring de un ``cleavage_df['source_sequence']``, se busca una fila de
    ``cleavage_df`` cuyo ``cleavage_position`` caiga EXACTO un residuo despues
    del ultimo residuo del candidato dentro de esa secuencia origen
    (``cleavage_position`` es 1-indexado y ya representa el residuo
    inmediatamente posterior al corte, ver docstring de ``predict_cleavage``).

    Args:
        traceback_df: Salida de ``build_traceback_report`` (netmhciipan_engine
            o netmhcpan_engine), con columna ``sequence_f5``.
        cleavage_df: Salida de ``predict_cleavage`` corrida sobre los mismos
            peptidos 'Seguros' de la Fase 4, con columna ``source_sequence``.

    Returns:
        Copia de ``traceback_df`` con dos columnas nuevas: ``netcleave_c_term_match``
        (bool) y ``netcleave_c_term_score`` (score crudo del corte, NA si no hubo match).
    """
    result = traceback_df.copy()
    result["netcleave_c_term_match"] = False
    result["netcleave_c_term_score"] = pd.NA

    if traceback_df.empty or cleavage_df.empty:
        return result

    for idx, row in traceback_df.iterrows():
        candidates = cleavage_df[cleavage_df["source_sequence"].str.contains(row["sequence_f5"], regex=False, na=False)]
        if candidates.empty:
            continue

        best_score = None
        for cleavage_row in candidates.itertuples(index=False):
            offset = cleavage_row.source_sequence.find(row["sequence_f5"])
            c_term_position = offset + len(row["sequence_f5"]) + 1
            if cleavage_row.cleavage_position == c_term_position:
                if best_score is None or cleavage_row.cleavage_score > best_score:
                    best_score = cleavage_row.cleavage_score

        if best_score is not None:
            result.at[idx, "netcleave_c_term_match"] = True
            result.at[idx, "netcleave_c_term_score"] = best_score

    return result
