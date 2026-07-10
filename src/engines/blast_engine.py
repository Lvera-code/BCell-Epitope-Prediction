"""Fase 4: Filtro de tolerancia inmunologica via BLASTp local contra el proteoma humano.

Requisito de entorno: este modulo NUNCA llama a un servicio remoto de BLAST.
Depende exclusivamente de un binario 'blastp' local (paquete NCBI BLAST+) y de
una base de datos ya indexada con 'makeblastdb', cuya ruta se lee de
``Settings.BLAST_HUMAN_DB`` (variable de entorno ``BLAST_HUMAN_DB``, nunca
hardcodeada). Si el binario o la base de datos faltan, se lanza
``BlastExecutionError`` con instrucciones accionables -exactamente el mismo
patron de salida elegante que ``BepiPredEngine`` usa en la Fase 2- en vez de
fallar con una traza opaca.

Seleccion dinamica de algoritmo (por peptido): BLAST recomienda un modo de
busqueda distinto segun la longitud de la secuencia consultada. Cada peptido
de ``epitopes_df`` (Fase 3) se evalua individualmente:

* longitud < ``Settings.BLAST_SHORT_PEPTIDE_MAX_LEN`` (30 aa por defecto) ->
  ``-task blastp-short`` (word_size y matriz de sustitucion ajustados para
  secuencias cortas).
* longitud >= ese umbral -> ``-task blastp`` (algoritmo estandar).

Los peptidos se agrupan por tarea y cada grupo se ejecuta en su propia
invocacion de ``subprocess.run`` (un unico comando no puede mezclar dos
``-task`` distintos), y los resultados se combinan al final.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import BlastExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Columnas del formato tabular '-outfmt 6' de BLAST, en orden fijo.
_OUTFMT6_COLUMNS: List[str] = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
]


def _check_blast_environment(db_path: Path) -> None:
    """Valida que 'blastp' este en el PATH y que la base de datos exista.

    Raises:
        BlastExecutionError: Con instrucciones de instalacion si falta el
            binario o la base de datos local del proteoma humano (incluyendo
            un recordatorio de la variable de entorno ``BLAST_HUMAN_DB``).
    """
    if shutil.which("blastp") is None:
        raise BlastExecutionError(
            "El binario 'blastp' no esta disponible en el PATH. Instala NCBI BLAST+ "
            "(ver README.md - Seccion de Instalacion) y vuelve a intentarlo."
        )

    if not Path(f"{db_path}.phr").is_file():
        raise BlastExecutionError(
            f"No se encontro la base de datos BLAST del proteoma humano en '{db_path}' "
            f"(falta '{db_path}.phr'). Define la variable de entorno BLAST_HUMAN_DB "
            "apuntando al prefijo de una base de datos ya indexada con 'makeblastdb', "
            "o descarga el proteoma de Homo sapiens e indexalo en la ruta por defecto "
            "'reference_db/human_proteome_db' (ver README.md - Seccion de Instalacion) "
            "antes de correr la Fase 4."
        )


def _select_task(sequence_length: int, short_max_len: int = Settings.BLAST_SHORT_PEPTIDE_MAX_LEN) -> str:
    """Selecciona dinamicamente el algoritmo de BLASTp segun la longitud del peptido.

    Args:
        sequence_length: Longitud (aa) del peptido candidato.
        short_max_len: Umbral (exclusivo) por debajo del cual se usa
            ``blastp-short`` en vez de ``blastp``.

    Returns:
        ``"blastp-short"`` si ``sequence_length < short_max_len``, si no ``"blastp"``.
    """
    return "blastp-short" if sequence_length < short_max_len else "blastp"


def _run_blastp_batch(
    records: List[Tuple[int, str]], task: str, db: Path, evalue: float
) -> pd.DataFrame:
    """Ejecuta, via ``subprocess.run``, un lote homogeneo de peptidos con un mismo '-task'.

    Args:
        records: Lista de ``(indice_original, secuencia)`` a consultar.
        task: ``"blastp-short"`` o ``"blastp"`` (todos los ``records`` deben
            corresponder a la misma tarea; el llamador se encarga de agrupar).
        db: Prefijo de la base de datos BLAST local.
        evalue: E-value usado en la busqueda.

    Returns:
        DataFrame en formato ``-outfmt 6`` con los hits encontrados (vacio si
        ``records`` esta vacio o BLAST no encontro ningun hit).

    Raises:
        BlastExecutionError: Si el proceso termina con error o excede el timeout.
    """
    if not records:
        return pd.DataFrame(columns=_OUTFMT6_COLUMNS)

    with tempfile.TemporaryDirectory(prefix=f"blastp_{task}_") as tmp:
        tmp_dir = Path(tmp)
        query_path = tmp_dir / "candidates.fasta"
        out_path = tmp_dir / "blast_results.tsv"

        with query_path.open("w", encoding="utf-8") as fh:
            for idx, seq in records:
                fh.write(f">peptide_{idx}\n{seq}\n")

        cmd = [
            "blastp", "-task", task,
            "-query", str(query_path), "-db", str(db),
            "-outfmt", "6", "-evalue", str(evalue),
            "-out", str(out_path),
        ]
        logger.info("Ejecutando BLASTp (-task %s) sobre %d peptido(s): %s", task, len(records), " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        except subprocess.CalledProcessError as exc:
            raise BlastExecutionError(
                f"blastp (-task {task}) termino con exit code {exc.returncode}: "
                f"{(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BlastExecutionError(f"blastp (-task {task}) excedio el tiempo limite de 300s.") from exc

        if out_path.stat().st_size == 0:
            return pd.DataFrame(columns=_OUTFMT6_COLUMNS)
        return pd.read_csv(out_path, sep="\t", names=_OUTFMT6_COLUMNS)


def run_blastp_filter(
    epitopes_df: pd.DataFrame,
    db_path: str = Settings.BLAST_HUMAN_DB,
    identity_threshold: float = Settings.BLAST_IDENTITY_THRESHOLD,
    evalue: float = Settings.BLAST_EVALUE,
) -> pd.DataFrame:
    """Ejecuta BLASTp local sobre cada peptido de ``epitopes_df`` y anota tolerancia.

    Cada peptido se enruta dinamicamente a ``blastp-short`` o ``blastp`` segun
    su longitud (ver :func:`_select_task`), en lotes separados por tarea.

    Args:
        epitopes_df: Salida de la Fase 3 (``extract_epitopes``), debe contener
            una columna ``sequence`` con el peptido candidato de cada fila.
        db_path: Prefijo (sin extension) de la base de datos BLAST local
            (por defecto, ``Settings.BLAST_HUMAN_DB``).
        identity_threshold: Porcentaje de identidad (exclusivo) por encima del
            cual un peptido se descarta por riesgo de autoinmunidad.
        evalue: E-value usado en ambas busquedas.

    Returns:
        Copia de ``epitopes_df`` con columnas nuevas: ``blast_task`` (tarea
        usada para ese peptido), ``max_pident`` (maxima identidad encontrada
        contra el proteoma humano, 0.0 si no hubo hits) y ``status``
        (``"Segura"`` o ``"Autoinmunidad"``).

    Raises:
        BlastExecutionError: Si 'blastp' no esta instalado, la base de datos
            no existe, o algun subproceso termina con error o timeout.
    """
    db = Path(db_path)
    _check_blast_environment(db)

    if epitopes_df.empty:
        return epitopes_df.assign(
            blast_task=pd.Series(dtype=str), max_pident=pd.Series(dtype=float), status=pd.Series(dtype=str)
        )

    result = epitopes_df.reset_index(drop=True).copy()
    result["blast_task"] = result["sequence"].str.len().apply(_select_task)

    short_records = [
        (idx, seq) for idx, seq, task in zip(result.index, result["sequence"], result["blast_task"])
        if task == "blastp-short"
    ]
    long_records = [
        (idx, seq) for idx, seq, task in zip(result.index, result["sequence"], result["blast_task"])
        if task == "blastp"
    ]

    hits_short = _run_blastp_batch(short_records, "blastp-short", db, evalue)
    hits_long = _run_blastp_batch(long_records, "blastp", db, evalue)
    hits = pd.concat([hits_short, hits_long], ignore_index=True)

    max_identity_per_query = hits.groupby("qseqid")["pident"].max() if not hits.empty else pd.Series(dtype=float)

    result["max_pident"] = [
        float(max_identity_per_query.get(f"peptide_{idx}", 0.0)) for idx in result.index
    ]
    result["status"] = result["max_pident"].apply(
        lambda pid: "Autoinmunidad" if pid > identity_threshold else "Segura"
    )
    return result


def print_blast_report(blast_df: pd.DataFrame) -> None:
    """Imprime el informe de tolerancia inmunologica: seguras vs. autoinmunidad."""
    if blast_df.empty:
        print("No hay peptidos candidatos de la Fase 3 para evaluar contra el proteoma humano.")
        return

    seq_width = max(30, blast_df["sequence"].str.len().max() + 2)
    header = f"{'Secuencia':<{seq_width}}{'Tarea BLAST':<14}{'Identidad max (%)':>18}{'Veredicto':>16}"
    print(header)
    print("-" * len(header))
    for row in blast_df.itertuples(index=False):
        print(f"{row.sequence:<{seq_width}}{row.blast_task:<14}{row.max_pident:>18.2f}{row.status:>16}")

    n_safe = int((blast_df["status"] == "Segura").sum())
    n_rejected = int((blast_df["status"] == "Autoinmunidad").sum())
    print(f"\nResumen Fase 4: {n_safe} segura(s) / {n_rejected} rechazada(s) por homologia con el proteoma humano.")
