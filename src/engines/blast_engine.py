"""Fase 4: Filtro de tolerancia inmunologica via BLASTp local contra el proteoma humano.

Requisito de entorno: este modulo NUNCA llama a un servicio remoto de BLAST.
Depende exclusivamente de un binario 'blastp' local (paquete NCBI BLAST+) y de
una base de datos ya indexada con 'makeblastdb', cuya ruta se lee de
``Settings.BLAST_HUMAN_DB`` (variable de entorno ``BLAST_HUMAN_DB``, nunca
hardcodeada). Si el binario o la base de datos faltan, se lanza
``BlastExecutionError`` con instrucciones accionables -exactamente el mismo
patron de salida elegante que ``BepiPredEngine`` usa en la Fase 2- en vez de
fallar con una traza opaca.

Seleccion dinamica de algoritmo y E-value (por peptido): BLAST recomienda un
modo de busqueda y una sensibilidad estadistica distintos segun la longitud
de la secuencia consultada. Cada peptido de ``epitopes_df`` (Fase 3) se
evalua individualmente:

* longitud <= ``Settings.BLAST_SHORT_PEPTIDE_MAX_LEN`` (30 aa por defecto) ->
  ``-task blastp-short`` (word_size y matriz de sustitucion ajustados para
  secuencias cortas), ``evalue=Settings.BLAST_EVALUE_SHORT`` (50 por
  defecto: con el e-value estandar de blastp, un match identico de un
  peptido corto contra el proteoma humano se descarta como "no
  significativo" por pura estadistica de longitud, arruinando el filtro de
  autoinmunidad justo donde mas importa).
* ``Settings.BLAST_SHORT_PEPTIDE_MAX_LEN`` < longitud <=
  ``Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN`` (100 aa por defecto) ->
  ``-task blastp``, ``evalue=Settings.BLAST_EVALUE_MEDIUM`` (0.1 por defecto).
* longitud > ese segundo umbral -> ``-task blastp``,
  ``evalue=Settings.BLAST_EVALUE_LONG`` (0.05 por defecto: en consultas
  largas un e-value laxo generaria ruido de homologias irrelevantes).

Los peptidos se agrupan por tramo (misma tarea + mismo E-value) y cada grupo
se ejecuta en su propia invocacion de ``subprocess.run`` (un unico comando no
puede mezclar dos ``-task``/``-evalue`` distintos), y los resultados se
combinan al final.
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
from src.utils.table_format import Column, print_fixed_width_table

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
        short_max_len: Umbral (inclusive) por debajo o igual al cual se usa
            ``blastp-short`` en vez de ``blastp``.

    Returns:
        ``"blastp-short"`` si ``sequence_length <= short_max_len``, si no ``"blastp"``.
    """
    return "blastp-short" if sequence_length <= short_max_len else "blastp"


def _select_evalue(
    sequence_length: int,
    short_max_len: int = Settings.BLAST_SHORT_PEPTIDE_MAX_LEN,
    medium_max_len: int = Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN,
) -> float:
    """Selecciona dinamicamente el E-value segun el tramo de longitud del peptido.

    La estadistica de BLAST depende fuertemente de la longitud de la consulta:
    un e-value estricto (por defecto ~10 o menor) descarta como "no
    significativos" hits identicos de peptidos cortos, arruinando el filtro
    de autoinmunidad justo donde mas importa. Para consultas largas ocurre lo
    contrario: un e-value laxo generaria ruido de homologias irrelevantes.

    Args:
        sequence_length: Longitud (aa) del peptido candidato.
        short_max_len: Umbral (inclusive) del tramo "corto".
        medium_max_len: Umbral (inclusive) del tramo "intermedio".

    Returns:
        ``Settings.BLAST_EVALUE_SHORT`` si ``sequence_length <= short_max_len``;
        ``Settings.BLAST_EVALUE_MEDIUM`` si ``short_max_len < sequence_length <= medium_max_len``;
        ``Settings.BLAST_EVALUE_LONG`` en caso contrario.
    """
    if sequence_length <= short_max_len:
        return Settings.BLAST_EVALUE_SHORT
    if sequence_length <= medium_max_len:
        return Settings.BLAST_EVALUE_MEDIUM
    return Settings.BLAST_EVALUE_LONG


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


def _max_identity_by_query(
    hits: pd.DataFrame, query_lengths: pd.Series, min_query_coverage: float
) -> pd.Series:
    """Maxima identidad por query, ignorando hits cuyo alineamiento no cubre lo suficiente.

    Logica pura (sin subprocess ni binarios), separada de
    :func:`run_blastp_filter` para poder testearla sin 'blastp' instalado
    (mismo criterio del resto de este modulo, ver docstring de
    ``tests/test_blast_engine.py``).

    CONFIRMADO EMPIRICAMENTE (ver ADR en ``Settings.BLAST_MIN_QUERY_COVERAGE``):
    sin este filtro, un fragmento de 5-6 aa 100% identico dentro de un
    peptido de 14-31 aa -estadisticamente esperable por azar contra el
    proteoma humano completo, no una homologia real- contaba igual que un
    homologo autentico de longitud completa, rechazando casi cualquier
    peptido corto por "Autoinmunidad" sin importar si de verdad se parecia a
    algo humano.

    Args:
        hits: DataFrame en formato ``-outfmt 6`` (columnas ``_OUTFMT6_COLUMNS``),
            con ``qseqid`` en formato ``'peptide_{idx}'`` donde ``idx`` es un
            indice presente en ``query_lengths``.
        query_lengths: ``indice -> longitud (aa)`` del peptido consultado
            (``epitopes_df["sequence"].str.len()`` en :func:`run_blastp_filter`).
        min_query_coverage: Fraccion minima (0-1) de la longitud del peptido
            que un alineamiento debe cubrir para contar hacia la maxima
            identidad (``Settings.BLAST_MIN_QUERY_COVERAGE`` por defecto).

    Returns:
        Serie indexada por ``qseqid`` con la maxima identidad entre los hits
        que superan ``min_query_coverage``. Vacia si ``hits`` esta vacio o
        ningun hit cubre lo suficiente.
    """
    if hits.empty:
        return pd.Series(dtype=float)

    hit_query_idx = hits["qseqid"].str.replace("peptide_", "", regex=False).astype(int)
    hit_query_length = hit_query_idx.map(query_lengths)
    coverage = hits["length"] / hit_query_length
    covered_hits = hits[coverage >= min_query_coverage]

    if covered_hits.empty:
        return pd.Series(dtype=float)
    return covered_hits.groupby("qseqid")["pident"].max()


def run_blastp_filter(
    epitopes_df: pd.DataFrame,
    db_path: str = Settings.BLAST_HUMAN_DB,
    identity_threshold: float = Settings.BLAST_IDENTITY_THRESHOLD,
    min_query_coverage: float = Settings.BLAST_MIN_QUERY_COVERAGE,
) -> pd.DataFrame:
    """Ejecuta BLASTp local sobre cada peptido de ``epitopes_df`` y anota tolerancia.

    Cada peptido se enruta dinamicamente a un tramo de (``-task``, E-value)
    segun su longitud (ver :func:`_select_task` / :func:`_select_evalue`), en
    lotes separados por tramo:

    * ``<= Settings.BLAST_SHORT_PEPTIDE_MAX_LEN`` aa -> ``blastp-short``,
      ``evalue=Settings.BLAST_EVALUE_SHORT`` (laxo: la estadistica de BLAST
      penaliza a los peptidos cortos y descartaria hits identicos reales).
    * ``Settings.BLAST_SHORT_PEPTIDE_MAX_LEN`` < longitud ``<=
      Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN`` aa -> ``blastp``,
      ``evalue=Settings.BLAST_EVALUE_MEDIUM``.
    * ``> Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN`` aa -> ``blastp``,
      ``evalue=Settings.BLAST_EVALUE_LONG`` (estricto: evita ruido de
      homologias irrelevantes en consultas largas).

    Args:
        epitopes_df: Salida de la Fase 3 (``extract_epitopes``), debe contener
            una columna ``sequence`` con el peptido candidato de cada fila.
        db_path: Prefijo (sin extension) de la base de datos BLAST local
            (por defecto, ``Settings.BLAST_HUMAN_DB``).
        identity_threshold: Porcentaje de identidad (exclusivo) por encima del
            cual un peptido se descarta por riesgo de autoinmunidad.
        min_query_coverage: Fraccion minima (0-1) de la longitud del peptido
            que un alineamiento de BLAST debe cubrir para contar hacia
            ``max_pident`` (ver :func:`_max_identity_by_query` y el ADR en
            ``Settings.BLAST_MIN_QUERY_COVERAGE``): sin este filtro, un
            fragmento minusculo 100% identico (estadisticamente esperable
            por azar contra un proteoma completo) rechazaba peptidos que no
            se parecen realmente a nada humano.

    Returns:
        Copia de ``epitopes_df`` con columnas nuevas: ``blast_task`` (tarea
        usada para ese peptido), ``blast_evalue`` (E-value del tramo
        aplicado), ``max_pident`` (maxima identidad encontrada contra el
        proteoma humano, 0.0 si no hubo hits) y ``status`` (``"Segura"`` o
        ``"Autoinmunidad"``).

    Raises:
        BlastExecutionError: Si 'blastp' no esta instalado, la base de datos
            no existe, o algun subproceso termina con error o timeout.
    """
    db = Path(db_path)
    _check_blast_environment(db)

    if epitopes_df.empty:
        return epitopes_df.assign(
            blast_task=pd.Series(dtype=str),
            blast_evalue=pd.Series(dtype=float),
            max_pident=pd.Series(dtype=float),
            status=pd.Series(dtype=str),
        )

    result = epitopes_df.reset_index(drop=True).copy()
    lengths = result["sequence"].str.len()
    result["blast_task"] = lengths.apply(_select_task)
    result["blast_evalue"] = lengths.apply(_select_evalue)

    hits_frames = []
    for task, evalue in result[["blast_task", "blast_evalue"]].drop_duplicates().itertuples(index=False):
        tier_mask = (result["blast_task"] == task) & (result["blast_evalue"] == evalue)
        records = list(zip(result.index[tier_mask], result.loc[tier_mask, "sequence"]))
        hits_frames.append(_run_blastp_batch(records, task, db, evalue))
    non_empty_frames = [df for df in hits_frames if not df.empty]
    hits = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame(columns=_OUTFMT6_COLUMNS)

    max_identity_per_query = _max_identity_by_query(hits, lengths, min_query_coverage)

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
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("Tarea BLAST", lambda r: r.blast_task, 14, "<"),
        Column("E-value", lambda r: f"{r.blast_evalue:.3g}", 10, ">"),
        Column("Identidad max (%)", lambda r: f"{r.max_pident:.2f}", 18, ">"),
        Column("Veredicto", lambda r: r.status, 16, ">"),
    ]
    print_fixed_width_table(blast_df.itertuples(index=False), columns)

    n_safe = int((blast_df["status"] == "Segura").sum())
    n_rejected = int((blast_df["status"] == "Autoinmunidad").sum())
    print(f"\nResumen Fase 4: {n_safe} segura(s) / {n_rejected} rechazada(s) por homologia con el proteoma humano.")
