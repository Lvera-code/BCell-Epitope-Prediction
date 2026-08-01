"""Fase 4d (OPCIONAL): amplitud de conservacion de secuencia contra un panel de referencia local.

Motivo (feedback de Carmen Elena Gomez, group leader Poxvirus and Vaccines,
2026-07-30): un epitopo poco conservado entre cepas/variantes de un mismo
patogeno protege solo contra la variante exacta usada para diseñarlo. Priorizar
epitopos conservados amplia la cobertura (un solo constructo protege contra mas
variantes circulantes) y presiona al patogeno de forma mas dura (las regiones
conservadas suelen estarlo porque son funcionalmente restringidas -- mutarlas
tiene un costo real de fitness para el patogeno).

A diferencia de Fase 4 (BLAST_HUMAN_DB, un proteoma humano fijo compartido por
cualquier corrida), no existe un panel de conservacion universal: conservacion
es relativo a las cepas/variantes del PATOGENO especifico bajo analisis, que
cambia de corrida en corrida. Por eso esta fase es enteramente OPCIONAL y no
tiene ninguna base de datos por defecto -- se activa solo si el usuario pasa
`--panel-conservacion <fasta>` (ver `pipeline.py`), un multi-FASTA SIN indexar
con secuencias de referencia (otras cepas/clados/variantes) del mismo
patogeno. Sin el flag, la fase se omite por completo: a diferencia de Fase 6
(bnAb, que siempre corre y puede legitimamente devolver vacio), aca ni
siquiera se invoca 'blastp'/'makeblastdb' si no hay panel -- no tiene sentido
intentarlo sin saber contra que comparar.

Indexado: el FASTA del panel se indexa una vez con 'makeblastdb' (mismo
paquete NCBI BLAST+ que ya requiere Fase 4) y el resultado se cachea en
``Settings.CONSERVATION_DB_CACHE_DIR``, con el nombre del directorio de cache
derivado de un hash del CONTENIDO del archivo -- si el usuario reusa el mismo
panel entre corridas, no se vuelve a indexar; si lo edita, el hash cambia y se
reindexa automaticamente sin intervencion manual.

Metrica -- AMPLITUD, no mejor-hit: se cuenta, por candidato, cuantas
secuencias DISTINTAS del panel matchean por encima de
``Settings.CONSERVATION_IDENTITY_THRESHOLD`` (con el mismo filtro de
cobertura minima de consulta que Fase 4, ver
``Settings.BLAST_MIN_QUERY_COVERAGE``), no solo la identidad maxima contra
cualquiera de ellas. Un candidato identico a una unica cepa rara del panel no
es "conservado" en el sentido que le importa a Elena (proteger contra MUCHAS
variantes circulantes); uno con match aceptable contra el 80% del panel si lo
es, aunque ninguno de esos matches individuales sea un 100% de identidad.

Puramente informativo, igual que Fase 6 (bnAb): NO descarta ningun candidato,
solo lo anota (``conservation_pct`` en la metadata del constructo, Fase 7) --
mismo criterio ya aplicado a N-glicosilacion (ver
``src.engines.construct_assembly``): la decision de priorizar conservacion
queda para una sesion posterior, una vez resuelto tambien el punto de
"regiones de interes" (Fase 7 no compone hoy multiples señales en un unico
ranking).

Reutiliza la ejecucion de BLASTp por lotes (``_select_task``/
``_select_evalue``/``_run_blastp_batch``) de ``blast_engine.py`` tal cual, sin
duplicarla -- misma mecanica de eleccion dinamica de task/E-value por
longitud de peptido que ya usa Fase 4.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config.settings import Settings
from src.engines.blast_engine import _OUTFMT6_COLUMNS, _run_blastp_batch, _select_evalue, _select_task
from src.utils.exceptions import BlastExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)


def _panel_cache_dir(panel_fasta: Path) -> Path:
    """Directorio de cache para la base BLAST indexada de este panel, por hash de CONTENIDO.

    Hash de contenido (no de ruta/mtime): si el usuario reusa el mismo path
    con un FASTA distinto, o edita el panel entre corridas, el hash cambia y
    se reindexa solo -- no hay forma de servir en silencio un indice
    desactualizado.
    """
    content_hash = hashlib.sha256(panel_fasta.read_bytes()).hexdigest()[:16]
    return Settings.CONSERVATION_DB_CACHE_DIR / content_hash


def _count_panel_sequences(panel_fasta: Path) -> int:
    """Cuenta secuencias en el FASTA del panel (una linea '>' por secuencia)."""
    with panel_fasta.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.startswith(">"))


def ensure_panel_db(panel_fasta_path: str) -> Tuple[Path, int]:
    """Indexa (si hace falta) el panel de referencia con 'makeblastdb' y cachea el resultado.

    Args:
        panel_fasta_path: Ruta a un multi-FASTA SIN indexar (secuencias de
            referencia -- otras cepas/clados/variantes -- del mismo patogeno
            bajo analisis en esta corrida).

    Returns:
        Tupla ``(db_prefix, n_panel_sequences)``: prefijo de la base BLAST
        indexada (cacheada por hash de contenido, ver ``_panel_cache_dir`` --
        si el archivo no cambio desde la ultima corrida, reusa el indice
        existente sin volver a llamar a 'makeblastdb') y el numero total de
        secuencias del panel (denominador de ``conservation_pct``).

    Raises:
        BlastExecutionError: Si el FASTA del panel no existe o esta vacio,
            'makeblastdb' no esta en el PATH, o el proceso de indexado falla.
    """
    panel_fasta = Path(panel_fasta_path)
    if not panel_fasta.is_file():
        raise BlastExecutionError(
            f"No se encontro el FASTA del panel de conservacion en '{panel_fasta}'."
        )
    if shutil.which("makeblastdb") is None:
        raise BlastExecutionError(
            "El binario 'makeblastdb' no esta disponible en el PATH (mismo paquete NCBI "
            "BLAST+ que ya requiere 'blastp' para la Fase 4 -- ver README.md - Seccion de "
            "Instalacion)."
        )

    n_sequences = _count_panel_sequences(panel_fasta)
    if n_sequences == 0:
        raise BlastExecutionError(
            f"El FASTA del panel de conservacion '{panel_fasta}' no contiene ninguna secuencia."
        )

    cache_dir = _panel_cache_dir(panel_fasta)
    db_prefix = cache_dir / "panel_db"
    if Path(f"{db_prefix}.phr").is_file():
        logger.info("Panel de conservacion ya indexado (cache hit): %s", db_prefix)
        return db_prefix, n_sequences

    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["makeblastdb", "-in", str(panel_fasta), "-dbtype", "prot", "-out", str(db_prefix)]
    logger.info("Indexando panel de conservacion (%d secuencia(s)): %s", n_sequences, " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        raise BlastExecutionError(
            f"makeblastdb termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BlastExecutionError("makeblastdb excedio el tiempo limite de 300s.") from exc

    return db_prefix, n_sequences


def _panel_breadth_by_query(
    hits: pd.DataFrame, query_lengths: pd.Series, identity_threshold: float, min_query_coverage: float
) -> pd.Series:
    """Cuenta, por query, cuantas secuencias DISTINTAS del panel matchean por encima de los umbrales.

    A diferencia de ``blast_engine._max_identity_by_query`` (que busca el
    MEJOR hit individual, para autoinmunidad), esta funcion mide amplitud:
    cuantos miembros distintos del panel (``sseqid`` unicos) supera un
    candidato, no solo el mas parecido -- ver rationale de amplitud vs.
    mejor-hit en el docstring del modulo.

    Args:
        hits: DataFrame en formato ``-outfmt 6`` (mismas columnas que
            ``blast_engine._OUTFMT6_COLUMNS``).
        query_lengths: ``indice -> longitud (aa)`` del candidato consultado.
        identity_threshold: Porcentaje de identidad minimo (inclusive) para
            que un hit cuente contra una secuencia del panel.
        min_query_coverage: Fraccion minima (0-1) de la longitud del
            candidato que un alineamiento debe cubrir para contar (mismo
            criterio anti-ruido que Fase 4, ver ADR en
            ``Settings.BLAST_MIN_QUERY_COVERAGE``): sin este filtro, un
            fragmento minusculo 100% identico por puro azar contaria igual
            que un match real de longitud completa.

    Returns:
        Serie indexada por ``qseqid`` con el numero de secuencias distintas
        del panel matcheadas. Vacia si ``hits`` esta vacio.
    """
    if hits.empty:
        return pd.Series(dtype=int)

    hit_query_idx = hits["qseqid"].str.replace("peptide_", "", regex=False).astype(int)
    hit_query_length = hit_query_idx.map(query_lengths)
    coverage = hits["length"] / hit_query_length
    qualifying = hits[(coverage >= min_query_coverage) & (hits["pident"] >= identity_threshold)]

    if qualifying.empty:
        return pd.Series(dtype=int)
    return qualifying.groupby("qseqid")["sseqid"].nunique()


def run_conservation_filter(
    candidates_df: pd.DataFrame,
    panel_fasta_path: str,
    identity_threshold: float = Settings.CONSERVATION_IDENTITY_THRESHOLD,
    min_query_coverage: float = Settings.BLAST_MIN_QUERY_COVERAGE,
) -> pd.DataFrame:
    """Ejecuta BLASTp local de ``candidates_df`` contra el panel y anota amplitud de conservacion.

    Reusa la misma seleccion dinamica de (``-task``, E-value) por tramo de
    longitud que Fase 4 (``blast_engine._select_task``/``_select_evalue``),
    en lotes por tramo homogeneo -- ver docstring de
    ``blast_engine.run_blastp_filter`` para el detalle de esa logica, sin
    duplicarlo aqui.

    Args:
        candidates_df: Candidatos a anotar, con una columna ``sequence`` (se
            usa tal cual con B-cell/HTL/CTL de Fase 4/5/5b -- no filtra
            filas, solo agrega columnas).
        panel_fasta_path: Ruta al multi-FASTA sin indexar del panel de
            referencia (ver ``ensure_panel_db``).
        identity_threshold: Porcentaje de identidad minimo (inclusive) para
            que un hit cuente contra una secuencia del panel (default
            ``Settings.CONSERVATION_IDENTITY_THRESHOLD``).
        min_query_coverage: Fraccion minima (0-1) de cobertura de consulta
            para que un hit cuente (default ``Settings.BLAST_MIN_QUERY_COVERAGE``,
            mismo criterio que Fase 4).

    Returns:
        Copia de ``candidates_df`` con 3 columnas nuevas: ``n_panel_matches``
        (secuencias distintas del panel matcheadas), ``n_panel_total``
        (tamaño del panel) y ``conservation_pct`` (``n_panel_matches /
        n_panel_total * 100``, redondeado a 2 decimales).

    Raises:
        BlastExecutionError: Ver ``ensure_panel_db`` y
            ``blast_engine._run_blastp_batch``.
    """
    if candidates_df.empty:
        return candidates_df.assign(
            n_panel_matches=pd.Series(dtype=int),
            n_panel_total=pd.Series(dtype=int),
            conservation_pct=pd.Series(dtype=float),
        )

    db_prefix, n_panel_total = ensure_panel_db(panel_fasta_path)

    result = candidates_df.reset_index(drop=True).copy()
    lengths = result["sequence"].str.len()
    tasks = lengths.apply(_select_task)
    evalues = lengths.apply(_select_evalue)

    hits_frames = []
    tiers = pd.DataFrame({"task": tasks, "evalue": evalues}).drop_duplicates()
    for task, evalue in tiers.itertuples(index=False):
        tier_mask = (tasks == task) & (evalues == evalue)
        records = list(zip(result.index[tier_mask], result.loc[tier_mask, "sequence"]))
        hits_frames.append(_run_blastp_batch(records, task, db_prefix, evalue))
    non_empty_frames = [df for df in hits_frames if not df.empty]
    hits = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame(columns=_OUTFMT6_COLUMNS)

    breadth = _panel_breadth_by_query(hits, lengths, identity_threshold, min_query_coverage)

    result["n_panel_matches"] = [int(breadth.get(f"peptide_{idx}", 0)) for idx in result.index]
    result["n_panel_total"] = n_panel_total
    result["conservation_pct"] = (result["n_panel_matches"] / n_panel_total * 100.0).round(2)
    return result


def print_conservation_report(conservation_df: pd.DataFrame) -> None:
    """Imprime el informe de amplitud de conservacion: candidatos ordenados de mas a menos conservado."""
    if conservation_df.empty:
        print("No hay candidatos para evaluar conservacion.")
        return

    seq_width = max(30, conservation_df["sequence"].str.len().max() + 2)
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("Matches", lambda r: f"{r.n_panel_matches}/{r.n_panel_total}", 12, ">"),
        Column("Conservacion (%)", lambda r: f"{r.conservation_pct:.2f}", 18, ">"),
    ]
    ordered = conservation_df.sort_values("conservation_pct", ascending=False)
    print_fixed_width_table(ordered.itertuples(index=False), columns)

    n_panel_total = int(conservation_df["n_panel_total"].iloc[0]) if not conservation_df.empty else 0
    mean_pct = conservation_df["conservation_pct"].mean()
    print(f"\nPanel de referencia: {n_panel_total} secuencia(s). Conservacion media: {mean_pct:.2f}%.")
