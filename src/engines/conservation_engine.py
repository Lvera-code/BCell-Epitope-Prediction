"""Fase 4d (OPCIONAL): amplitud de conservacion de secuencia contra un panel de referencia local.

Motivo: un epitopo poco conservado entre cepas/variantes de un mismo
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
queda para trabajo futuro, una vez resuelto tambien el punto de
"regiones de interes" (Fase 7 no compone hoy multiples señales en un unico
ranking).

Reutiliza la ejecucion de BLASTp por lotes (``_select_task``/
``_select_evalue``/``_run_blastp_batch``) de ``blast_engine.py`` tal cual, sin
duplicarla -- misma mecanica de eleccion dinamica de task/E-value por
longitud de peptido que ya usa Fase 4.

ADR -- deteccion del "acantilado de cobertura terminal" (peptidos con cola no
nativa de cristalizacion/clonaje)
----------------------------------------------------------------------
Caso real que motiva este mecanismo (verificado sobre el panel de 17
estructuras, PDB 4XAW, gp41 MPER): el candidato B-cell cristalizado incluye
una cola no nativa de 2-3 residuos (`KKK`) añadida para la cristalizacion.
Al consultarlo tal cual contra el panel de conservacion, el filtro de
cobertura minima de query (``min_query_coverage``) descarta casi todos los
hits reales (la cola nunca alinea), hundiendo la amplitud medida a 0.23%
pese a que el motivo nativo (sin la cola) esta conservado en 60.09% del
panel -- una cifra completamente distinta, no un margen de error.

Se investigo, y se descarto, resolver esto leyendo la cabecera de deposito
del PDB (flags ``ENGINEERED``/``SYNTHETIC``, o ausencia de referencia
UniProt): verificado sobre el panel completo que ``ENGINEERED`` no
discrimina (15 de 17 estructuras lo llevan, incluidas cadenas 100% nativas)
y ``SYNTHETIC`` da un falso positivo real en este mismo panel (8FDD:
sintetizado quimicamente, pero 100% nativo en secuencia) -- "sintetizado"
no equivale a "contiene residuos no nativos", y ademas ninguna de las 2
rutas de parseo disponibles (PDB legacy vs. mmCIF moderno) expone estos
flags con la misma consistencia. Comparar contra el UniProt canonico de la
proteina de origen fallaria justo en el caso que motiva esto (4XAW no tiene
referencia UniProt en su cabecera) y añadiria una dependencia de red a un
pipeline deliberadamente 100% local.

En vez de depender de metadatos de deposito (inexistentes o poco fiables),
se detecta la señal directamente en los propios hits de BLASTp de esta
fase: para cada candidato con suficientes hits (``_TERMINAL_TRIM_MIN_HITS``)
por encima de ``identity_threshold``, se calcula la cobertura fraccional
por POSICION de query (que fraccion de esos hits alinea cada residuo). Un
residuo no nativo tipicamente cristalizado como cola muestra un acantilado
real: cobertura ~100% en el interior y ~0-1% en un tramo contiguo de >=2
residuos en un extremo -- a diferencia de un candidato nativo (verificado
con los 2 candidatos de control de gp120/3NGB en el mismo panel: cobertura
uniformemente alta o con caida gradual, nunca un acantilado terminal
limpio). Cuando se detecta, se anota (``conservation_query_trim_start``/
``_end``, 1-indexados sobre ``sequence``) y se re-consulta BLASTp SOLO ese
sub-rango para obtener ``conservation_pct_native_core`` -- verificado sobre
4XAW: recupera el 60.09% real, sin tocar ``conservation_pct`` (que sigue
siendo el valor honesto sobre el candidato tal cual, informativo, no un
reemplazo).

Mecanicamente, esto SOLO tiene sentido para candidatos B-CELL: son los
unicos que pueden provenir de un constructo cristalizado con una cola
añadida real. Los candidatos HTL/CTL son ventanas de barrido de secuencia
sobre la proteina completa (Fase 5/5b) -- nunca un constructo de laboratorio
-- asi que cualquier "acantilado" en uno de ellos es, por construccion,
variabilidad de conservacion real en el borde arbitrario de esa ventana, no
una cola no nativa. Verificado exhaustivamente sobre los 62 candidatos
B-cell/HTL/CTL reales de conservacion del panel de validacion (3NGB, 4XAW,
los 2 unicos antigenos con panel LANL/CATNAP disponible): activar la
deteccion tambien en HTL/CTL genera 4 falsos positivos reales, uno por
candidato (p. ej. ``ENFNMWKNNMVEQMQ``, un HTL de 15aa donde solo los 2
residuos iniciales caen a baja cobertura -- variabilidad real de gp120, no
un artefacto). Por eso ``run_conservation_filter`` expone
``detect_terminal_trim`` como parametro (``False`` por defecto) y
``pipeline.py`` (Fase 6b) lo activa unicamente para el bloque B-cell.

Puramente informativo, igual que el resto de la fase: nunca
descarta ni recorta el candidato en si, solo expone la cifra corregida y un
aviso en el log para que el investigador decida.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.engines.blast_engine import _OUTFMT6_COLUMNS, _run_blastp_batch, _select_evalue, _select_task
from src.utils.exceptions import BlastExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

# Umbrales del acantilado de cobertura terminal (ver ADR del modulo). No son
# parametros de produccion que calibrar por panel -- describen la FORMA de un
# artefacto de cola de cristalizacion/clonaje (interior ~solido, extremo
# ~vacio), verificados sobre el panel de 17 estructuras (caso 4XAW y 2
# controles negativos de 3NGB).
_TERMINAL_TRIM_LOW_COVERAGE = 0.10
_TERMINAL_TRIM_HIGH_COVERAGE = 0.80
_TERMINAL_TRIM_MIN_RUN = 2
_TERMINAL_TRIM_MAX_RUN = 5
_TERMINAL_TRIM_MIN_HITS = 10


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


def _query_position_coverage(group: pd.DataFrame, length: int) -> np.ndarray:
    """Cobertura fraccional (0-1) por posicion 0-indexada de un query, sobre los hits de ``group``.

    Args:
        group: Subconjunto de ``hits`` (formato ``-outfmt 6``) de UN solo
            query, ya filtrado por ``pident >= identity_threshold``.
        length: Longitud (aa) del query.
    """
    coverage = np.zeros(length, dtype=float)
    for qstart, qend in zip(group["qstart"], group["qend"]):
        start0 = max(int(qstart) - 1, 0)
        end0 = min(int(qend), length)
        coverage[start0:end0] += 1.0
    return coverage / len(group)


def _detect_terminal_trim(coverage_frac: np.ndarray) -> Optional[Tuple[int, int]]:
    """Nucleo (start, end) 0-indexado inclusive soportado por el panel, o ``None`` si no hay acantilado.

    Ver ADR del modulo. Requiere un tramo contiguo, en UN extremo, de entre
    ``_TERMINAL_TRIM_MIN_RUN`` y ``_TERMINAL_TRIM_MAX_RUN`` residuos con
    cobertura por debajo de ``_TERMINAL_TRIM_LOW_COVERAGE``, y que el resto
    del query (nucleo) tenga cobertura uniformemente por encima de
    ``_TERMINAL_TRIM_HIGH_COVERAGE``.

    El tope superior (``_TERMINAL_TRIM_MAX_RUN``) es la correccion a un falso
    positivo real encontrado sobre el propio panel de validacion (candidato
    nativo 312-324 de gp120/3NGB, region hipervariable del CD4bs): un tramo
    de 5 residuos casi universales (ancla corta real) seguido de 8 residuos
    con cobertura uniformemente baja (~5-6%, NO cercana a 0%) parece
    superficialmente un acantilado, pero es variabilidad biologica real de
    una region hipervariable -- la mayoria del peptido, no una cola breve --
    no una cola no nativa de cristalizacion/clonaje (que en este mismo panel
    mide 2 residuos, caso 4XAW). Un tramo de baja cobertura mas largo que
    ``_TERMINAL_TRIM_MAX_RUN`` se trata como señal biologica real y NO se
    recorta (ese extremo del query se mantiene, en vez de forzar un nucleo
    corto y descartar la mayor parte del candidato).
    """
    n = len(coverage_frac)
    lead = 0
    while lead < n and coverage_frac[lead] < _TERMINAL_TRIM_LOW_COVERAGE:
        lead += 1
    trail = 0
    while trail < n - lead and coverage_frac[n - 1 - trail] < _TERMINAL_TRIM_LOW_COVERAGE:
        trail += 1

    lead_is_tag = _TERMINAL_TRIM_MIN_RUN <= lead <= _TERMINAL_TRIM_MAX_RUN
    trail_is_tag = _TERMINAL_TRIM_MIN_RUN <= trail <= _TERMINAL_TRIM_MAX_RUN
    if not lead_is_tag and not trail_is_tag:
        return None

    core_start = lead if lead_is_tag else 0
    core_end = (n - 1 - trail) if trail_is_tag else n - 1
    if core_start > core_end:
        return None
    if coverage_frac[core_start : core_end + 1].min() < _TERMINAL_TRIM_HIGH_COVERAGE:
        return None
    return core_start, core_end


def _native_core_conservation(
    result: pd.DataFrame,
    trim_by_index: Dict[int, Tuple[int, int]],
    db_prefix: Path,
    n_panel_total: int,
    identity_threshold: float,
    min_query_coverage: float,
) -> Dict[int, float]:
    """Re-consulta BLASTp SOLO el sub-rango nativo detectado para cada candidato recortado.

    Segunda pasada de BLASTp deliberadamente separada de la principal (mismo
    patron que el re-chequeo de autotolerancia sobre ``sequence_f5`` en
    ``pipeline.py``): son consultas distintas (subsecuencia, no la secuencia
    completa del candidato), no tiene sentido mezclarlas en el mismo lote.
    """
    trimmed_lengths = pd.Series(
        {idx: end - start + 1 for idx, (start, end) in trim_by_index.items()}
    )
    tasks = trimmed_lengths.apply(_select_task)
    evalues = trimmed_lengths.apply(_select_evalue)

    hits_frames = []
    tiers = pd.DataFrame({"task": tasks, "evalue": evalues}).drop_duplicates()
    for task, evalue in tiers.itertuples(index=False):
        tier_idx = trimmed_lengths.index[(tasks == task) & (evalues == evalue)]
        records = [
            (idx, result.loc[idx, "sequence"][trim_by_index[idx][0] : trim_by_index[idx][1] + 1])
            for idx in tier_idx
        ]
        hits_frames.append(_run_blastp_batch(records, task, db_prefix, evalue, max_target_seqs=n_panel_total))
    non_empty_frames = [df for df in hits_frames if not df.empty]
    trimmed_hits = (
        pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame(columns=_OUTFMT6_COLUMNS)
    )

    breadth = _panel_breadth_by_query(trimmed_hits, trimmed_lengths, identity_threshold, min_query_coverage)
    return {
        idx: round(int(breadth.get(f"peptide_{idx}", 0)) / n_panel_total * 100.0, 2)
        for idx in trim_by_index
    }


def run_conservation_filter(
    candidates_df: pd.DataFrame,
    panel_fasta_path: str,
    identity_threshold: float = Settings.CONSERVATION_IDENTITY_THRESHOLD,
    min_query_coverage: float = Settings.BLAST_MIN_QUERY_COVERAGE,
    detect_terminal_trim: bool = False,
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
        detect_terminal_trim: Activa la deteccion del acantilado de
            cobertura terminal (ver ADR del modulo). ``False`` por defecto:
            SOLO tiene sentido mecanistico para candidatos B-cell derivados
            de una estructura PDB (unicos que pueden llevar una cola no
            nativa de cristalizacion/clonaje) -- verificado exhaustivamente
            sobre los 62 candidatos B-cell/HTL/CTL reales de conservacion
            del panel de validacion (3NGB, 4XAW): activarlo para HTL/CTL
            genera 4 falsos positivos reales, uno por candidato (ventanas de
            barrido de secuencia sobre la proteina completa, nunca un
            constructo cristalizado -- cualquier "acantilado" ahi es
            variabilidad de secuencia real en el borde arbitrario de esa
            ventana, no una cola no nativa). El llamador (``pipeline.py``,
            Fase 6b) lo activa unicamente para el bloque B-cell.

    Returns:
        Copia de ``candidates_df`` con 3 columnas nuevas: ``n_panel_matches``
        (secuencias distintas del panel matcheadas), ``n_panel_total``
        (tamaño del panel) y ``conservation_pct`` (``n_panel_matches /
        n_panel_total * 100``, redondeado a 2 decimales); mas 3 columnas
        informativas casi siempre vacias (``pd.NA``/``NaN``), pobladas solo
        cuando se detecta un acantilado de cobertura terminal (ver ADR del
        modulo): ``conservation_query_trim_start``/``_end`` (1-indexados
        sobre ``sequence``, el sub-rango realmente soportado por el panel) y
        ``conservation_pct_native_core`` (conservacion recalculada SOLO
        sobre ese sub-rango). ``conservation_pct`` nunca se modifica por
        esto -- sigue siendo el valor honesto sobre el candidato completo.

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
        # A diferencia de Fase 4 (solo importa la identidad MAXIMA), esta
        # fase cuenta la amplitud completa -- TODAS las secuencias del panel
        # que pasan el umbral, no solo la mejor -- asi que el limite por
        # defecto de BLAST (500 hits/query) truncaria en silencio el conteo
        # de cualquier candidato conservado en mas del 17% del panel (ver
        # docstring de ``blast_engine._run_blastp_batch``). Se pide
        # explicitamente cubrir el panel entero.
        hits_frames.append(_run_blastp_batch(records, task, db_prefix, evalue, max_target_seqs=n_panel_total))
    non_empty_frames = [df for df in hits_frames if not df.empty]
    hits = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame(columns=_OUTFMT6_COLUMNS)

    breadth = _panel_breadth_by_query(hits, lengths, identity_threshold, min_query_coverage)

    result["n_panel_matches"] = [int(breadth.get(f"peptide_{idx}", 0)) for idx in result.index]
    result["n_panel_total"] = n_panel_total
    result["conservation_pct"] = (result["n_panel_matches"] / n_panel_total * 100.0).round(2)

    # Deteccion del acantilado de cobertura terminal (ver ADR del modulo y
    # docstring de ``detect_terminal_trim``): SOLO si el llamador la activa
    # explicitamente (candidatos B-cell), sobre los hits que ya pasan
    # identity_threshold, ANTES del filtro de cobertura minima de query (que
    # es precisamente lo que el acantilado explica que se pierda).
    trim_by_index: Dict[int, Tuple[int, int]] = {}
    if detect_terminal_trim and not hits.empty:
        pident_hits = hits[hits["pident"] >= identity_threshold]
        if not pident_hits.empty:
            hit_query_idx = pident_hits["qseqid"].str.replace("peptide_", "", regex=False).astype(int)
            for idx, group in pident_hits.groupby(hit_query_idx):
                if len(group) < _TERMINAL_TRIM_MIN_HITS or idx not in lengths.index:
                    continue
                length = int(lengths.loc[idx])
                coverage_frac = _query_position_coverage(group, length)
                trim = _detect_terminal_trim(coverage_frac)
                if trim is not None and trim != (0, length - 1):
                    trim_by_index[idx] = trim

    result["conservation_query_trim_start"] = pd.Series(dtype="Int64")
    result["conservation_query_trim_end"] = pd.Series(dtype="Int64")
    result["conservation_pct_native_core"] = pd.Series(dtype="float64")

    if trim_by_index:
        native_core_pct = _native_core_conservation(
            result, trim_by_index, db_prefix, n_panel_total, identity_threshold, min_query_coverage
        )
        for idx, (start0, end0) in trim_by_index.items():
            result.loc[idx, "conservation_query_trim_start"] = start0 + 1
            result.loc[idx, "conservation_query_trim_end"] = end0 + 1
            result.loc[idx, "conservation_pct_native_core"] = native_core_pct[idx]
            logger.warning(
                "Candidato '%s' (fila %d): acantilado de cobertura terminal detectado en el panel de "
                "conservacion -- solo los residuos %d-%d (de %d) estan realmente soportados. "
                "conservation_pct=%.2f%% (candidato completo) vs. conservation_pct_native_core=%.2f%% "
                "(sub-rango soportado). Posible cola no nativa de cristalizacion/clonaje; revisar antes "
                "de reportar la cifra sobre el candidato completo.",
                result.loc[idx, "sequence"], idx, start0 + 1, end0 + 1, len(result.loc[idx, "sequence"]),
                result.loc[idx, "conservation_pct"], native_core_pct[idx],
            )

    return result


def print_conservation_report(conservation_df: pd.DataFrame) -> None:
    """Imprime el informe de amplitud de conservacion: candidatos ordenados de mas a menos conservado.

    La tabla en consola omite candidatos con ``conservation_pct == 0`` (0%
    de conservacion no es informacion accionable fila por fila -- son
    ruido visual, no hallazgos) -- el CSV persistido (``final_path`` en
    ``pipeline.py``) SIEMPRE tiene el 100% de los candidatos, 0% incluido,
    mismo mecanismo que el truncado a ``_MAX_CONSOLE_ROWS`` de Fase 6/6c.
    La media de conservacion se calcula sobre TODOS los candidatos (no solo
    los mostrados), es una metrica real del set completo.
    """
    if conservation_df.empty:
        print("No hay candidatos para evaluar conservacion.")
        return

    n_panel_total = int(conservation_df["n_panel_total"].iloc[0])
    mean_pct = conservation_df["conservation_pct"].mean()

    displayed = conservation_df[conservation_df["conservation_pct"] > 0]
    n_omitted = len(conservation_df) - len(displayed)

    if displayed.empty:
        print(f"Ningun candidato conservado (0% en los {len(conservation_df)} evaluados). Detalle completo en el CSV.")
        print(f"\nPanel de referencia: {n_panel_total} secuencia(s). Conservacion media: {mean_pct:.2f}%.")
        return

    seq_width = max(30, displayed["sequence"].str.len().max() + 2)
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("Matches", lambda r: f"{r.n_panel_matches}/{r.n_panel_total}", 12, ">"),
        Column("Conservacion (%)", lambda r: f"{r.conservation_pct:.2f}", 18, ">"),
    ]
    ordered = displayed.sort_values("conservation_pct", ascending=False)
    print_fixed_width_table(ordered.itertuples(index=False), columns)
    if n_omitted:
        print(f"({n_omitted} candidato(s) con 0% de conservacion omitido(s) de la tabla -- ver CSV para el detalle completo.)")

    print(f"\nPanel de referencia: {n_panel_total} secuencia(s). Conservacion media: {mean_pct:.2f}%.")

    if "conservation_pct_native_core" in conservation_df.columns:
        trimmed = conservation_df[conservation_df["conservation_pct_native_core"].notna()]
        for row in trimmed.itertuples():
            print(
                f"  [AVISO] '{row.sequence}': acantilado de cobertura terminal detectado -- solo "
                f"{row.conservation_query_trim_start}-{row.conservation_query_trim_end} esta soportado "
                f"por el panel. conservation_pct={row.conservation_pct:.2f}% (candidato completo) vs. "
                f"conservation_pct_native_core={row.conservation_pct_native_core:.2f}% (sub-rango nativo)."
            )
