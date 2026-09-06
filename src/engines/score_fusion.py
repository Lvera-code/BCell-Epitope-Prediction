"""Fase 3 (fusion continua): combina los scores crudos de los motores de
antigenicidad activos ANTES de umbralizar, en vez de umbralizar cada motor
por separado (con su propio umbral de produccion) y unir despues las
regiones resultantes (mecanismo anterior, ver ``src.engines.consensus``).

ADR -- por que fusionar antes de umbralizar
----------------------------------------------------------------------
La union logica anotada (``consensus.build_annotated_union_table``) predice
un volumen de residuos varias veces mayor que el epitopo real (hasta 20x en
el peor caso del panel de validacion): al ser una union de 4 motores cada
uno con su propio umbral, cualquier motor que cruce su umbral en solitario
basta para que la region completa avance, sin ningun mecanismo que penalice
el desacuerdo entre motores. Combinar los scores continuos ANTES de decidir
(en vez de combinar decisiones binarias ya tomadas) permite que el
desacuerdo entre motores SI cuente: una region donde solo un motor tiene
señal moderada y los demas no dicen nada puntua mas bajo en la fusion que
una donde varios motores coinciden, incluso si ninguno individualmente
supera su propio umbral aislado.

Validado sobre el panel de 17 estructuras (ver Material Suplementario,
Fase 2/3): la fusion (suma de los scores normalizados de los motores
activos, umbral calibrado por combinacion de motores via leave-one-out
cross-validation) reduce el volumen de falsos positivos en ~89% frente a
la union de produccion anterior, a cambio de una perdida de recall real
(no es un reemplazo sin coste). Se probo exhaustivamente producto, suma,
maximo, ponderacion por calidad discriminativa, normalizacion por
percentil-rango, optimizacion del umbral para precision, y una regresion
logistica por pliegue como techo teorico de una combinacion lineal con
pesos aprendidos: ninguna alternativa supera de forma consistente a la
suma sin ponderar con umbral absoluto.

ADR -- suma, con un umbral calibrado por separado para cada combinacion
----------------------------------------------------------------------
La calibracion original se hizo sobre el modo por defecto (estructura, los
4 motores activos a la vez), con la suma de los scores normalizados como
operador de fusion. El pipeline soporta ademas dos rutas con solo 2 motores
activos (FASTA puro: BepiPred+EpiDope; ``structure_only``:
DiscoTope+ScanNet, ver ``pipeline.py``): reusar el mismo umbral de 4
motores sin recalibrar en esos 2 modos se probo directamente y rompe
ambos, en direcciones opuestas (verificado sobre las 17 estructuras del
panel, recombinando los scores crudos ya cacheados en subconjuntos de 2
motores -- ningun dato fabricado): demasiado bajo para BepiPred+EpiDope
(EpiDope tiende a puntuar alto sin DiscoTope/ScanNet diluyendo la suma,
dispara los falsos positivos) y demasiado alto para DiscoTope+ScanNet
(DiscoTope es negativo en la mayoria de residuos de este panel y ScanNet
rara vez supera 0.5, la mayoria de antigenos quedan sin ningun candidato).

Por eso ``FUSION_THRESHOLDS`` tiene una entrada calibrada de forma
INDEPENDIENTE para cada combinacion de motores soportada (no un unico
valor reescalado matematicamente desde el de 4 motores): cada umbral se
recalculo desde cero sobre la suma de esa combinacion especifica,
maximizando F1 con la rejilla exacta de valores unicos del score
fusionado (sin submuestrear), tanto para la estimacion honesta de
generalizacion (leave-one-out cross-validation) como para el umbral final
de despliegue sobre las 17 estructuras completas. El umbral de 4 motores
(1.612165) es identico al ya usado en la validacion original del sistema.

ADR -- limites de normalizacion y umbrales (Tabla S1 nueva, Material
Suplementario)
----------------------------------------------------------------------
Los limites min-max por motor (``FUSION_BOUNDS``) se calcularon sobre el
score crudo de cada motor a lo largo de las 17 estructuras del panel de
validacion, de forma independiente de que otros motores estuvieran activos
a la vez (cada motor produce el mismo score exacto corra solo o
acompañado). Los valores normalizados se recortan a [0,1]
(``numpy.clip``) para que un antigeno futuro fuera de este panel, cuyo
score crudo exceda estos limites, no produzca un valor fuera de rango ni
rompa la fusion -- verificado con valores extremos sinteticos.

Los umbrales de decision (``FUSION_THRESHOLDS``) se calibraron maximizando
F1 (micro) sobre la rejilla exacta de valores unicos del score fusionado en
las 17 estructuras completas, tras validar honestamente la generalizacion
esperada via leave-one-out cross-validation (umbral y limites recalculados
en cada pliegue sin ver la estructura evaluada, para no sobreajustar al
propio panel que despues reporta los resultados).

Se prioriza F1 micro (agregado sobre residuos de todo el panel, no
promediado por antigeno) por reflejar directamente el volumen total de
falsos positivos, sobre F1 macro -- una alternativa de 3 motores
(excluyendo EpiDope, ponderada por calidad discriminativa) da mejor F1
macro pero peor F1 micro; se descarto por anadir un esquema de pesos sin
mejorar la metrica que mas importa para este objetivo, y por requerir
reescribir el argumento de complementariedad entre motores ya establecido
en la Discusion del articulo (caso fHbp sitio 1, BepiPred+DiscoTope).

ADR -- ventana de decision (por que NO se reutiliza la ventana de 9aa)
----------------------------------------------------------------------
Cada motor individual usa una ventana deslizante de 9 aminoacidos con
tolerancia a residuos por debajo del umbral (ver
``src.engines.epitope_mapping.find_valid_windows``) como mecanismo de
DECISION. Aplicar esa misma ventana sobre la señal ya fusionada se probo
explicitamente y empeora todas las combinaciones probadas sin excepcion: el
score fusionado es mas puntiagudo residuo a residuo que cualquier motor
individual (es una media de 4 señales normalizadas), y promediar sobre 9
residuos suprime señal real en vez de suavizar ruido. Se descarta como
criterio de decision. Lo que si hace falta -- y se mantiene -- es agrupar
residuos positivos ya decididos en regiones contiguas (pura adyacencia, sin
promediar ni tolerar huecos) y descartar las mas cortas de
``MIN_FINAL_PEPTIDE_LENGTH``: no cambia ninguna metrica por si solo
(verificado), pero es necesario porque una region de 2-3 residuos es
inutilizable en las fases siguientes (NetMHCpan necesita un nucleo de
8-11aa, NetMHCIIpan una ventana de 15aa).

ADR -- completado hasta TARGET_CANDIDATES + reserva perezosa (sustituye al
antiguo mecanismo de respaldo unico)
----------------------------------------------------------------------
Con el umbral absoluto, algunos antigenos de señal globalmente debil no
generan suficientes regiones validas -- desde ninguna hasta 1-2 de las 3
que Fase 7 puede ensamblar por clase (``Settings.CONSTRUCT_TOP_N_PER_CLASS``,
verificado: 5-7 de 17 antigenos segun el modo, antes de aplicar este
mecanismo). Como Fase 7 solo puede ensamblar bloques B-cell de lo que
sobrevive aqui, un antigeno con menos candidatos de los que Fase 7 podria
usar pierde una oportunidad real de que un bloque correcto llegue al
constructo -- mismo argumento que ya usa la Discusion del articulo para
descartar el consenso estricto entre motores sobre la union anterior, ahora
generalizado en dos direcciones verificadas sobre el panel de 17
estructuras (ver Material Suplementario):

1. El antiguo mecanismo (una unica ventana de respaldo cuando CERO
   residuos superan el umbral) resolvia el caso mas extremo pero dejaba sin
   tocar antigenos con 1-2 regiones sobre umbral -- que tampoco alcanzan las
   ``TARGET_CANDIDATES`` que Fase 7 podria aprovechar. Se generaliza: SIEMPRE
   que un antigeno tenga menos de ``TARGET_CANDIDATES`` regiones sobre
   umbral, se completa hasta ese numero con las ventanas de
   ``FALLBACK_WINDOW`` residuos de mayor score medio de fusion en el resto
   de la proteina, sin solaparse entre si ni con las regiones ya aceptadas
   (``_top_windows``, greedy por score descendente). ``TARGET_CANDIDATES``
   reusa ``Settings.CONSTRUCT_TOP_N_PER_CLASS`` (3): no es un hiperparametro
   nuevo que calibrar, es el mismo limite que Fase 7 ya aplica al
   ensamblar -- verificado por leave-one-out cross-validation sobre la
   rejilla {1..5}, que elige 3 en 17/17 pliegues.
2. El mecanismo anterior solo garantizaba que un candidato llegara a Fase 4
   (auto-tolerancia/BLASTp), no que la sobreviviera: un antigeno cuyo unico
   candidato de respaldo resulta descartado por homologia humana real se
   queda sin ningun bloque B-cell en el constructo (caso verificado: 7STS).
   Se añade una reserva de ``RESERVE_EXTRA`` ventanas adicionales (mismo
   criterio, sin solape con nada ya emitido) quando se activa el
   completado del punto 1; estas viajan con ``candidate_source='reserve'``
   y participan igual que cualquier otra fila en Fase 4 (BLASTp) y en el
   ranking por percentil de Fase 7 (``_select_bcell_candidates``), que ya
   corta en las mejores ``top_n`` supervivientes -- ninguna de las dos fases
   necesita saber que una fila viene de la reserva para que la promocion
   ocurra: si Fase 4 descarta los candidatos de ``threshold``/``top_up``, la
   reserva que SI sobrevive queda disponible para que Fase 7 la seleccione,
   sin cambio de control de flujo en ``pipeline.py`` ni en
   ``construct_assembly.py``.

Cuando un antigeno ya alcanza ``TARGET_CANDIDATES`` regiones sobre umbral,
no se genera ningun candidato adicional (ni completado ni reserva): el
mecanismo no toca el volumen de antigenos donde el umbral ya funciona,
verificado byte a byte sobre el panel completo (12 de 17 antigenos
identicos al mecanismo anterior). Columna ``candidate_source`` (valores
``'threshold'``/``'top_up'``/``'reserve'``) reemplaza a la antigua columna
booleana ``fallback``, que se mantiene como alias derivado
(``True`` si ``candidate_source != 'threshold'``) para no romper
``print_fusion_table`` ni el trazado existente hacia Fase 4/7.

Se probaron y descartaron, con datos reales del panel completo (no solo
los antigenos problematicos): un umbral de "rescate" secundario mas bajo
(con o sin puerta de acuerdo entre motores) empeora el compromiso
precision/recall porque no mueve la region hacia el epitopo real, solo
engorda la misma region equivocada -- el mismo modo de fallo ya descartado
del umbral adaptativo por percentil; reutilizar los umbrales YA calibrados
de 2 motores como regla OR reintroduce el volumen excesivo que la fusion
elimino; recombinar la señal de ranking (rango percentil, votos por motor,
z-scores) no mejora el acierto en el primer puesto porque, en los casos que
fallan, los 4 motores coinciden en la misma region equivocada -- no hay
complementariedad que explotar ahi; relajar la longitud minima de 9aa
infla el volumen sin mejorar F1; expandir espacialmente en 3D desde las
coordenadas (parche de superficie) da precision marginal muy por debajo de
la media del panel. ``FALLBACK_WINDOW=9`` en si mismo se confirmo optimo
(barrido de longitud de ventana, N y separacion minima).

Verificado sobre el panel completo (post Fase 4): microF1 sube de 0.306 a
0.353 y macroF1 de 0.287 a 0.383 frente al mecanismo anterior, con un unico
antigeno degradado de forma acotada (3NGB, un candidato adicional de bajo
rendimiento) y cero antigenos sin ningun candidato superviviente de Fase 4.
Limite residual explicito, no resuelto por este mecanismo: 2 antigenos
(8UP2, 7STR) siguen sin ningun acierto porque la ventana que si solaparia
el epitopo real es descartada por Fase 4 (homologia humana real, no un
fallo de este mecanismo); ver Discusion del articulo.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.engines import bepipred_engine, discotope_engine, epidope_engine, scannet_engine
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

MIN_FINAL_PEPTIDE_LENGTH = 9
FALLBACK_WINDOW = 9

# Reusa el mismo limite que Fase 7 aplica al ensamblar por clase (ver ADR del
# modulo): no es un hiperparametro nuevo e independiente de este mecanismo.
TARGET_CANDIDATES = Settings.CONSTRUCT_TOP_N_PER_CLASS

# Ventanas adicionales de reserva mas alla de TARGET_CANDIDATES, generadas
# SOLO cuando el completado del punto 1 del ADR se activa (regiones sobre
# umbral < TARGET_CANDIDATES). Ver ADR para el porque de la reserva perezosa.
RESERVE_EXTRA = 3

# accession_col/score_col por motor, mismo patron que
# ``src.engines.construct_assembly._RAW_SCORE_ENGINES`` (version publica de
# ese diccionario: se reexporta desde alli para no duplicarlo).
RAW_SCORE_ENGINES: Dict[str, Tuple[str, str]] = {
    "bepipred": (bepipred_engine.ACCESSION_COLUMN, bepipred_engine.SCORE_COLUMN),
    "epidope": (epidope_engine.ACCESSION_COLUMN, epidope_engine.SCORE_COLUMN),
    "discotope": (discotope_engine.ACCESSION_COLUMN, discotope_engine.SCORE_COLUMN),
    "scannet": (scannet_engine.ACCESSION_COLUMN, scannet_engine.SCORE_COLUMN),
}

# Candidatos de columna de residuo por motor (BepiPred-3.0 expone varios
# nombres posibles segun version, ver ``bepipred_engine.RESIDUE_COLUMN_CANDIDATES``;
# el resto expone un unico nombre fijo).
_RAW_RESIDUE_COLUMNS: Dict[str, Sequence[str]] = {
    "bepipred": bepipred_engine.RESIDUE_COLUMN_CANDIDATES,
    "epidope": (epidope_engine.RESIDUE_COLUMN,),
    "discotope": (discotope_engine.RESIDUE_COLUMN,),
    "scannet": (scannet_engine.RESIDUE_COLUMN,),
}

# Limites (min, max) de normalizacion por motor. Ver ADR del modulo.
FUSION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "bepipred": (0.006587, 0.373576),
    "epidope": (0.328649, 0.922995),
    "discotope": (-1.721370, 5.960310),
    "scannet": (0.000000, 0.802000),
}

# Umbral de la SUMA de scores normalizados, calibrado por separado (desde
# cero, no reescalado) para cada combinacion de motores soportada por el
# pipeline (ver ADR del modulo). Claves como frozenset para no depender del
# orden de activacion.
FUSION_THRESHOLDS: Dict[frozenset, float] = {
    frozenset({"bepipred", "epidope", "discotope", "scannet"}): 1.612165,
    frozenset({"bepipred", "epidope"}): 1.164683,
    frozenset({"discotope", "scannet"}): 0.344902,
}

_OUTPUT_BASE_COLUMNS = [
    "accession", "start", "end", "length", "sequence", "origen", "fused_score",
    "candidate_source", "fallback",
]


def _resolve_residue_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _accession_id(accession: str) -> str:
    """Normaliza un accession a su primer token (ver ``consensus.accession_id``)."""
    return accession.split()[0] if accession else accession


def _merge_adjacent(pred: np.ndarray) -> List[Tuple[int, int]]:
    """Agrupa posiciones ``True`` contiguas (0-indexadas) en rangos ``(start, end)`` inclusive.

    Pura adyacencia: sin promediar ni tolerar huecos, a diferencia de
    ``src.engines.epitope_mapping.find_valid_windows`` (ver ADR del modulo,
    por que no se reutiliza esa ventana aqui).
    """
    n = len(pred)
    regions: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not pred[i]:
            i += 1
            continue
        j = i
        while j < n and pred[j]:
            j += 1
        regions.append((i, j - 1))
        i = j
    return regions


def _windows_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _top_windows(
    fused: np.ndarray, window: int, k: int, exclude: Sequence[Tuple[int, int]] = ()
) -> List[Tuple[int, int]]:
    """Hasta ``k`` ventanas de ``window`` residuos, sin solape entre si ni con ``exclude``.

    Generaliza la antigua ``_best_fixed_window`` (equivalente a ``k=1,
    exclude=()``): greedy por score medio descendente, sin umbral ni
    tolerancia de huecos (ver ADR del modulo). Devuelve menos de ``k``
    ventanas si la proteina no admite mas huecos libres sin solape (p. ej.
    una proteina de 13-16 residuos solo admite una ventana de 9 no
    solapante con otra ya aceptada).
    """
    n = len(fused)
    if k <= 0 or n == 0:
        return []
    if n <= window:
        candidate = (0, n - 1)
        if any(_windows_overlap(candidate, ex) for ex in exclude):
            return []
        return [candidate]

    means = np.array([fused[i : i + window].mean() for i in range(n - window + 1)])
    order = np.argsort(-means)

    chosen: List[Tuple[int, int]] = []
    blocked: List[Tuple[int, int]] = list(exclude)
    for start in order:
        candidate = (int(start), int(start) + window - 1)
        if any(_windows_overlap(candidate, b) for b in blocked):
            continue
        chosen.append(candidate)
        blocked.append(candidate)
        if len(chosen) >= k:
            break
    return chosen


def fuse_and_extract_regions(
    raw_dfs: Dict[str, pd.DataFrame],
    sequence_lookup: Dict[str, str],
    min_length: int = MIN_FINAL_PEPTIDE_LENGTH,
) -> pd.DataFrame:
    """Fase 3: fusiona los scores crudos de los motores activos y extrae regiones candidatas.

    Sustituye a ``src.engines.consensus.build_annotated_union_table`` (ver
    ADR del modulo para el razonamiento completo). Mismo esquema de salida
    base (``accession``/``start``/``end``/``length``/``sequence``) mas
    columnas nuevas (``fused_score``, ``candidate_source``, ``fallback``) y
    una columna ``{motor}_score`` por motor activo (media del score
    NORMALIZADO de ese motor dentro de la region, para trazabilidad -- ya no
    determina la seleccion, que ahora depende del score fusionado conjunto).

    Args:
        raw_dfs: Diccionario ``nombre_motor -> DataFrame`` de scores crudos
            por residuo (salida directa de Fase 2, antes de cualquier
            ``extract_epitopes`` individual). Claves esperadas:
            ``'bepipred'``, ``'epidope'``, ``'discotope'``, ``'scannet'``;
            un ``DataFrame`` ``None`` o vacio para una clave se trata como
            "ese motor no corrio para ninguna accession".
        sequence_lookup: ``accession -> secuencia completa`` (ver
            ``pipeline._build_full_sequence_lookup``), usado para
            reconstruir la subsecuencia de cada region final.
        min_length: Filtro de longitud minima aplicado a las regiones que
            SI superan el umbral (no aplica a las regiones de completado ni
            de reserva, que siempre miden exactamente ``FALLBACK_WINDOW``
            residuos).

    Returns:
        DataFrame con una fila por region final. Incluye, para cada
        accession con menos de ``TARGET_CANDIDATES`` regiones sobre umbral,
        el completado hasta ese numero (``candidate_source='top_up'``) mas
        hasta ``RESERVE_EXTRA`` filas de reserva (``candidate_source=
        'reserve'``); ver ADR del modulo. Vacio si ningun motor de
        ``raw_dfs`` tiene datos.

    Raises:
        ValueError: Si la combinacion de motores activos en ``raw_dfs`` no
            tiene un umbral calibrado en ``FUSION_THRESHOLDS`` (las 3 rutas
            soportadas por el pipeline: 4 motores, o los 2 pares fijos de
            motores de secuencia/estructura).
    """
    active_engines = [key for key, df in raw_dfs.items() if df is not None and not df.empty]
    if not active_engines:
        return pd.DataFrame(columns=_OUTPUT_BASE_COLUMNS)

    threshold_key = frozenset(active_engines)
    if threshold_key not in FUSION_THRESHOLDS:
        raise ValueError(
            f"No hay umbral de fusion calibrado para la combinacion de motores activos "
            f"{sorted(active_engines)}. Combinaciones soportadas: "
            f"{[sorted(k) for k in FUSION_THRESHOLDS]}."
        )
    threshold = FUSION_THRESHOLDS[threshold_key]

    per_engine_groups: Dict[str, Dict[str, pd.DataFrame]] = {}
    per_engine_cols: Dict[str, Tuple[str, Optional[str]]] = {}
    for engine_key in active_engines:
        accession_col, score_col = RAW_SCORE_ENGINES[engine_key]
        df = raw_dfs[engine_key]
        residue_col = _resolve_residue_column(df, _RAW_RESIDUE_COLUMNS[engine_key])
        per_engine_cols[engine_key] = (score_col, residue_col)
        per_engine_groups[engine_key] = {
            aid: group.reset_index(drop=True)
            for aid, group in df.groupby(df[accession_col].map(_accession_id), sort=False)
        }

    all_ids = list(
        dict.fromkeys(aid for groups in per_engine_groups.values() for aid in groups.keys())
    )
    if not all_ids:
        return pd.DataFrame(columns=_OUTPUT_BASE_COLUMNS)

    engine_abbrev = {"bepipred": "Bp", "epidope": "Ed", "discotope": "Dt", "scannet": "Sn"}

    records = []
    for accession in all_ids:
        norm_by_engine: Dict[str, np.ndarray] = {}
        residue_letters: Optional[List[str]] = None
        n_residues: Optional[int] = None

        for engine_key in active_engines:
            group = per_engine_groups[engine_key].get(accession)
            if group is None:
                continue
            score_col, residue_col = per_engine_cols[engine_key]
            if n_residues is not None and len(group) != n_residues:
                logger.warning(
                    "Accession '%s': el motor '%s' devuelve %d residuo(s), pero otro motor activo "
                    "ya establecio %d para la misma accession. Posible desalineacion de scores "
                    "crudos; se omite este motor para esta accession.",
                    accession, engine_key, len(group), n_residues,
                )
                continue
            lo, hi = FUSION_BOUNDS[engine_key]
            norm_by_engine[engine_key] = np.clip((group[score_col].values - lo) / (hi - lo), 0.0, 1.0)
            if n_residues is None:
                n_residues = len(group)
                if residue_col is not None and residue_col in group.columns:
                    residue_letters = group[residue_col].astype(str).tolist()

        if not norm_by_engine or n_residues is None:
            continue

        origen_label = "+".join(engine_abbrev[e] for e in active_engines if e in norm_by_engine)
        fused = np.sum(np.vstack(list(norm_by_engine.values())), axis=0)
        full_sequence = sequence_lookup.get(accession, "")

        pred = fused >= threshold
        threshold_regions = [(s, e) for s, e in _merge_adjacent(pred) if (e - s + 1) >= min_length]

        regions_with_source: List[Tuple[Tuple[int, int], str]] = [
            (region, "threshold") for region in threshold_regions
        ]
        if len(threshold_regions) < TARGET_CANDIDATES:
            top_up = _top_windows(
                fused, FALLBACK_WINDOW, TARGET_CANDIDATES - len(threshold_regions), exclude=threshold_regions
            )
            regions_with_source.extend((region, "top_up") for region in top_up)
            already_emitted = threshold_regions + top_up
            reserve = _top_windows(fused, FALLBACK_WINDOW, RESERVE_EXTRA, exclude=already_emitted)
            regions_with_source.extend((region, "reserve") for region in reserve)

        for (start0, end0), candidate_source in regions_with_source:
            start, end = start0 + 1, end0 + 1
            length = end - start + 1
            if full_sequence:
                sequence = full_sequence[start0 : end0 + 1]
            elif residue_letters is not None:
                sequence = "".join(residue_letters[start0 : end0 + 1])
            else:
                sequence = ""

            record = {
                "accession": accession,
                "start": start,
                "end": end,
                "length": length,
                "sequence": sequence,
                "origen": origen_label,
                "fused_score": float(fused[start0 : end0 + 1].mean()),
                "candidate_source": candidate_source,
                "fallback": candidate_source != "threshold",
            }
            for engine_key, norm_scores in norm_by_engine.items():
                record[f"{engine_key}_score"] = float(norm_scores[start0 : end0 + 1].mean())
            records.append(record)

    columns = _OUTPUT_BASE_COLUMNS + [f"{e}_score" for e in active_engines]
    union_df = pd.DataFrame.from_records(records, columns=columns)
    if union_df.empty:
        return union_df

    union_df = union_df.sort_values(["accession", "start", "end"]).reset_index(drop=True)
    return union_df


def print_fusion_table(union_df: pd.DataFrame, engine_keys: Optional[Sequence[str]] = None) -> None:
    """Imprime la tabla de regiones fusionadas en consola (equivalente a ``consensus.print_union_table``)."""
    if union_df.empty:
        print(f"No se encontraron regiones de epitopo tras la fusion de scores.")
        return

    if engine_keys is None:
        engine_keys = [c[: -len("_score")] for c in union_df.columns if c.endswith("_score") and c != "fused_score"]

    columns = [
        Column("accession", lambda r: r.accession, 28, "<"),
        Column("start", lambda r: str(r.start), 7, ">"),
        Column("end", lambda r: str(r.end), 7, ">"),
        Column("len", lambda r: str(r.length), 6, ">"),
        Column("fused", lambda r: f"{r.fused_score:.4f}", 8, ">", prefix="  "),
        Column("origen_cand.", lambda r: r.candidate_source, 12, "<", prefix="  "),
        Column("sequence", lambda r: r.sequence, 0, "<", prefix="  "),
    ]
    print_fixed_width_table(union_df.itertuples(index=False), columns, group_by=lambda r: r.accession)

    source_counts = union_df["candidate_source"].value_counts()
    n_top_up = int(source_counts.get("top_up", 0))
    n_reserve = int(source_counts.get("reserve", 0))
    print(
        f"\nResumen Fase 3 (fusion): {len(union_df)} region(es) sobre umbral "
        f"({n_top_up} de completado, {n_reserve} de reserva) sobre {union_df['accession'].nunique()} accession(es)."
    )
