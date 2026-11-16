"""Phase 3 (continuous fusion): combines the raw scores of the active
antigenicity engines BEFORE thresholding, instead of thresholding each engine
separately (with its own production threshold) and then joining the
resulting regions (the previous mechanism, see ``src.engines.consensus``).

ADR: why fuse before thresholding
----------------------------------------------------------------------
The annotated logical union (``consensus.build_annotated_union_table``) predicts
a volume of residues several times larger than the real epitope (up to 20x in
the worst case of the validation panel): being a union of 4 engines each
with its own threshold, any engine crossing its threshold on its own
is enough for the complete region to advance, without any mechanism that penalizes
disagreement between engines. Combining the continuous scores BEFORE deciding
(instead of combining binary decisions already taken) lets
disagreement between engines actually count: a region where only one engine has a
moderate signal and the others say nothing scores lower in the fusion than
one where several engines agree, even if none individually
exceeds its own isolated threshold.

Validated on the panel of 17 structures (see Supplementary Material,
Phase 2/3): the fusion (sum of the normalized scores of the active
engines, threshold calibrated per engine combination via leave-one-out
cross-validation) reduces the volume of false positives by ~89% compared with
the previous production union, at the cost of a real loss of recall
(it is not a cost-free replacement). Product, sum,
maximum, weighting by discriminative quality, percentile-rank
normalization, threshold optimization for precision, and a per-fold logistic
regression as the theoretical ceiling of a linear combination with
learned weights were all tested exhaustively: no alternative consistently beats the
unweighted sum with an absolute threshold.

ADR: sum, with a threshold calibrated separately for each combination
----------------------------------------------------------------------
The original calibration was done on the default mode (structure, all
4 engines active at once), with the sum of the normalized scores as the
fusion operator. The pipeline additionally supports two paths with only 2 engines
active (pure FASTA: BepiPred+EpiDope; ``structure_only``:
DiscoTope+ScanNet, see ``pipeline.py``): reusing the same 4-engine
threshold without recalibrating in those 2 modes was tested directly and breaks
both, in opposite directions (verified on the 17 structures of the
panel, recombining the already cached raw scores into subsets of 2
engines, with no fabricated data): too low for BepiPred+EpiDope
(EpiDope tends to score high without DiscoTope/ScanNet diluting the sum,
which triggers false positives) and too high for DiscoTope+ScanNet
(DiscoTope is negative in most residues of this panel and ScanNet
rarely exceeds 0.5, so most antigens end up without any candidate).

That is why ``FUSION_THRESHOLDS`` has an INDEPENDENTLY calibrated entry
for each supported engine combination (not a single
value mathematically rescaled from the 4-engine one): each threshold was
recomputed from scratch on the sum of that specific combination,
maximizing F1 with the exact grid of unique values of the
fused score (without subsampling), both for the honest estimate of
generalization (leave-one-out cross-validation) and for the final deployment
threshold over the 17 complete structures. The 4-engine threshold
(1.612165) is identical to the one already used in the system's original validation.

ADR: normalization bounds and thresholds (new Table S1, Supplementary
Material)
----------------------------------------------------------------------
The per-engine min-max bounds (``FUSION_BOUNDS``) were computed on the
raw score of each engine across the 17 structures of the validation
panel, independently of whether other engines were active
at the same time (each engine produces the exact same score whether it runs alone or
accompanied). The normalized values are clipped to [0,1]
(``numpy.clip``) so that a future antigen outside this panel, whose
raw score exceeds these bounds, does not produce an out-of-range value or
break the fusion, verified with synthetic extreme values.

The decision thresholds (``FUSION_THRESHOLDS``) were calibrated by maximizing
F1 (micro) over the exact grid of unique values of the fused score in
the 17 complete structures, after honestly validating the expected
generalization via leave-one-out cross-validation (threshold and bounds recomputed
in each fold without seeing the evaluated structure, so as not to overfit to the
very panel that then reports the results).

Micro F1 is prioritized (aggregated over residues of the whole panel, not
averaged per antigen) because it directly reflects the total volume of
false positives, over macro F1. A 3-engine alternative
(excluding EpiDope, weighted by discriminative quality) gives better macro
F1 but worse micro F1; it was discarded for adding a weighting
scheme without improving the metric that matters most for this objective, and
for requiring a rewrite of the engine complementarity argument already established
in the article's Discussion (the fHbp site 1 case, BepiPred+DiscoTope).

ADR: decision window (why the 9aa window is NOT reused)
----------------------------------------------------------------------
Each individual engine uses a sliding window of 9 amino acids with
tolerance for residues below the threshold (see
``src.engines.epitope_mapping.find_valid_windows``) as its DECISION
mechanism. Applying that same window over the already fused signal was tested
explicitly and worsens every combination tested without exception: the
fused score is more peaked residue by residue than any individual
engine (it is a mean of 4 normalized signals), and averaging over 9
residues suppresses real signal instead of smoothing noise. It is discarded as a
decision criterion. What IS needed, and is kept, is grouping
already decided positive residues into contiguous regions (pure adjacency, without
averaging or tolerating gaps) and discarding those shorter than
``MIN_FINAL_PEPTIDE_LENGTH``: on its own it changes no metric
(verified), but it is necessary because a region of 2-3 residues is
unusable in the following phases (NetMHCpan needs a core of
8-11aa, NetMHCIIpan a window of 15aa).

ADR: top-up to TARGET_CANDIDATES + lazy reserve (replaces the
old single-fallback mechanism)
----------------------------------------------------------------------
With the absolute threshold, some antigens with a globally weak signal do not
generate enough valid regions, from none at all to 1-2 of the 3
that Phase 7 can assemble per class (``Settings.CONSTRUCT_TOP_N_PER_CLASS``,
verified: 5-7 of 17 antigens depending on the mode, before applying this
mechanism). Since Phase 7 can only assemble B-cell blocks out of what
survives here, an antigen with fewer candidates than Phase 7 could
use loses a real opportunity for a correct block to reach the
construct. This is the same argument the article's Discussion already uses to
discard strict consensus between engines over the previous union, now
generalized in two directions verified on the panel of 17
structures (see Supplementary Material):

1. The old mechanism (a single fallback window when ZERO
   residues exceed the threshold) solved the most extreme case but left
   untouched antigens with 1-2 regions above threshold, which also do not reach the
   ``TARGET_CANDIDATES`` that Phase 7 could exploit. It is generalized: WHENEVER
   an antigen has fewer than ``TARGET_CANDIDATES`` regions above
   threshold, it is topped up to that number with the windows of
   ``FALLBACK_WINDOW`` residues with the highest mean fusion score in the rest
   of the protein, without overlapping each other or the already accepted regions
   (``_top_windows``, greedy by descending score). ``TARGET_CANDIDATES``
   reuses ``Settings.CONSTRUCT_TOP_N_PER_CLASS`` (3): it is not a new
   hyperparameter to calibrate, it is the same limit that Phase 7 already applies when
   assembling, verified by leave-one-out cross-validation over the
   grid {1..5}, which chooses 3 in 17/17 folds.
2. The previous mechanism only guaranteed that a candidate reached Phase 4
   (self-tolerance/BLASTp), not that it survived it: an antigen whose only
   fallback candidate turns out to be discarded for real human homology
   ends up without any B-cell block in the construct (verified case: 7STS).
   A reserve of ``RESERVE_EXTRA`` additional windows is added (same
   criterion, without overlapping anything already emitted) when the
   top-up of point 1 is activated; these travel with ``candidate_source='reserve'``
   and participate like any other row in Phase 4 (BLASTp) and in the
   percentile ranking of Phase 7 (``_select_bcell_candidates``), which already
   cuts at the best ``top_n`` survivors. Neither of the two phases
   needs to know that a row comes from the reserve for the promotion to
   happen: if Phase 4 discards the ``threshold``/``top_up`` candidates, the
   reserve that DOES survive remains available for Phase 7 to select,
   without a change of control flow in ``pipeline.py`` or in
   ``construct_assembly.py``.

When an antigen already reaches ``TARGET_CANDIDATES`` regions above threshold,
no additional candidate is generated (neither top-up nor reserve): the
mechanism does not touch the volume of antigens where the threshold already works,
verified byte for byte over the complete panel (12 of 17 antigens
identical to the previous mechanism). The ``candidate_source`` column (values
``'threshold'``/``'top_up'``/``'reserve'``) replaces the old
boolean ``fallback`` column, which is kept as a derived alias
(``True`` if ``candidate_source != 'threshold'``) so as not to break
``print_fusion_table`` or the existing tracing through to Phase 4/7.

The following were tested and discarded, with real data from the complete panel (not only
the problematic antigens): a lower secondary "rescue" threshold
(with or without an agreement gate between engines) worsens the
precision/recall trade-off because it does not move the region towards the real epitope, it only
enlarges the same wrong region, the same failure mode already discarded
for the adaptive percentile threshold; reusing the ALREADY calibrated
2-engine thresholds as an OR rule reintroduces the excessive volume that the fusion
removed; recombining the ranking signal (percentile rank, votes per engine,
z-scores) does not improve first-place accuracy because, in the cases that
fail, all 4 engines agree on the same wrong region, so there is no
complementarity to exploit there; relaxing the minimum length of 9aa
inflates the volume without improving F1; expanding spatially in 3D from the
coordinates (a surface patch) gives marginal precision well below
the panel mean. ``FALLBACK_WINDOW=9`` itself was confirmed optimal
(a sweep of window length, N and minimum separation).

Verified over the complete panel (post Phase 4): microF1 rises from 0.306 to
0.353 and macroF1 from 0.287 to 0.383 compared with the previous mechanism, with a single
antigen degraded in a bounded way (3NGB, one additional low-performing
candidate) and zero antigens without any surviving Phase 4 candidate.
Explicit residual limitation, not solved by this mechanism: 2 antigens
(8UP2, 7STR) still have no hit at all because the window that would overlap
the real epitope is discarded by Phase 4 (real human
homology, not a failure of this mechanism); see the article's Discussion.
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
