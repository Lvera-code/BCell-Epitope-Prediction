"""Phase 7: automatic assembly of the multi-epitope vaccine construct.

It takes the candidates that already survived the rest of the pipeline (B-cell from
Phase 4/4b/4c, HTL from Phase 5, CTL from Phase 5b), selects the best
``Settings.CONSTRUCT_TOP_N_PER_CLASS`` per class, and concatenates them with the
standard linkers of the multi-epitope vaccine design field into a single
FASTA, together with a 100% traceable metadata table (which individual peptide
contributed each stretch, in what order, which linker was used at each junction). The logic is
ALMOST 100% pure (no subprocess): Phase 6b (``conservation_engine``,
OPTIONAL, only if the user passed ``--panel-conservacion``) runs BLASTp
BEFORE getting here and delivers ``conservation_df`` already computed, exactly
like ``algpred_df``/``stackgly_df``. This invariant is what allows
most of this module to be tested with synthetic DataFrames. The ONLY deliberate
exception: ``_pad_short_bcell_candidates`` DOES invoke AlgPred2/StackGlyEmbed
for real (a re-check of the extended sequence with flanks, see its
docstring), justified because that extended sequence is new content
that Phase 4b/4c never evaluated, not something that came precomputed.

Assembly rules (standard convention of the field, with multiple sources
that agree):

* CTL intra-block linker: ``AAY``, a proteasome cleavage site in
  mammalian cells, it releases each epitope correctly during
  antigen processing and avoids spurious junctional binding epitopes.
* HTL intra-block linker: ``GPGPG``, a universal spacer (Livingston
  et al. 2002), it breaks junctional immunogenicity and preserves the ability
  of each epitope to induce a Th response.
* B-cell intra-block linker: ``KK``, it maintains the specificity of each
  individual epitope.
* Linker between blocks of different classes: ``GPGPG`` (the same universal
  spacer, used consistently in the literature as a bridge between
  blocks, not only intra-HTL).
* Block order: B-cell -> HTL -> CTL. Final decision: there is no strong
  consensus in the literature on the optimal order (the linkers already guarantee
  correct release by antigen processing, independently of
  position); it is anchored on B-cell because that was the project's original
  humoral focus (bnAb/HIV).
* Without adjuvant: an ACTIVE decision not to include one in this version. The
  choice (beta-defensin, PADRE, flagellin, L7/L12, etc.) requires
  biological/strategic criteria specific to the pathogen/host, out of
  scope for this pipeline. ``assemble_construct`` accepts an
  optional ``adjuvant_sequence`` (``None`` by default) to insert it at
  the N-terminus with its own rigid EAAAK linker (Arai et al. 2001) without
  redesigning anything, if it is decided to add one later.

Top-N selection per class (necessary in practice: a real run with
GP120 gave 18 valid candidates in HTL/Phase 5 alone, too many for a
manageable construct):

* B-cell: from ``safe_df`` (Phase 4 'Segura'), ALL candidates enter the
  ranking, including those marked 'Allergen' by AlgPred2 (Phase 4b).
  They used to be discarded here; this was reverted
  because that same AlgPred2 verdict turned out not to correspond with real
  clinical/population data in most of the 9 candidates
  evaluated in the publication validation (7 of 9 marked 'Allergen'
  without a documented allergenicity correlate, correlating instead
  with the width of the region reaching them from Phase 3, not with the real
  chemistry of the antigen). Excluding automatically on the basis of that verdict
  lost real and safe candidates. Just as with glycosylation
  (next paragraph): there are also described antibodies that
  recognize regions with sequence similarity to known allergens
  without that implying real risk, so there is no longer a
  universal mechanistic reason to exclude. Instead, each candidate is
  annotated with its allergenicity verdict (the ``allergen`` column, visible
  in ``source_score_note`` of the final construct) so that the decision
  is informed, not automatic. N-glycosylation (StackGlyEmbed,
  Phase 4c) does not exclude candidates either:
  there are described antibodies that specifically recognize glycosylated
  regions (e.g. HIV envelope epitopes), so discarding
  by glycosylation lost biologically valid candidates without a
  universal mechanistic reason. Instead, each candidate is annotated with
  its glycosylation status (the ``glycosylated`` column, visible in
  ``source_score_note`` of the final construct). A peptide WITHOUT any
  sequon in the Phase 4c report is annotated as ``glycosylated=False``
  (Phase 4c only produces rows for real sequons). The top-N is chosen
  by the HIGHEST of its available ``'{motor}_score'`` values (BepiPred/EpiDope/
  DiscoTope/ScanNet, whichever exists for that row), without a prior filter
  by allergenicity or glycosylation. ``documented_region`` is also annotated
  (Phase 6c, IEDB): ``True`` if the candidate overlaps
  with a protection/neutralization epitope ALREADY DOCUMENTED in some
  studied pathogen, purely informative, just like
  ``conservation_pct``, it does NOT influence the ranking yet. It only applies to
  B-cell (not HTL/CTL, see the docstring of ``src.engines.iedb_engine``).
* HTL/CTL: from the ``'Candidato Valido'`` entries of Phase 5/5b (``build_traceback_report``
  already filters to those), ALL windows are kept regardless
  of whether they overlap a 'Glicosilado' sequon from Phase 4c, the same reasoning
  as B-cell above. ``_glycosylated_regions``/``_overlaps_glyco_region``
  (the same positional overlap mechanism that ``tmbed_engine`` uses
  to mask TM/signal in Phase 3b) now only ANNOTATE the
  ``glycosylated`` column per row, they do not discard. The windows are collapsed by
  ``core_9aa`` (the same binding core evaluated in neighbouring position
  windows is the same prediction, not distinct epitopes, the same criterion
  as ``_deduplicate_protein_mode_windows`` of
  ``netmhciipan_engine.py``/``netmhcpan_engine.py``, but here collapsing
  ALL promiscuity, not only by exact triple), keeping the best
  row (more promiscuous alleles, then lower %Rank; in CTL it additionally prioritizes
  ``netcleave_c_term_match == True`` first). Top-N over those deduplicated
  rows, the same ordering criterion.
* HTL/CTL are NOT excluded by allergenicity (AlgPred2), unlike
  B-cell: AlgPred2 predicts IgE/mast cell potential, which depends on the
  epitope circulating INTACT and exposed to be recognized by an antibody,
  exactly the B-cell scenario. A core of 8-11 aa that lives hidden
  in the MHC groove never circulates free in that way, so the
  mechanistic basis for discarding it as "allergenic" is much
  weaker. That filter is deliberately not applied there, so as not to discard
  valid MHC candidates without a solid biological reason.

Design decision: the "epitope" inserted into the HTL/CTL blocks is
``sequence_f5`` (the complete evaluated window, a 15-mer for HTL, the
complete evaluated peptide for CTL), NOT just ``core_9aa`` (the real binding
core evaluated by NetMHCIIpan/NetMHCpan). The flanking
residues around the core also contribute to recognition
(stability of the peptide-MHC/TCR interaction, correct antigen
processing), so reducing the inserted block to the minimal core
discarded that contribution unnecessarily. ``core_9aa`` is still used
to deduplicate overlapping candidates (the same binding core = the same
prediction, whatever the complete window containing it) and is
recorded in the traceability, but the sequence that enters the final
construct is the complete window.

Handling of overlaps between candidate epitopes: the user's FINAL decision
NOT to merge epitopes of DIFFERENT CLASSES even if they overlap in
position within the source protein (e.g. a B-cell that overlaps
with an HTL). Merging them would break the semantics of the linkers, since
each block expects a peptide of that specific class, not a hybrid. The
INTRA-class merge (two candidates of the SAME class that overlap) is already
resolved by Phase 3 (annotated union of overlapping regions of the same engine
type, before the classes separate in Phase 4b/4c/5/5b), so there is no
need to resolve it again here.

Maximum length of B-cell candidates (``Settings.CONSTRUCT_BCELL_MAX_LENGTH``,
20 aa): unlike HTL/CTL (a fixed-size window imposed by
NetMHCIIpan/NetMHCpan), a B-cell candidate comes from the annotated union of
Phase 3, which can transitively merge overlapping regions of several
engines into a single sequence of tens of aa, longer than a typical linear
B-cell epitope, and without individually evaluating each sub-stretch by
AlgPred2/glycosylation. If the selected candidate exceeds the maximum,
``_trim_long_bcell_candidates`` trims it to the sub-window with the highest
per-residue score (re-reading the raw Phase 2 CSVs, the same anti-scale-bias
criterion of per-engine percentile as the ranking of ``_select_bcell_candidates``)
instead of inserting the complete merged region. The trim happens AFTER
the top-N (the ranking is still by the strength of the complete region), and
is recorded in ``trimmed_from_length`` for traceability.

Native flanks on SHORT B-cell candidates (``Settings.
CONSTRUCT_BCELL_FLANK_THRESHOLD``, 15 aa / ``Settings.
CONSTRUCT_BCELL_FLANK_PADDING``, 3 aa per side): a B-cell candidate at the
length floor (9 aa, ``MIN_FINAL_PEPTIDE_LENGTH`` of ``consensus.py``) does NOT
have the same cushion of context as HTL/CTL. There, the flanks of
``sequence_f5`` were ALREADY evaluated by NetMHCIIpan/NetMHCpan alongside the
core (real computational evidence, we merely decided not to discard it). The
B-cell engines do not have that two-level structure: the limit of the
Phase 3 union IS ALREADY the complete limit of what any engine marked
as antigenic. There is no smaller "core" hidden inside a
larger window that the engine evaluated and we are not using.
Extending a short candidate adds native residues that NO engine flagged as
antigenic: it is a DESIGN assumption (structural context/exposure of the
paratope beyond an arbitrary statistical threshold, a practice recognized
in the multi-epitope vaccine literature), not a tool finding like
the rest of this pipeline. That is why it is annotated separately
(``flanked_from_length``) and is NOT mixed with the same level of certainty as
``bepipred_score``/etc. ``_pad_short_bcell_candidates`` re-checks the
ALREADY extended sequence with AlgPred2/StackGlyEmbed for real (it is new
content, Phase 4b/4c never saw it) and annotates the allergenicity verdict of
the extended version in the ``allergen`` column. The padding is ALWAYS
applied, it is no longer discarded if the extended version comes out 'Allergen'
(this reverts the previous invariant: allergenicity no longer
excludes anything in B-cell, see the paragraph above). Glycosylation,
as before, is only re-annotated (it never excluded in B-cell). It applies only to candidates <15 aa: those that already measure 15-20 aa
are not touched (there is no evidence that they lack context), and those that
exceed 20 aa go to the trim above, never to padding (disjoint ranges
by construction).
"""

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.engines import bepipred_engine, discotope_engine, epidope_engine, scannet_engine
from src.engines.algpred_engine import predict_allergenicity
from src.engines.blast_engine import filter_self_tolerant
from src.engines.stackglyembed_engine import predict_nglycosylation
from src.utils.table_format import Column, print_fixed_width_table

# Motores que producen score POR-RESIDUO cacheado en '{input_stem}_{motor}_raw.csv'
# (mismo nombre de archivo que escribe '_cached_raw_scores'/'_cached_structural_raw_scores'
# en pipeline.py) -- usado por '_trim_long_bcell_candidates' para encontrar la
# sub-ventana de mayor score real dentro de una region B-cell fusionada
# demasiado larga. Reusa las constantes ACCESSION_COLUMN/SCORE_COLUMN de cada
# motor en vez de hardcodear nombres de columna, para no desincronizarse si un
# motor renombra su columna de salida.
_RAW_SCORE_ENGINES = {
    "bepipred": (bepipred_engine.ACCESSION_COLUMN, bepipred_engine.SCORE_COLUMN),
    "epidope": (epidope_engine.ACCESSION_COLUMN, epidope_engine.SCORE_COLUMN),
    "discotope": (discotope_engine.ACCESSION_COLUMN, discotope_engine.SCORE_COLUMN),
    "scannet": (scannet_engine.ACCESSION_COLUMN, scannet_engine.SCORE_COLUMN),
}


class _Block(NamedTuple):
    label: str
    rows: List
    intra_linker: str
    sequence_getter: object  # Callable[[row], str]


def _load_per_residue_scores(output_dir: Path, input_stem: str, accession: str) -> pd.DataFrame:
    """Combina el score por-residuo de ``accession`` entre todos los motores con raw CSV cacheado.

    Cada motor contribuye su score normalizado a percentil DENTRO de su
    propia columna (mismo criterio anti-sesgo-de-escala que el ranking de
    ``_select_bcell_candidates``), promediado entre motores disponibles.
    Los CSV crudos ya estan cacheados por Fase 2 (``_cached_raw_scores``/
    ``_cached_structural_raw_scores`` en ``pipeline.py``): esta funcion solo
    los relee, no vuelve a correr ningun motor. La posicion (1-indexada) se
    infiere del ORDEN de las filas dentro de cada ``accession`` -- los CSV
    crudos de estos 4 motores no traen una columna de posicion explicita,
    solo residuo en orden de secuencia.

    Returns:
        DataFrame con columnas ``position``/``combined_score``, o vacio si
        ningun motor tiene raw CSV cacheado para este ``input_stem`` (p. ej.
        si Fase 2 aun no corrio, lo cual no deberia pasar en Fase 7).
    """
    per_engine_percentiles = []
    for engine_key, (accession_col, score_col) in _RAW_SCORE_ENGINES.items():
        raw_path = output_dir / f"{input_stem}_{engine_key}_raw.csv"
        if not raw_path.is_file():
            continue
        raw_df = pd.read_csv(raw_path)
        if accession_col not in raw_df.columns or score_col not in raw_df.columns:
            continue
        acc_rows = raw_df[raw_df[accession_col] == accession].reset_index(drop=True)
        if acc_rows.empty:
            continue
        percentile = acc_rows[score_col].rank(pct=True)
        percentile.index = acc_rows.index + 1  # posicion 1-indexada
        per_engine_percentiles.append(percentile)

    if not per_engine_percentiles:
        return pd.DataFrame(columns=["position", "combined_score"])

    combined = pd.concat(per_engine_percentiles, axis=1).mean(axis=1, skipna=True)
    return combined.rename("combined_score").rename_axis("position").reset_index()


# Motores ESTRUCTURALES cuyo epitopo puede ser conformacional/discontinuo en
# secuencia lineal (ver Seccion 3.2/discusion del paper): 'origen' contiene su
# etiqueta corta cuando ese motor contribuyo a la region fusionada. threshold_fn
# recibe la lista de scores crudos de la accession y devuelve el umbral de
# produccion real usado en Fase 3 para ESA accession (fijo para DiscoTope-3.0;
# adaptativo por percentil para ScanNet, igual que 'extract_epitopes').
_STRUCTURAL_MOTORS = {
    "discotope": ("Dt", lambda scores: Settings.DISCOTOPE_THRESHOLD),
    "scannet": ("Sn", lambda scores: float(np.percentile(scores, Settings.SCANNET_THRESHOLD_PERCENTILE))),
}


def _annotate_conformational_coverage(
    selected: pd.DataFrame, output_dir: Path, input_stem: str,
) -> pd.DataFrame:
    """Anota que fraccion del parche conformacional real quedo dentro de la ventana sintetizada.

    DiscoTope-3.0 y ScanNet puntuan epitopos conformacionales que pueden ser
    discontinuos en secuencia lineal (Seccion 3.2). Colapsar esa senal a una
    ventana contigua para producir un peptido sintetizable es una decision de
    diseno deliberada (no una propiedad biologica real) -- esta funcion NO
    cambia que se sintetiza, solo hace explicito cuanto del parche real quedo
    fuera: para cada candidato cuyo 'origen' incluya 'Dt' y/o 'Sn', calcula
    que porcentaje de los residuos que ESE motor puntuo por encima de su
    propio umbral de produccion, EN TODA LA PROTEINA, cae dentro del rango
    [start, end] finalmente sintetizado (post pad/trim). Si el candidato
    combina ambos motores estructurales, se promedia la cobertura de cada
    uno. Requiere el raw CSV de Fase 2 ya cacheado (misma fuente que
    ``_load_per_residue_scores``); si no esta disponible, no anota nada para
    ese motor en vez de fallar.
    """
    if selected.empty or "origen" not in selected.columns:
        return selected
    annotated = selected.copy()
    annotated["conformational_coverage_pct"] = pd.NA

    for engine_key, (motor_label, threshold_fn) in _STRUCTURAL_MOTORS.items():
        accession_col, score_col = _RAW_SCORE_ENGINES[engine_key]
        raw_path = output_dir / f"{input_stem}_{engine_key}_raw.csv"
        if not raw_path.is_file():
            continue
        raw_df = pd.read_csv(raw_path)
        if accession_col not in raw_df.columns or score_col not in raw_df.columns:
            continue

        for idx, row in annotated.iterrows():
            if motor_label not in str(row.get("origen", "")):
                continue
            acc_rows = raw_df[raw_df[accession_col] == row["accession"]].reset_index(drop=True)
            if acc_rows.empty:
                continue
            scores = acc_rows[score_col].tolist()
            threshold = threshold_fn(scores)
            over_threshold_positions = [i + 1 for i, s in enumerate(scores) if s >= threshold]  # 1-indexado
            if not over_threshold_positions:
                continue
            inside = sum(1 for p in over_threshold_positions if row["start"] <= p <= row["end"])
            pct = 100.0 * inside / len(over_threshold_positions)
            prev = annotated.at[idx, "conformational_coverage_pct"]
            annotated.at[idx, "conformational_coverage_pct"] = pct if pd.isna(prev) else (prev + pct) / 2.0

    return annotated


def _best_subwindow(sequence: str, start: int, per_residue: pd.DataFrame, max_length: int) -> Tuple[str, int, int]:
    """Recorta ``sequence`` (que arranca en la posicion absoluta ``start``) a ``max_length`` aa.

    Desliza una ventana de ``max_length`` y elige la de mayor score
    combinado (suma de ``per_residue['combined_score']`` dentro del rango).
    Si ``per_residue`` esta vacio (ningun motor con raw cacheado para esa
    ``accession`` -- no deberia ocurrir en uso normal), recorta centrado
    como ultimo recurso en vez de fallar.

    Returns:
        Tupla ``(sub_secuencia, nuevo_start, nuevo_end)``, ambos 1-indexados
        y absolutos (misma convencion que ``start``/``end`` de ``safe_df``).
    """
    length = len(sequence)
    if length <= max_length:
        return sequence, start, start + length - 1

    if per_residue.empty:
        offset = (length - max_length) // 2
    else:
        scores = per_residue.set_index("position")["combined_score"]
        window_sums = [
            scores.reindex(range(start + i, start + i + max_length)).sum(skipna=True)
            for i in range(length - max_length + 1)
        ]
        offset = max(range(len(window_sums)), key=lambda i: window_sums[i])

    new_start = start + offset
    new_end = new_start + max_length - 1
    return sequence[offset:offset + max_length], new_start, new_end


def _load_native_residues(output_dir: Path, input_stem: str, accession: str) -> Optional[pd.Series]:
    """Devuelve la secuencia nativa completa de ``accession`` (posicion 1-indexada -> residuo).

    Relee el primer raw CSV de Fase 2 disponible para esa ``accession`` --
    misma fuente que ``_load_per_residue_scores``, no una copia independiente
    (evita que padding y scoring se desincronicen si algun motor difiere en
    que posiciones parseo, ver ADR de ``position_mapping`` en
    ``consensus.py``). Sirve tanto para el camino FASTA (BepiPred/EpiDope)
    como estructura (DiscoTope/ScanNet, cadena unica) -- los 4 raw CSV traen
    una columna ``Residue`` con el aminoacido en cada posicion.

    Returns:
        ``pd.Series`` indexada por posicion 1-indexada, o ``None`` si ningun
        motor tiene raw CSV cacheado para esa accession (no deberia ocurrir
        en uso normal).
    """
    for engine_key, (accession_col, _) in _RAW_SCORE_ENGINES.items():
        raw_path = output_dir / f"{input_stem}_{engine_key}_raw.csv"
        if not raw_path.is_file():
            continue
        raw_df = pd.read_csv(raw_path)
        if accession_col not in raw_df.columns or "Residue" not in raw_df.columns:
            continue
        acc_rows = raw_df[raw_df[accession_col] == accession].reset_index(drop=True)
        if acc_rows.empty:
            continue
        residues = acc_rows["Residue"].copy()
        residues.index = acc_rows.index + 1  # posicion 1-indexada
        return residues
    return None


def _pad_short_bcell_candidates(
    candidates: pd.DataFrame, output_dir: Path, input_stem: str, threshold: int, padding: int,
) -> pd.DataFrame:
    """Extiende con residuos nativos flanqueantes cada candidato B-cell mas corto que ``threshold``.

    Ver la seccion "Flancos nativos..." del docstring del modulo para el
    razonamiento completo (por que esto NO es lo mismo que ``sequence_f5``
    en HTL/CTL). Resumen operativo:

    1. Para cada candidato con ``len(sequence) < threshold``, propone
       extenderlo ``padding`` aa a cada lado usando la secuencia nativa real
       (``_load_native_residues``), recortado a los limites de la proteina.
    2. Re-chequea TODAS las propuestas de una sola vez con AlgPred2
       (``predict_allergenicity``) y StackGlyEmbed (``predict_nglycosylation``)
       -- unica excepcion de este modulo a "sin subprocess real", ver
       docstring del modulo.
    3. El padding SIEMPRE se aplica: el veredicto de
       alergenicidad de la version extendida se anota en ``allergen``, igual
       que la glicosilacion se anota en ``glycosylated`` -- ninguna de las
       dos excluye nada en B-cell.

    Candidatos ya en el borde de la proteina (``new_start == start`` y
    ``new_end == end``, no hay hacia donde extender) se dejan sin tocar. Si
    ``_load_native_residues`` no encuentra raw cacheado para una accession,
    ese candidato tambien se deja sin tocar (mismo fallback conservador que
    ``_trim_long_bcell_candidates``: preferir no tocar antes que fallar).

    Anota ``flanked_from_length`` (largo original antes del padding, ``NaN``
    si no se extendio) para trazabilidad en ``source_score_note``.
    """
    if candidates.empty or output_dir is None or input_stem is None:
        return candidates

    padded = candidates.copy()
    short_rows = padded[padded["sequence"].str.len() < threshold]
    if short_rows.empty:
        padded["flanked_from_length"] = pd.NA
        return padded

    proposals: Dict[int, Tuple[str, int, int]] = {}
    for idx, row in short_rows.iterrows():
        residues = _load_native_residues(output_dir, input_stem, row["accession"])
        if residues is None or residues.empty:
            continue
        new_start = max(int(residues.index.min()), int(row["start"]) - padding)
        new_end = min(int(residues.index.max()), int(row["end"]) + padding)
        if new_start == row["start"] and new_end == row["end"]:
            continue
        new_seq = "".join(residues.loc[new_start:new_end])
        proposals[idx] = (new_seq, new_start, new_end)

    padded["flanked_from_length"] = pd.NA
    if not proposals:
        return padded

    proposal_seqs = sorted({seq for seq, _, _ in proposals.values()})
    algpred_check = predict_allergenicity(proposal_seqs, output_dir, filename_prefix=f"{input_stem}_bcell_flank_")
    glyco_check = predict_nglycosylation(proposal_seqs, output_dir, filename_prefix=f"{input_stem}_bcell_flank_")
    allergen_seqs = set(algpred_check[algpred_check["algpred_veredicto"] == "Allergen"]["sequence"]) \
        if not algpred_check.empty else set()
    glyco_risky_seqs = set(glyco_check[glyco_check["stackglyembed_veredicto"] == "Glicosilado"]["sequence"]) \
        if not glyco_check.empty else set()

    for idx, (new_seq, new_start, new_end) in proposals.items():
        original_length = len(padded.at[idx, "sequence"])
        padded.at[idx, "sequence"] = new_seq
        padded.at[idx, "start"] = new_start
        padded.at[idx, "end"] = new_end
        padded.at[idx, "allergen"] = new_seq in allergen_seqs
        padded.at[idx, "glycosylated"] = new_seq in glyco_risky_seqs
        padded.at[idx, "flanked_from_length"] = original_length

    return padded


def _trim_long_bcell_candidates(
    candidates: pd.DataFrame, output_dir: Path, input_stem: str, max_length: int,
) -> pd.DataFrame:
    """Recorta cada candidato B-cell que supere ``max_length`` a su mejor sub-ventana (ver ``_best_subwindow``).

    Solo se aplica a los candidatos YA SELECCIONADOS (top-N), no a todo
    ``safe_df``: el ranking sigue siendo por la region fusionada completa
    (``_select_bcell_candidates``), el recorte es puramente para la
    secuencia que efectivamente entra al constructo. Anota
    ``trimmed_from_length`` (largo original, ``NaN`` si no se recorto) para
    trazabilidad en ``source_score_note``.
    """
    if candidates.empty:
        return candidates
    trimmed = candidates.copy()
    original_lengths = trimmed["sequence"].str.len()
    for idx, row in trimmed.iterrows():
        if len(row["sequence"]) <= max_length:
            continue
        per_residue = _load_per_residue_scores(output_dir, input_stem, row["accession"])
        new_seq, new_start, new_end = _best_subwindow(row["sequence"], int(row["start"]), per_residue, max_length)
        trimmed.at[idx, "sequence"] = new_seq
        trimmed.at[idx, "start"] = new_start
        trimmed.at[idx, "end"] = new_end
    trimmed["trimmed_from_length"] = original_lengths.where(original_lengths > max_length)
    return trimmed


def _select_bcell_candidates(
    safe_df: pd.DataFrame, algpred_df: pd.DataFrame, stackgly_df: pd.DataFrame, top_n: int,
    conservation_map: Optional[Dict[str, float]] = None,
    iedb_matched_seqs: Optional[set] = None,
    output_dir: Optional[Path] = None,
    input_stem: Optional[str] = None,
    max_length: Optional[int] = None,
    flank_threshold: Optional[int] = None,
    flank_padding: Optional[int] = None,
    blast_db: str = Settings.BLAST_HUMAN_DB,
    identity_threshold: float = Settings.BLAST_IDENTITY_THRESHOLD,
) -> pd.DataFrame:
    """Anota ``safe_df`` con alergenicidad/glicosilacion/conservacion/region-documentada (sin excluir), rankea por consenso entre motores, top-N.

    Ya no filtra por 'Non-Allergen' -- todos los
    candidatos de ``safe_df`` entran al ranking, anotados con su veredicto
    de AlgPred2 en la columna ``allergen`` (``True`` = 'Allergen'). Ver el
    parrafo de alergenicidad en el docstring del modulo para el porque.

    El ranking usa el percentil de cada candidato DENTRO de la columna
    ``{motor}_score`` a la que pertenece (``rank(pct=True)``), no el score
    crudo: los motores no comparten escala (DiscoTope-3.0 es un score
    calibrado que puede superar 1, mientras BepiPred/EpiDope/ScanNet son
    probabilidades en [0,1]), asi que promediar/maximizar los valores crudos
    sesgaba el ranking hacia el motor con el rango numerico mas grande, no
    hacia el candidato mas fuerte. El promedio de percentiles (``mean``, no
    ``max``) premia el CONSENSO entre motores: un candidato visto como fuerte
    por varios motores le gana a uno que un solo motor ve como excepcional.
    """
    if safe_df.empty:
        return safe_df

    allergen_seqs = set(algpred_df[algpred_df["algpred_veredicto"] == "Allergen"]["sequence"]) \
        if not algpred_df.empty else set()
    glyco_risky_seqs = set(stackgly_df[stackgly_df["stackglyembed_veredicto"] == "Glicosilado"]["sequence"]) \
        if not stackgly_df.empty else set()

    candidates = safe_df.copy()
    if candidates.empty:
        return candidates
    candidates["allergen"] = candidates["sequence"].isin(allergen_seqs)
    candidates["glycosylated"] = candidates["sequence"].isin(glyco_risky_seqs)
    if conservation_map:
        candidates["conservation_pct"] = candidates["sequence"].map(conservation_map)
    if iedb_matched_seqs:
        candidates["documented_region"] = candidates["sequence"].isin(iedb_matched_seqs)

    score_cols = [c for c in candidates.columns if c.endswith("_score")]
    if score_cols:
        percentiles = candidates[score_cols].rank(pct=True, na_option="keep")
        candidates["_rank_score"] = percentiles.mean(axis=1, skipna=True)
    else:
        candidates["_rank_score"] = 0.0
    candidates = candidates.sort_values("_rank_score", ascending=False)
    selected = candidates.head(top_n).drop(columns="_rank_score")

    if output_dir is not None and input_stem is not None:
        selected = _pad_short_bcell_candidates(
            selected, output_dir, input_stem,
            flank_threshold or Settings.CONSTRUCT_BCELL_FLANK_THRESHOLD,
            flank_padding or Settings.CONSTRUCT_BCELL_FLANK_PADDING,
        )
        selected = _trim_long_bcell_candidates(
            selected, output_dir, input_stem, max_length or Settings.CONSTRUCT_BCELL_MAX_LENGTH
        )
        # Cobertura del parche conformacional real (Dt/Sn) dentro de la ventana YA
        # recortada/extendida -- debe ir DESPUES de pad/trim, sobre el rango
        # [start, end] que efectivamente se sintetiza, no sobre la region fusionada
        # original de Fase 3 (ver docstring de '_annotate_conformational_coverage').
        selected = _annotate_conformational_coverage(selected, output_dir, input_stem)
        # Re-chequeo de autotolerancia sobre la secuencia FINAL (ya recortada/extendida),
        # DESPUES de pad/trim: ni la Fase 4 original (corrio sobre la region padre, potencialmente
        # mas larga) ni el padding de flancos (solo re-chequea AlgPred2/StackGlyEmbed) cubren esto
        # -- ver docstring de 'blast_engine.filter_self_tolerant'.
        selected = filter_self_tolerant(selected, "sequence", db_path=blast_db, identity_threshold=identity_threshold)
    return selected


def _dedupe_by_core(candidate_df: pd.DataFrame, sort_columns: List[Tuple[str, bool]]) -> pd.DataFrame:
    """Colapsa filas con el mismo 'core_9aa', quedandose con la 'mejor' segun ``sort_columns``.

    ``sort_columns``: lista de ``(columna, ascending)`` en orden de prioridad
    (primera = criterio principal de desempate).
    """
    if candidate_df.empty:
        return candidate_df
    by_cols = [c for c, _ in sort_columns]
    ascending = [asc for _, asc in sort_columns]
    ordered = candidate_df.sort_values(by=by_cols, ascending=ascending)
    return ordered.drop_duplicates(subset="core_9aa", keep="first")


def _glycosylated_regions(safe_df: pd.DataFrame, stackgly_df: pd.DataFrame) -> pd.DataFrame:
    """Traduce cada sequon 'Glicosilado' de Fase 4c a su posicion ABSOLUTA en la proteina completa.

    ``stackgly_df['sequon_position']`` es 1-indexado pero LOCAL al peptido
    'Segura' padre (ver ``stackglyembed_engine.print_glycosylation_report``),
    no a la proteina completa -- para poder comparar contra el ``start``/``end``
    (absolutos) de ``htl_df``/``ctl_df`` hay que sumarle el ``start`` de ese
    peptido padre en ``safe_df``.

    Returns:
        DataFrame con columnas ``accession``/``start``/``end`` (1-indexado,
        3 residuos del motivo N-X-[S/T]), una fila por sequon glicosilado.
        Vacio si no hay ningun sequon 'Glicosilado'.
    """
    if stackgly_df.empty or safe_df.empty:
        return pd.DataFrame(columns=["accession", "start", "end"])
    glyco = stackgly_df[stackgly_df["stackglyembed_veredicto"] == "Glicosilado"]
    if glyco.empty:
        return pd.DataFrame(columns=["accession", "start", "end"])
    parent = safe_df[["accession", "start", "sequence"]].rename(columns={"start": "parent_start"})
    merged = glyco.merge(parent, on="sequence", how="inner")
    merged["start"] = merged["parent_start"] + merged["sequon_position"] - 1
    merged["end"] = merged["start"] + 2
    return merged[["accession", "start", "end"]]


def _overlaps_any_region(row, regions: pd.DataFrame) -> bool:
    """Indica si ``row`` (con ``accession``/``start``/``end``) solapa alguna fila de ``regions``.

    Mismo mecanismo de solapamiento por posicion que
    ``tmbed_engine.discard_overlapping_regions`` usa para enmascarar TM/senal
    en Fase 3b, generalizado para cualquier conjunto de regiones nativas
    (sequones glicosilados -- ver :func:`_overlaps_glyco_region` --, o
    bloques ya seleccionados de otra clase -- ver
    :func:`_deduplicate_cross_class`).
    """
    if regions.empty:
        return False
    acc_regions = regions[regions["accession"] == row["accession"]]
    if acc_regions.empty:
        return False
    return bool(((acc_regions["start"] <= row["end"]) & (acc_regions["end"] >= row["start"])).any())


def _overlaps_glyco_region(row, glyco_regions: pd.DataFrame) -> bool:
    """Indica si ``row`` (con ``accession``/``start``/``end``) solapa algun sequon glicosilado.

    Aplicado a nivel de candidato individual HTL/CTL (ver
    :func:`_overlaps_any_region` para el mecanismo generico).
    """
    return _overlaps_any_region(row, glyco_regions)


def _deduplicate_cross_class(candidates: pd.DataFrame, already_selected: pd.DataFrame) -> pd.DataFrame:
    """Excluye de ``candidates`` cualquier fila que solape (misma accession, mismo
    rango nativo ``start``-``end``) un bloque YA seleccionado de una clase de
    mayor prioridad en el orden de ensamblaje (B-cell -> HTL -> CTL).

    Corrige una limitacion real encontrada en la validacion del sistema: la
    deduplicacion de solapamientos anterior (``_dedupe_by_core``) operaba
    solo DENTRO de cada clase, nunca entre clases distintas -- un mismo
    tramo del antigeno nativo podia aparecer duplicado en el constructo
    final, una vez como bloque B-cell y otra vez anidado dentro de una
    ventana HTL o CTL (verificado en 9 de 17 estructuras del panel de
    validacion, no solo en los 2 casos documentados originalmente). Se
    aplica ANTES del corte ``top_n`` (no despues) para que, si un candidato
    queda excluido por solapamiento, el siguiente mejor candidato no
    solapante ocupe su lugar en vez de dejar la clase con menos de
    ``top_n`` bloques.

    Args:
        candidates: Candidatos de la clase actual (HTL o CTL), antes de
            aplicar ``top_n``.
        already_selected: Bloques YA seleccionados de clases de mayor
            prioridad (p. ej. B-cell al filtrar HTL; B-cell + HTL al
            filtrar CTL). Vacio -- no excluye nada.

    Returns:
        ``candidates`` sin las filas que solapan ``already_selected``.
    """
    if candidates.empty or already_selected.empty:
        return candidates
    mask = ~candidates.apply(lambda r: _overlaps_any_region(r, already_selected), axis=1)
    return candidates[mask]


def _select_htl_candidates(
    htl_df: pd.DataFrame, glyco_regions: pd.DataFrame, top_n: int,
    conservation_map: Optional[Dict[str, float]] = None,
    exclude_overlapping: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Colapsa por 'core_9aa' (mejor promiscuidad/%Rank), anota glicosilacion/conservacion (sin excluir), top-N.

    Un glicano dentro del nucleo de 9 aa que se mete en el surco del MHC-II
    podria en teoria interferir con la union, pero existen anticuerpos/celulas
    T descritos que reconocen especificamente epitopos glicosilados (mismo
    razonamiento que B-cell, ver docstring de modulo) -- ya no se descarta por
    esto, solo se anota en la columna ``glycosylated`` para decision informada.
    ``conservation_map`` (``sequence_f5`` -> ``conservation_pct``, de Fase 6b)
    se mapea igual, ausente si no se paso ``--panel-conservacion``.
    ``exclude_overlapping`` (bloques B-cell ya seleccionados) se aplica ANTES
    del corte ``top_n``, ver :func:`_deduplicate_cross_class`.
    """
    if htl_df.empty:
        return htl_df
    candidates = htl_df.copy()
    candidates["glycosylated"] = candidates.apply(lambda r: _overlaps_glyco_region(r, glyco_regions), axis=1)
    if conservation_map:
        candidates["conservation_pct"] = candidates["sequence_f5"].map(conservation_map)
    if exclude_overlapping is not None:
        candidates = _deduplicate_cross_class(candidates, exclude_overlapping)
    sort_columns = [("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(candidates, sort_columns)
    deduped = deduped.sort_values(by=[c for c, _ in sort_columns], ascending=[a for _, a in sort_columns])
    return deduped.head(top_n)


def _select_ctl_candidates(
    ctl_df: pd.DataFrame, glyco_regions: pd.DataFrame, top_n: int,
    conservation_map: Optional[Dict[str, float]] = None,
    exclude_overlapping: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Colapsa por 'core_9aa', anota glicosilacion/conservacion (sin excluir), prioriza NetCleave/promiscuidad/%Rank, top-N.

    Mismo criterio que ``_select_htl_candidates`` (ver ese docstring): ya no
    se descarta por glicosilacion, solo se anota; mismo mapeo opcional de
    conservacion, mismo mecanismo de ``exclude_overlapping`` ANTES del corte
    ``top_n`` (aca, tipicamente bloques B-cell + HTL ya seleccionados).
    """
    if ctl_df.empty:
        return ctl_df
    candidates = ctl_df.copy()
    candidates["glycosylated"] = candidates.apply(lambda r: _overlaps_glyco_region(r, glyco_regions), axis=1)
    if conservation_map:
        candidates["conservation_pct"] = candidates["sequence_f5"].map(conservation_map)
    if exclude_overlapping is not None:
        candidates = _deduplicate_cross_class(candidates, exclude_overlapping)
    sort_columns = [("netcleave_c_term_match", False), ("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(candidates, sort_columns)
    deduped = deduped.sort_values(by=[c for c, _ in sort_columns], ascending=[a for _, a in sort_columns])
    return deduped.head(top_n)


def _score_note(row, fields: List[str]) -> str:
    parts = []
    for field in fields:
        value = getattr(row, field, None)
        # pd.isna() cubre tanto NaN de float (ej. 'trimmed_from_length', via
        # Series.where) como pd.NA (ej. 'flanked_from_length') -- el chequeo
        # anterior (isinstance(value, float)) solo filtraba el primero.
        if value is not None and not pd.isna(value):
            parts.append(f"{field}={value}")
    return ", ".join(parts)


def assemble_construct(
    safe_df: pd.DataFrame,
    algpred_df: pd.DataFrame,
    stackgly_df: pd.DataFrame,
    htl_df: pd.DataFrame,
    ctl_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    input_stem: Optional[str] = None,
    top_n_per_class: int = None,
    bcell_max_length: Optional[int] = None,
    bcell_flank_threshold: Optional[int] = None,
    bcell_flank_padding: Optional[int] = None,
    adjuvant_sequence: Optional[str] = None,
    conservation_df: Optional[pd.DataFrame] = None,
    iedb_df: Optional[pd.DataFrame] = None,
    blast_db: str = Settings.BLAST_HUMAN_DB,
    identity_threshold: float = Settings.BLAST_IDENTITY_THRESHOLD,
) -> Tuple[str, pd.DataFrame]:
    """Selecciona candidatos top-N por clase y ensambla el constructo final.

    Args:
        safe_df: Salida de Fase 4 (``status == 'Segura'``).
        algpred_df: Salida de Fase 4b (``predict_allergenicity``).
        stackgly_df: Salida de Fase 4c (``predict_nglycosylation``).
        htl_df: Salida de Fase 5 (``candidatos_finales.csv`` / ``build_traceback_report``).
        ctl_df: Salida de Fase 5b (``candidatos_finales_mhc1.csv``, con anotacion NetCleave).
        output_dir: Carpeta de ``outputs`` -- SOLO se usa para releer los
            raw CSV por-residuo de Fase 2 y recortar candidatos B-cell que
            excedan ``Settings.CONSTRUCT_BCELL_MAX_LENGTH`` (ver
            ``_trim_long_bcell_candidates``). ``None`` desactiva el recorte
            (compatibilidad con llamadas/tests que no lo necesitan).
        input_stem: Nombre del archivo de entrada sin extension, para ubicar
            esos mismos raw CSV (``{input_stem}_{motor}_raw.csv``).
        top_n_per_class: Maximo de epitopos por clase (default ``Settings.CONSTRUCT_TOP_N_PER_CLASS``, 3).
        bcell_max_length: Largo maximo de un candidato B-cell antes de
            recortarse a su mejor sub-ventana (default
            ``Settings.CONSTRUCT_BCELL_MAX_LENGTH``, 20). Sin efecto si
            ``output_dir``/``input_stem`` son ``None`` (recorte desactivado).
        bcell_flank_threshold: Largo por debajo del cual un candidato B-cell
            se extiende con flancos nativos (default ``Settings.
            CONSTRUCT_BCELL_FLANK_THRESHOLD``, 15). bcell_flank_padding:
            cuantos aa nativos agregar a cada lado (default ``Settings.
            CONSTRUCT_BCELL_FLANK_PADDING``, 3). Ver
            ``_pad_short_bcell_candidates`` -- re-chequea la version
            extendida con AlgPred2/StackGlyEmbed de verdad, unica excepcion
            de este modulo a "sin subprocess". Sin efecto si
            ``output_dir``/``input_stem`` son ``None``.
        adjuvant_sequence: Secuencia de adjuvante opcional a anteponer en el
            N-terminal (con linker rigido EAAAK). ``None`` por defecto -- ver
            docstring del modulo, ningun adjuvante se elige automaticamente.
        conservation_df: Salida OPCIONAL de Fase 6b (``conservation_engine``),
            con columnas ``sequence``/``conservation_pct``. ``None`` por
            defecto (el usuario no paso ``--panel-conservacion``) -- en ese
            caso no se anota nada, mismo comportamiento que antes de este
            parametro. Igual que ``glycosylated``, es puramente informativo
            (visible en ``source_score_note``), NO influye en la seleccion
            top-N: la decision de usarlo como criterio de ranking queda
            pendiente de trabajo futuro.
        iedb_df: Salida de Fase 6c (``iedb_engine``, siempre corre), con
            columnas ``sequence``/``epitope_sequence``/... Anota
            ``documented_region`` (bool) SOLO en candidatos B-cell -- IEDB
            aca es especificamente ensayos de anticuerpo, no aplica al
            mecanismo de HTL/CTL (ver docstring de ``iedb_engine``). Igual
            que ``conservation_pct``, puramente informativo.
        blast_db/identity_threshold: Mismos parametros que Fase 4, para el
            re-chequeo de autotolerancia del candidato B-cell YA recortado/
            extendido (``filter_self_tolerant``, aplicado DESPUES de
            ``bcell_max_length``/``bcell_flank_*`` dentro de
            ``_select_bcell_candidates``). A diferencia de esos otros
            parametros, este SI puede excluir un candidato del top-N (no es
            informativo): Fase 4 corrio sobre la region padre, potencialmente
            mucho mas larga, asi que no cubre la secuencia final por si sola.
            Sin efecto si ``output_dir``/``input_stem`` son ``None``.

    Returns:
        Tupla ``(construct_sequence, metadata_df)``: la secuencia del
        constructo ensamblado (string), y un DataFrame con una fila por
        SEGMENTO (epitopo o linker, en orden) con columnas ``block``,
        ``sequence``, ``start``/``end`` (1-indexado, posicion en el
        constructo final), ``source_accession``, ``source_start``,
        ``source_end`` (``None`` para segmentos de linker/adjuvante) y
        ``source_score_note`` (resumen legible de los scores que motivaron
        la seleccion, vacio para linkers). Concatenar
        ``metadata_df['sequence']`` en orden reconstruye ``construct_sequence``
        exactamente.

        Si las 3 clases quedan vacias tras la seleccion, devuelve
        ``("", DataFrame vacio)`` -- no hay ningun candidato con el cual ensamblar nada.
    """
    top_n = top_n_per_class if top_n_per_class is not None else Settings.CONSTRUCT_TOP_N_PER_CLASS
    conservation_map = (
        dict(zip(conservation_df["sequence"], conservation_df["conservation_pct"]))
        if conservation_df is not None and not conservation_df.empty else None
    )
    iedb_matched_seqs = (
        set(iedb_df["sequence"]) if iedb_df is not None and not iedb_df.empty else None
    )

    glyco_regions = _glycosylated_regions(safe_df, stackgly_df)
    bcell_selected = _select_bcell_candidates(
        safe_df, algpred_df, stackgly_df, top_n, conservation_map, iedb_matched_seqs,
        output_dir=output_dir, input_stem=input_stem, max_length=bcell_max_length,
        flank_threshold=bcell_flank_threshold, flank_padding=bcell_flank_padding,
        blast_db=blast_db, identity_threshold=identity_threshold,
    )
    # Deduplicacion cross-clase (ver _deduplicate_cross_class): un candidato
    # HTL/CTL que solape un bloque YA seleccionado de mayor prioridad
    # (B-cell -> HTL -> CTL) se excluye ANTES del corte top_n, para que el
    # siguiente mejor candidato no solapante ocupe su lugar.
    htl_selected = _select_htl_candidates(
        htl_df, glyco_regions, top_n, conservation_map, exclude_overlapping=bcell_selected,
    )
    bcell_and_htl = pd.concat([bcell_selected, htl_selected], ignore_index=True, sort=False)
    ctl_selected = _select_ctl_candidates(
        ctl_df, glyco_regions, top_n, conservation_map, exclude_overlapping=bcell_and_htl,
    )

    blocks: List[_Block] = []
    if not bcell_selected.empty:
        blocks.append(_Block(
            "B-cell", list(bcell_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_BCELL, lambda r: r.sequence,
        ))
    if not htl_selected.empty:
        blocks.append(_Block(
            "HTL", list(htl_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_HTL, lambda r: r.sequence_f5,
        ))
    if not ctl_selected.empty:
        blocks.append(_Block(
            "CTL", list(ctl_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_CTL, lambda r: r.sequence_f5,
        ))

    if not blocks and not adjuvant_sequence:
        return "", pd.DataFrame(columns=[
            "block", "sequence", "start", "end", "source_accession",
            "source_start", "source_end", "source_score_note",
        ])

    segments = []
    cursor = 1

    def _add(block_label: str, sequence: str, source_row=None, score_note: str = "") -> None:
        nonlocal cursor
        start = cursor
        end = cursor + len(sequence) - 1
        segments.append(
            {
                "block": block_label,
                "sequence": sequence,
                "start": start,
                "end": end,
                "source_accession": getattr(source_row, "accession", None),
                "source_start": getattr(source_row, "start", None),
                "source_end": getattr(source_row, "end", None),
                "source_score_note": score_note,
            }
        )
        cursor = end + 1

    if adjuvant_sequence:
        _add("Adjuvante", adjuvant_sequence)
        _add("Linker", Settings.CONSTRUCT_LINKER_ADJUVANTE)

    bcell_score_fields = [
        "bepipred_score", "epidope_score", "discotope_score", "scannet_score",
        "allergen", "glycosylated", "conservation_pct", "documented_region",
        "trimmed_from_length", "flanked_from_length", "conformational_coverage_pct",
    ]
    htl_ctl_score_fields = [
        "n_alelos_promiscuos", "n_alelos_evaluados", "population_coverage_pct",
        "min_rank_el", "glycosylated", "conservation_pct",
    ]

    for block_idx, block in enumerate(blocks):
        score_fields = bcell_score_fields if block.label == "B-cell" else htl_ctl_score_fields
        if block.label == "CTL":
            score_fields = score_fields + ["netcleave_c_term_match", "netcleave_c_term_score"]

        for i, row in enumerate(block.rows):
            _add(block.label, block.sequence_getter(row), row, _score_note(row, score_fields))
            if i < len(block.rows) - 1:
                _add(f"Linker (intra-{block.label})", block.intra_linker)
        if block_idx < len(blocks) - 1:
            _add("Linker (inter-bloque)", Settings.CONSTRUCT_LINKER_INTERBLOQUE)

    construct_sequence = "".join(s["sequence"] for s in segments)
    metadata_df = pd.DataFrame(segments)
    return construct_sequence, metadata_df


def print_construct_breakdown(metadata_df: pd.DataFrame) -> None:
    """Imprime el constructo desglosado: Bloque/Start/End/Origen, un segmento por fila, en orden.

    Analogo al resto de tablas de desglose del pipeline (ver
    ``algpred_engine.print_allergenicity_report``): la consola solo mostraba
    antes la secuencia concatenada del constructo completo, sin ver de que
    peptidos/linkers individuales esta hecho ni de donde salio cada uno. La
    secuencia de cada segmento se ve coloreada en ``print_construct_colored``
    (linkers en rojo), no repetida acá como texto plano -- esta tabla es
    solo la trazabilidad (posicion + origen); el detalle completo (score que
    motivo la seleccion, posicion en la proteina de origen) sigue persistido
    en ``constructo_metadata.csv``.
    """
    if metadata_df.empty:
        return

    def _origen(r) -> str:
        return r.source_accession if pd.notna(r.source_accession) else "-"

    columns = [
        Column("Bloque", lambda r: r.block, 22, "<"),
        Column("Start", lambda r: str(r.start), 6, ">"),
        Column("End", lambda r: str(r.end), 6, ">"),
        # prefix="  ": "End" es right-aligned, sin este separador explicito
        # quedaria pegada a "Origen" (mismo caso que tmbed_engine.py/signalp_engine.py).
        Column("Origen", _origen, 0, "<", prefix="  "),
    ]
    print_fixed_width_table(metadata_df.itertuples(index=False), columns)


# Amarillo/negrita para los linkers dentro del constructo impreso -- MISMO
# color que el nucleo MHC (netmhciipan_engine.py) y el sequon
# (stackglyembed_engine.py), para mantener un unico codigo de resaltado
# consistente en todo el pipeline.
_LINKER_ANSI_START = "\033[1;33m"
_LINKER_ANSI_END = "\033[0m"


def print_construct_colored(metadata_df: pd.DataFrame) -> None:
    """Imprime la secuencia completa del constructo con los linkers en amarillo.

    Complementa ``print_construct_breakdown`` (da start/end/origen por
    segmento pero no la secuencia): acá se ve el constructo entero tal cual
    queda en el FASTA, con cada tramo de linker coloreado para diferenciarlo
    a simple vista de los tramos de epitopo real. No es una tabla de ancho
    fijo -- es una sola linea concatenada, asi que no aplica el problema de
    alineado de ``table_format.Column`` (los codigos ANSI solo importan para
    ``len()`` cuando hay padding de columna de por medio).
    """
    if metadata_df.empty:
        return
    parts = [
        f"{_LINKER_ANSI_START}{row.sequence}{_LINKER_ANSI_END}" if row.block.startswith("Linker") else row.sequence
        for row in metadata_df.itertuples(index=False)
    ]
    print("".join(parts))


def print_multi_accession_warning(metadata_df: pd.DataFrame) -> None:
    """Advierte si el constructo se armo con candidatos de mas de un accession distinto.

    Chequeo simple por CANTIDAD de accessions distintos, sin evaluar similitud
    de secuencia entre ellos -- no distingue "cepas relacionadas del mismo
    patogeno" (ej. distintas cepas de VIH, resultado esperado y correcto) de
    "antigenos sin relacion" (ej. mezclar dos patogenos distintos en un mismo
    FASTA de entrada). Queda a criterio de quien lee la advertencia: revisar
    la columna 'Origen' de ``print_construct_breakdown`` para decidir si el
    resultado tiene sentido biologico.
    """
    accessions = metadata_df["source_accession"].dropna().unique()
    if len(accessions) > 1:
        print(f"\n[AVISO] Constructo realizado a partir de diferentes accessions: {', '.join(sorted(accessions))}")
