"""REAL population coverage of HTL/CTL candidates, weighted by allele frequency.

Rationale: ``n_alelos_promiscuos`` (Phase 5/5b) counts how many alleles of the
reference panel classify SB/WB for a candidate, treating each allele equally.
But the real frequencies of HLA alleles vary enormously between each other:
3 globally very common alleles is not the same as 3 rare alleles. The first
candidate would protect a much larger fraction of the real population than the
second, even though both have ``n_alelos_promiscuos == 3``. This module
computes that real fraction using the same methodology as the IEDB
Population Coverage tool (Bui HH, Sidney J, Dinh K, Southwood S, Newman
MJ, Sette A. "Predicting population coverage of T-cell epitope-based
diagnostics and vaccines." BMC Bioinformatics. 2006).

Formula (it assumes Hardy-Weinberg equilibrium, standard Mendelian inheritance):
for a given locus (e.g. HLA-A) and a SUBSET of its alleles (those that a
specific candidate hits SB/WB), the "phenotypic frequency" -the probability
that an individual carries AT LEAST ONE of those alleles at that locus- is:

    phenotypic_frequency = 1 - (1 - sum(frequencies of those alleles)) ** 2

(the binomial formula for "at least one of two independent alleles,
inherited one from each parent, falls in the subset"). Combining several
loci (e.g. HLA-A + HLA-B + HLA-C for MHC-I; DR + DQ + DP for MHC-II),
assuming independence BETWEEN loci (the standard simplification of IEDB's
basic calculation, it does not correct for linkage disequilibrium between genes of
different loci):

    population_coverage = 1 - product(1 - phenotypic_frequency_locus)

Source of allele frequencies: ``reference_db/allele_frequencies/world_pooled_afnd.csv``,
derived from the Allele Frequency Net Database (allelefrequencies.net) via the
MIT-licensed mirror github.com/slowkow/allelefrequencies. For each allele of ``NETMHCPAN_REFERENCE_PANEL``/
``IEDB_REFERENCE_PANEL``, the frequency is the sample-size-weighted average
(``n``) across ALL studies/populations available in AFND
for that allele, a pooled worldwide average, NOT stratified by region/
ethnicity (unlike IEDB's own tool, which does allow choosing a specific
population). Documented as a deliberate simplification, not an
attempt to replicate IEDB's methodology exactly.

2 known gaps in the data, documented in the CSV (the ``method`` column):
* DRB3_0101/DRB3_0202/DRB4_0101/DRB5_0101: no data in AFND (secondary DR
  genes, linked to specific DRB1 haplotypes, reported with far
  less consistency between population studies than the primary genes).
  ``frequency`` is left empty for these 4 alleles. They are EXCLUDED from the calculation
  (frequency 0 is not assumed, which would underestimate the real coverage).
* The panel's DQ/DP alleles are alpha-beta combos (e.g. 'HLA-DQA10501-DQB10201'):
  AFND reports DQA1/DQB1 (or DPA1/DPB1) frequencies SEPARATELY, not those of the
  joint heterodimer. The combo frequency is approximated as the
  PRODUCT of the 2 gene frequencies (an independence assumption; in
  practice DQA1 and DQB1 are in real linkage disequilibrium, so
  this is an approximation, not a directly measured value).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import Settings


def phenotypic_frequency(allele_frequencies: List[float]) -> float:
    """P(un individuo porta AL MENOS UNO de estos alelos en un locus), Hardy-Weinberg.

    ``allele_frequencies``: frecuencias (0-1) de los alelos del subconjunto
    DENTRO de un mismo locus. La suma puede exceder 1 si el subconjunto es
    grande relativo al locus completo; se recorta a 1 antes de aplicar la
    formula (una suma >1 solo puede venir de redondeo/solapamiento en los
    datos de origen, nunca de una probabilidad real >1).
    """
    total = min(sum(allele_frequencies), 1.0)
    return 1 - (1 - total) ** 2


def combined_population_coverage(phenotypic_frequencies_by_locus: List[float]) -> float:
    """Cobertura poblacional combinando varios loci, asumiendo independencia entre ellos."""
    coverage = 1.0
    for pf in phenotypic_frequencies_by_locus:
        coverage *= (1 - pf)
    return 1 - coverage


def load_allele_frequencies(path: str = None) -> pd.DataFrame:
    """Carga la tabla de frecuencias (``panel_allele``/``locus``/``frequency``).

    Args:
        path: Ruta al CSV (default ``Settings.ALLELE_FREQUENCY_PATH``). Filas
            con ``frequency`` vacio (huecos de datos conocidos, ver docstring
            del modulo) se conservan con ``NaN`` -- ``population_coverage_for_alleles``
            las excluye explicitamente en vez de tratarlas como 0.

    Returns:
        DataFrame indexado por ``panel_allele``, columnas ``locus``/``frequency``.
        Vacio si ``path`` no existe o esta vacio (poblacion coverage queda
        desactivada, no es un error).
    """
    resolved = path or Settings.ALLELE_FREQUENCY_PATH
    if not resolved or not Path(resolved).is_file():
        return pd.DataFrame(columns=["locus", "frequency"]).rename_axis("panel_allele")
    table = pd.read_csv(resolved)
    return table.set_index("panel_allele")[["locus", "frequency"]]


def population_coverage_for_alleles(
    alleles: List[str], freq_table: pd.DataFrame
) -> Tuple[Optional[float], List[str]]:
    """Cobertura poblacional de un candidato, dado el subconjunto de alelos que golpea SB/WB.

    Args:
        alleles: Alelos del panel que este candidato clasifico SB/WB
            (``'promiscuous_alleles'``, ver ``netmhciipan_engine``/
            ``netmhcpan_engine``).
        freq_table: Salida de ``load_allele_frequencies``.

    Returns:
        Tupla ``(coverage_pct, alelos_excluidos)``: ``coverage_pct`` en [0,100],
        ``None`` si NINGUNO de los alelos tiene frecuencia conocida (no hay
        base para calcular nada). ``alelos_excluidos`` es la lista de
        ``alleles`` sin frecuencia en ``freq_table`` (huecos de datos
        conocidos, ver docstring del modulo) -- se excluyen del calculo, NO
        se asume frecuencia 0 (subestimaria la cobertura real).
    """
    if not alleles or freq_table.empty:
        return None, list(alleles)

    known = freq_table.reindex(alleles).dropna(subset=["frequency"])
    excluded = [a for a in alleles if a not in known.index]
    if known.empty:
        return None, excluded

    phenotypic_by_locus = [
        phenotypic_frequency(group["frequency"].tolist())
        for _, group in known.groupby("locus")
    ]
    coverage = combined_population_coverage(phenotypic_by_locus)
    return round(coverage * 100, 2), excluded


def annotate_population_coverage(
    df: pd.DataFrame, promiscuous_alleles_col: str = "promiscuous_alleles", freq_table: pd.DataFrame = None,
) -> pd.DataFrame:
    """Anota ``df`` con ``population_coverage_pct`` por fila, a partir de ``promiscuous_alleles_col``.

    Puramente informativo (no filtra ni cambia el orden/seleccion de
    candidatos) -- mismo criterio que ``conservation_pct``/``documented_region``
    en Fase 6b/6c. Si ``freq_table`` esta vacia (``Settings.ALLELE_FREQUENCY_PATH``
    no configurado o archivo ausente), ``population_coverage_pct`` queda
    ``NaN`` en todas las filas -- el pipeline sigue funcionando exactamente
    igual que antes de este modulo.
    """
    result = df.copy()
    if result.empty:
        result["population_coverage_pct"] = pd.Series(dtype=float)
        return result

    table = freq_table if freq_table is not None else load_allele_frequencies()
    if table.empty:
        result["population_coverage_pct"] = pd.NA
        return result

    coverages = []
    for alleles_str in result[promiscuous_alleles_col]:
        alleles = [a for a in str(alleles_str).split(",") if a]
        coverage, _excluded = population_coverage_for_alleles(alleles, table)
        coverages.append(coverage)
    result["population_coverage_pct"] = coverages
    return result
