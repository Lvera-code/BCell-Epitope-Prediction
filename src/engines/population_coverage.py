"""Cobertura poblacional REAL de candidatos HTL/CTL, ponderada por frecuencia alelica.

Motivo: ``n_alelos_promiscuos`` (Fase 5/5b) cuenta cuantos alelos del panel de
referencia clasifican SB/WB para un candidato, tratando cada alelo por igual.
Pero las frecuencias reales de los alelos HLA varian enormemente entre si --
3 alelos muy comunes globalmente no es lo mismo que 3 alelos raros: el primer
candidato protegeria a una fraccion mucho mayor de la poblacion real que el
segundo, aunque ambos tengan ``n_alelos_promiscuos == 3``. Este modulo
calcula esa fraccion real usando la misma metodologia que la herramienta
Population Coverage del IEDB (Bui HH, Sidney J, Dinh K, Southwood S, Newman
MJ, Sette A. "Predicting population coverage of T-cell epitope-based
diagnostics and vaccines." BMC Bioinformatics. 2006).

Formula (asume equilibrio de Hardy-Weinberg, herencia mendeliana estandar):
para un locus dado (ej. HLA-A) y un SUBCONJUNTO de sus alelos (los que un
candidato especifico golpea SB/WB), la "frecuencia fenotipica" -probabilidad
de que un individuo porte AL MENOS UNO de esos alelos en ese locus- es:

    phenotypic_frequency = 1 - (1 - sum(frecuencias de esos alelos)) ** 2

(la formula binomial de "al menos uno de dos alelos independientes,
heredados uno de cada progenitor, cae en el subconjunto"). Combinando varios
loci (ej. HLA-A + HLA-B + HLA-C para MHC-I; DR + DQ + DP para MHC-II),
asumiendo independencia ENTRE loci (simplificacion estandar del calculo
basico de IEDB, no corrige por desequilibrio de ligamiento entre genes de
loci distintos):

    population_coverage = 1 - product(1 - phenotypic_frequency_locus)

Fuente de frecuencias alelicas: ``reference_db/allele_frequencies/world_pooled_afnd.csv``,
derivado de la Allele Frequency Net Database (allelefrequencies.net) via el
mirror MIT-licenciado github.com/slowkow/allelefrequencies. Para cada alelo de ``NETMHCPAN_REFERENCE_PANEL``/
``IEDB_REFERENCE_PANEL``, la frecuencia es el promedio ponderado por tamano
de muestra (``n``) entre TODOS los estudios/poblaciones disponibles en AFND
para ese alelo -- un promedio mundial agrupado, NO estratificado por region/
etnia (a diferencia del propio tool de IEDB, que si permite elegir una
poblacion especifica). Documentado como simplificacion deliberada, no un
intento de replicar exactamente la metodologia de IEDB.

2 huecos conocidos en los datos, documentados en el CSV (columna ``method``):
* DRB3_0101/DRB3_0202/DRB4_0101/DRB5_0101: sin datos en AFND (genes DR
  secundarios, ligados a haplotipos DRB1 especificos, reportados con mucha
  menos consistencia entre estudios poblacionales que los genes primarios).
  ``frequency`` queda vacio para estos 4 alelos -- se EXCLUYEN del calculo
  (no se asume frecuencia 0, que subestimaria la cobertura real).
* Alelos DQ/DP del panel son combos alfa-beta (ej. 'HLA-DQA10501-DQB10201'):
  AFND reporta frecuencias de DQA1/DQB1 (o DPA1/DPB1) por SEPARADO, no del
  heterodimero conjunto. Se aproxima la frecuencia del combo como el
  PRODUCTO de las 2 frecuencias de gen (asuncion de independencia -- en la
  practica DQA1 y DQB1 estan en desequilibrio de ligamiento real, asi que
  esto es una aproximacion, no un valor medido directamente).
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
