"""Tests del calculo de cobertura poblacional real (src/engines/population_coverage.py).

Logica 100% pura (pandas, sin subprocess), formula de Bui et al. 2006 (misma
base que la herramienta Population Coverage del IEDB). Los tests usan tablas
de frecuencia SINTETICAS (no el CSV real de AFND) para verificar la formula
en si, no los numeros biologicos concretos.
"""

import math

import pandas as pd
import pytest

from src.engines.population_coverage import (
    annotate_population_coverage,
    combined_population_coverage,
    load_allele_frequencies,
    phenotypic_frequency,
    population_coverage_for_alleles,
)


# --- phenotypic_frequency: formula de Hardy-Weinberg ------------------------------------


def test_phenotypic_frequency_un_solo_alelo():
    # P(al menos uno de dos alelos independientes cae en un alelo de frecuencia f)
    # = 1 - (1-f)^2.
    assert phenotypic_frequency([0.5]) == pytest.approx(1 - 0.5**2)


def test_phenotypic_frequency_alelo_frecuencia_cero():
    assert phenotypic_frequency([0.0]) == pytest.approx(0.0)


def test_phenotypic_frequency_alelo_frecuencia_uno():
    assert phenotypic_frequency([1.0]) == pytest.approx(1.0)


def test_phenotypic_frequency_varios_alelos_suma():
    # Dos alelos de 0.2 cada uno -> suma 0.4 -> 1 - 0.6^2 = 0.64.
    assert phenotypic_frequency([0.2, 0.2]) == pytest.approx(1 - 0.6**2)


def test_phenotypic_frequency_recorta_suma_mayor_a_uno():
    # Suma > 1 (solo posible por ruido/redondeo de datos de origen) se recorta a 1.0
    # antes de aplicar la formula, nunca produce una "probabilidad" > 1.
    assert phenotypic_frequency([0.7, 0.7]) == pytest.approx(1 - 0.0**2)


# --- combined_population_coverage: combinacion entre loci (independencia) ---------------


def test_combined_coverage_un_solo_locus():
    assert combined_population_coverage([0.5]) == pytest.approx(0.5)


def test_combined_coverage_varios_loci_independientes():
    # 1 - (1-0.5)*(1-0.5) = 0.75
    assert combined_population_coverage([0.5, 0.5]) == pytest.approx(0.75)


def test_combined_coverage_lista_vacia_es_cero():
    assert combined_population_coverage([]) == pytest.approx(0.0)


def test_combined_coverage_cobertura_total_en_cualquier_locus_da_cobertura_total():
    assert combined_population_coverage([1.0, 0.1]) == pytest.approx(1.0)


# --- load_allele_frequencies ------------------------------------------------------------


def test_load_allele_frequencies_archivo_inexistente_devuelve_vacio():
    result = load_allele_frequencies("/no/existe/en/absoluto.csv")
    assert result.empty


def test_load_allele_frequencies_lee_columnas_esperadas(tmp_path):
    csv_path = tmp_path / "freqs.csv"
    csv_path.write_text("panel_allele,locus,frequency,n_total,n_studies,method\nHLA-A01:01,A,0.1,100,5,x\n")

    result = load_allele_frequencies(str(csv_path))

    assert list(result.columns) == ["locus", "frequency"]
    assert result.loc["HLA-A01:01", "frequency"] == pytest.approx(0.1)


# --- population_coverage_for_alleles: por candidato --------------------------------------


def _freq_table(rows):
    """rows: lista de (panel_allele, locus, frequency)."""
    df = pd.DataFrame(rows, columns=["panel_allele", "locus", "frequency"])
    return df.set_index("panel_allele")


def test_population_coverage_un_locus_dos_alelos():
    table = _freq_table([("A1", "A", 0.2), ("A2", "A", 0.3)])

    coverage, excluded = population_coverage_for_alleles(["A1", "A2"], table)

    assert coverage == pytest.approx(round(phenotypic_frequency([0.2, 0.3]) * 100, 2))
    assert excluded == []


def test_population_coverage_combina_dos_loci():
    table = _freq_table([("A1", "A", 0.5), ("B1", "B", 0.5)])

    coverage, excluded = population_coverage_for_alleles(["A1", "B1"], table)

    # Cada locus por separado: phenotypic_frequency([0.5]) = 0.75. Combinados:
    # combined_population_coverage([0.75, 0.75]) = 1 - 0.25*0.25 = 0.9375.
    assert coverage == pytest.approx(93.75)
    assert excluded == []


def test_population_coverage_excluye_alelos_sin_frecuencia_conocida():
    table = _freq_table([("A1", "A", 0.5)])

    coverage, excluded = population_coverage_for_alleles(["A1", "DESCONOCIDO"], table)

    assert coverage == pytest.approx(phenotypic_frequency([0.5]) * 100)
    assert excluded == ["DESCONOCIDO"]


def test_population_coverage_ningun_alelo_conocido_devuelve_none():
    table = _freq_table([("A1", "A", 0.5)])

    coverage, excluded = population_coverage_for_alleles(["DESCONOCIDO"], table)

    assert coverage is None
    assert excluded == ["DESCONOCIDO"]


def test_population_coverage_lista_vacia_devuelve_none():
    table = _freq_table([("A1", "A", 0.5)])

    coverage, excluded = population_coverage_for_alleles([], table)

    assert coverage is None
    assert excluded == []


def test_population_coverage_tabla_vacia_devuelve_none():
    coverage, excluded = population_coverage_for_alleles(["A1"], pd.DataFrame(columns=["locus", "frequency"]))

    assert coverage is None
    assert excluded == ["A1"]


# --- annotate_population_coverage: integracion sobre un DataFrame de candidatos ---------


def test_annotate_population_coverage_anota_por_fila():
    table = _freq_table([("A1", "A", 0.5), ("B1", "B", 0.5), ("A2", "A", 0.1)])
    df = pd.DataFrame({
        "sequence_f5": ["PEP1", "PEP2"],
        "promiscuous_alleles": ["A1,B1", "A2"],
    })

    result = annotate_population_coverage(df, freq_table=table)

    assert result.loc[0, "population_coverage_pct"] == pytest.approx(93.75)
    assert result.loc[1, "population_coverage_pct"] == pytest.approx(round(phenotypic_frequency([0.1]) * 100, 2))


def test_annotate_population_coverage_sin_tabla_deja_nan_sin_romper():
    df = pd.DataFrame({"sequence_f5": ["PEP1"], "promiscuous_alleles": ["A1,A2"]})

    result = annotate_population_coverage(df, freq_table=pd.DataFrame(columns=["locus", "frequency"]))

    assert pd.isna(result.loc[0, "population_coverage_pct"])


def test_annotate_population_coverage_df_vacio():
    result = annotate_population_coverage(pd.DataFrame(columns=["sequence_f5", "promiscuous_alleles"]))
    assert result.empty
    assert "population_coverage_pct" in result.columns


def test_annotate_population_coverage_alelos_vacios_no_rompe():
    table = _freq_table([("A1", "A", 0.5)])
    df = pd.DataFrame({"sequence_f5": ["PEP1"], "promiscuous_alleles": [""]})

    result = annotate_population_coverage(df, freq_table=table)

    assert pd.isna(result.loc[0, "population_coverage_pct"])
