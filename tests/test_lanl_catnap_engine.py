"""Tests de la Fase 6 (src/engines/lanl_catnap_engine.py): cruce de candidatos contra
epitopos lineales de bnAb conocidos (LANL Immunology DB) y potencia de neutralizacion
(CATNAP).

A diferencia de los demas motores nuevos, este NO invoca ningun subprocess (pandas puro
sobre CSVs locales, ver docstring del modulo): se prueba de punta a punta con archivos
``ab_all.csv``/``abs_*.txt`` sinteticos escritos en ``tmp_path``, replicando el formato real
(cabecera de LANL con 613 columnas 'Note N', formato tab-separado de CATNAP).
"""

import csv

import pandas as pd
import pytest

from src.engines.lanl_catnap_engine import (
    _load_bnab_epitopes,
    _load_catnap_potency,
    _longest_common_substring_len,
    query_bnab_crossref,
)

_LANL_HEADER = [
    "Antibody record #", "Table", "Antibody name (alias)", "Epitope", "Epitope name",
    "HXB2 protein location", "Author location (strain)", "HXB2 DNA location", "Subtype",
    "Research contact", "Binding region", "Modified from", "Neutralizing",
    "ADCC effector function", "Antibody type", "Species", "Isotype",
]


def _write_lanl_csv(path, rows):
    """Escribe un ``ab_all.csv`` sintetico. ``rows`` es una lista de dicts con las
    columnas usadas por el motor; el resto de ``_LANL_HEADER`` queda vacio."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_LANL_HEADER)
        for row in rows:
            writer.writerow([row.get(col, "") for col in _LANL_HEADER])


def _write_catnap_abs(path, rows):
    """Escribe un ``abs_*.txt`` sintetico (tab-separado, subconjunto de columnas reales)."""
    df = pd.DataFrame(rows, columns=["Name", "Mean panel IC50", "# of viruses tested"])
    df.to_csv(path, sep="\t", index=False)


# --- _longest_common_substring_len -------------------------------------------------------


def test_lcs_substring_completa():
    assert _longest_common_substring_len("ABCDEF", "ABCDEF") == 6


def test_lcs_sin_solapamiento():
    assert _longest_common_substring_len("AAAA", "BBBB") == 0


def test_lcs_substring_parcial():
    assert _longest_common_substring_len("XXXWEKIRLRPGGXXX", "GELDRWEKIRLRPGG") == 10  # 'WEKIRLRPGG'


def test_lcs_string_vacia():
    assert _longest_common_substring_len("", "ABC") == 0
    assert _longest_common_substring_len("ABC", "") == 0


# --- _load_bnab_epitopes: filtrado de epitopos lineales utilizables ---------------------


def test_load_bnab_epitopes_omite_conformacionales_y_compuestos(tmp_path):
    csv_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(
        csv_path,
        [
            {"Antibody name (alias)": "Ab1", "Epitope": "GELDRWEKIRLRPGG"},  # lineal, valido
            {"Antibody name (alias)": "Ab2", "Epitope": ""},  # vacio, se omite
            {"Antibody name (alias)": "Ab3", "Epitope": "ELDRWEKI + ALDKIE"},  # compuesto, se omite
            {"Antibody name (alias)": "Ab4", "Epitope": "conformational epitope, discontinuous"},  # tiene espacios/coma
        ],
    )

    result = _load_bnab_epitopes(csv_path)

    assert list(result["antibody_name"]) == ["Ab1"]
    assert result.iloc[0]["epitope_sequence"] == "GELDRWEKIRLRPGG"


def test_load_bnab_epitopes_exige_epitopo_todo_en_mayusculas():
    # La regex de validacion (_AA_ONLY) es case-sensitive por diseno: los datos
    # reales de ab_all.csv siempre vienen en mayusculas, asi que una entrada en
    # minuscula (formato inesperado) se descarta como cualquier otro epitopo
    # que no matchea el patron 'solo residuos AA', no se normaliza a la fuerza.
    from src.engines.lanl_catnap_engine import _AA_ONLY

    assert _AA_ONLY.fullmatch("GELDRWEKIRLRPGG")
    assert not _AA_ONLY.fullmatch("geldrwekirlrpgg")


def test_load_bnab_epitopes_columnas_faltantes_lanza_value_error(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w") as fh:
        fh.write("col1,col2\nval1,val2\n")

    with pytest.raises(ValueError, match="faltan"):
        _load_bnab_epitopes(csv_path)


# --- _load_catnap_potency ------------------------------------------------------------------


def test_load_catnap_potency_normaliza_nombre_y_parsea_numeros(tmp_path):
    abs_path = tmp_path / "abs.txt"
    _write_catnap_abs(abs_path, [{"Name": "10E8", "Mean panel IC50": "0.506", "# of viruses tested": "1321"}])

    result = _load_catnap_potency(abs_path)

    row = result[result["antibody_name_norm"] == "10E8"].iloc[0]
    assert row["catnap_mean_ic50"] == pytest.approx(0.506)
    assert row["catnap_n_viruses"] == 1321


def test_load_catnap_potency_valores_no_numericos_quedan_nan(tmp_path):
    abs_path = tmp_path / "abs.txt"
    _write_catnap_abs(abs_path, [{"Name": "AbX", "Mean panel IC50": ">150", "# of viruses tested": "5"}])

    result = _load_catnap_potency(abs_path)

    assert pd.isna(result.iloc[0]["catnap_mean_ic50"])


# --- query_bnab_crossref: integracion de punta a punta -----------------------------------


def test_query_vacio_no_carga_nada(tmp_path):
    result = query_bnab_crossref([], tmp_path / "no_existe.csv")
    assert result.empty


def test_match_exacto_reporta_longitud_completa(tmp_path):
    csv_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(csv_path, [{"Antibody name (alias)": "10E8", "Epitope": "NWFDISNWLWYIK", "Neutralizing": "yes"}])

    result = query_bnab_crossref(["NWFDISNWLWYIK"], csv_path, min_overlap=6)

    assert len(result) == 1
    assert result.iloc[0]["antibody_name"] == "10E8"
    assert result.iloc[0]["match_length"] == 13


def test_solapamiento_parcial_por_encima_del_umbral_matchea(tmp_path):
    csv_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(csv_path, [{"Antibody name (alias)": "Ab1", "Epitope": "GELDRWEKIRLRPGG"}])

    result = query_bnab_crossref(["XXXWEKIRLRPGGXXX"], csv_path, min_overlap=6)

    assert len(result) == 1
    assert result.iloc[0]["match_length"] == 10


def test_solapamiento_por_debajo_del_umbral_no_matchea(tmp_path):
    csv_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(csv_path, [{"Antibody name (alias)": "Ab1", "Epitope": "GELDRWEKIRLRPGG"}])

    # Comparte 'RWE' (3 residuos) con el epitopo, muy por debajo de min_overlap=6.
    result = query_bnab_crossref(["ZZZRWEZZZ"], csv_path, min_overlap=6)

    assert result.empty


def test_epitopo_de_referencia_mas_corto_que_el_umbral_exige_match_completo(tmp_path):
    csv_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(csv_path, [{"Antibody name (alias)": "Ab1", "Epitope": "EKIRLR"}])  # 6 aa, == min_overlap

    # Contiene el epitopo completo de 6 aa -> matchea aunque min_overlap=10 (mas laxo que el propio epitopo).
    result = query_bnab_crossref(["XXXEKIRLRXXX"], csv_path, min_overlap=10)

    assert len(result) == 1
    assert result.iloc[0]["match_length"] == 6


def test_ruido_sin_relacion_no_matchea(tmp_path):
    csv_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(csv_path, [{"Antibody name (alias)": "Ab1", "Epitope": "GELDRWEKIRLRPGG"}])

    result = query_bnab_crossref(["AAAAAAAAAAAAAAAA"], csv_path, min_overlap=6)

    assert result.empty


def test_potencia_catnap_se_anexa_cuando_el_nombre_coincide(tmp_path):
    ab_all_path = tmp_path / "ab_all.csv"
    abs_path = tmp_path / "abs.txt"
    _write_lanl_csv(ab_all_path, [{"Antibody name (alias)": "10E8", "Epitope": "NWFDISNWLWYIK", "Neutralizing": "yes"}])
    _write_catnap_abs(abs_path, [{"Name": "10E8", "Mean panel IC50": "0.506", "# of viruses tested": "1321"}])

    result = query_bnab_crossref(["NWFDISNWLWYIK"], ab_all_path, catnap_abs_path=abs_path, min_overlap=6)

    assert result.iloc[0]["catnap_mean_ic50"] == pytest.approx(0.506)
    assert result.iloc[0]["catnap_n_viruses"] == 1321


def test_sin_catnap_abs_path_columnas_quedan_na(tmp_path):
    ab_all_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(ab_all_path, [{"Antibody name (alias)": "10E8", "Epitope": "NWFDISNWLWYIK"}])

    result = query_bnab_crossref(["NWFDISNWLWYIK"], ab_all_path, catnap_abs_path=None, min_overlap=6)

    assert pd.isna(result.iloc[0]["catnap_mean_ic50"])


def test_multiples_candidatos_multiples_referencias(tmp_path):
    csv_path = tmp_path / "ab_all.csv"
    _write_lanl_csv(
        csv_path,
        [
            {"Antibody name (alias)": "Ab1", "Epitope": "GELDRWEKIRLRPGG"},
            {"Antibody name (alias)": "Ab2", "Epitope": "NWFDISNWLWYIK"},
        ],
    )

    result = query_bnab_crossref(["GELDRWEKIRLRPGG", "AAAAAAAAAAAAAAAA", "NWFDISNWLWYIK"], csv_path, min_overlap=6)

    assert set(result["sequence"]) == {"GELDRWEKIRLRPGG", "NWFDISNWLWYIK"}
    assert len(result) == 2
