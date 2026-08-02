"""Tests de la Fase 6c (src/engines/iedb_engine.py): cruce de candidatos B-cell contra
epitopos IEDB con proteccion/neutralizacion documentada.

Igual que test_lanl_catnap_engine.py (motor gemelo): pandas puro sobre un CSV local, se
prueba de punta a punta con un ``bcell_protective_epitopes.csv`` sintetico escrito en
``tmp_path``, replicando el formato ya filtrado (7 columnas planas, no el header de
2 niveles del bulk export crudo de IEDB -- ese filtrado es responsabilidad del paso de
SETUP, no de este motor).
"""

import pandas as pd
import pytest

from src.engines.iedb_engine import query_iedb_crossref

_IEDB_COLUMNS = [
    "epitope_sequence", "source_organism", "response_measured",
    "qualitative_measure", "method", "host", "pmid",
]


def _write_iedb_csv(path, rows):
    """Escribe un ``bcell_protective_epitopes.csv`` sintetico. ``rows`` es una lista de
    dicts con las columnas usadas por el motor; el resto de ``_IEDB_COLUMNS`` queda vacio."""
    df = pd.DataFrame([{col: row.get(col, "") for col in _IEDB_COLUMNS} for row in rows])
    df.to_csv(path, index=False)


# --- query_iedb_crossref: integracion de punta a punta -----------------------------------


def test_query_vacio_no_carga_nada(tmp_path):
    result = query_iedb_crossref([], tmp_path / "no_existe.csv")
    assert result.empty


def test_match_exacto_reporta_longitud_completa(tmp_path):
    csv_path = tmp_path / "bcell_protective_epitopes.csv"
    _write_iedb_csv(csv_path, [{
        "epitope_sequence": "NWFDISNWLWYIK", "source_organism": "Human immunodeficiency virus 1",
        "response_measured": "neutralization", "qualitative_measure": "Positive",
    }])

    result = query_iedb_crossref(["NWFDISNWLWYIK"], csv_path, min_overlap=6)

    assert len(result) == 1
    assert result.iloc[0]["source_organism"] == "Human immunodeficiency virus 1"
    assert result.iloc[0]["match_length"] == 13


def test_solapamiento_parcial_por_encima_del_umbral_matchea(tmp_path):
    csv_path = tmp_path / "bcell_protective_epitopes.csv"
    _write_iedb_csv(csv_path, [{"epitope_sequence": "GELDRWEKIRLRPGG"}])

    result = query_iedb_crossref(["XXXWEKIRLRPGGXXX"], csv_path, min_overlap=6)

    assert len(result) == 1
    assert result.iloc[0]["match_length"] == 10


def test_solapamiento_por_debajo_del_umbral_no_matchea(tmp_path):
    csv_path = tmp_path / "bcell_protective_epitopes.csv"
    _write_iedb_csv(csv_path, [{"epitope_sequence": "GELDRWEKIRLRPGG"}])

    # Comparte 'RWE' (3 residuos) con el epitopo, muy por debajo de min_overlap=6.
    result = query_iedb_crossref(["ZZZRWEZZZ"], csv_path, min_overlap=6)

    assert result.empty


def test_epitopo_de_referencia_mas_corto_que_el_umbral_exige_match_completo(tmp_path):
    csv_path = tmp_path / "bcell_protective_epitopes.csv"
    _write_iedb_csv(csv_path, [{"epitope_sequence": "EKIRLR"}])  # 6 aa

    # Contiene el epitopo completo de 6 aa -> matchea aunque min_overlap=10 (mas laxo que el propio epitopo).
    result = query_iedb_crossref(["XXXEKIRLRXXX"], csv_path, min_overlap=10)

    assert len(result) == 1
    assert result.iloc[0]["match_length"] == 6


def test_ruido_sin_relacion_no_matchea(tmp_path):
    csv_path = tmp_path / "bcell_protective_epitopes.csv"
    _write_iedb_csv(csv_path, [{"epitope_sequence": "GELDRWEKIRLRPGG"}])

    result = query_iedb_crossref(["AAAAAAAAAAAAAAAA"], csv_path, min_overlap=6)

    assert result.empty


def test_multiples_candidatos_multiples_referencias_multiples_organismos(tmp_path):
    # A diferencia de LANL/CATNAP (HIV especifico), este motor no filtra por
    # organismo: candidatos de patogenos distintos matchean sus propias
    # referencias en el mismo CSV sin interferencia cruzada.
    csv_path = tmp_path / "bcell_protective_epitopes.csv"
    _write_iedb_csv(csv_path, [
        {"epitope_sequence": "GELDRWEKIRLRPGG", "source_organism": "Influenza A virus"},
        {"epitope_sequence": "NWFDISNWLWYIK", "source_organism": "Plasmodium falciparum"},
    ])

    result = query_iedb_crossref(
        ["GELDRWEKIRLRPGG", "AAAAAAAAAAAAAAAA", "NWFDISNWLWYIK"], csv_path, min_overlap=6
    )

    assert set(result["sequence"]) == {"GELDRWEKIRLRPGG", "NWFDISNWLWYIK"}
    assert set(result["source_organism"]) == {"Influenza A virus", "Plasmodium falciparum"}
    assert len(result) == 2


def test_columnas_faltantes_lanza_value_error(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("col1,col2\nval1,val2\n")

    with pytest.raises(ValueError, match="faltan"):
        query_iedb_crossref(["AAAAAAAAA"], csv_path)


def test_epitopo_de_referencia_se_normaliza_a_mayusculas(tmp_path):
    csv_path = tmp_path / "bcell_protective_epitopes.csv"
    _write_iedb_csv(csv_path, [{"epitope_sequence": "geldrwekirlrpgg"}])  # minusculas en el CSV

    result = query_iedb_crossref(["GELDRWEKIRLRPGG"], csv_path, min_overlap=6)

    assert len(result) == 1
    assert result.iloc[0]["epitope_sequence"] == "GELDRWEKIRLRPGG"


def test_reference_db_real_carga_sin_error():
    # Guardarraíl de integracion: el CSV real distribuido con el repo (no un
    # fixture sintetico) debe seguir teniendo el esquema que este motor espera.
    from pathlib import Path
    real_path = Path(__file__).resolve().parent.parent / "reference_db" / "iedb" / "bcell_protective_epitopes.csv"

    result = query_iedb_crossref(["NANPNANPNANP"], real_path, min_overlap=6)  # motivo repetitivo real de P. falciparum

    assert not result.empty
    assert "Plasmodium falciparum" in set(result["source_organism"].dropna())
