"""Tests de la Fase 4: seleccion dinamica de task/E-value y filtro de cobertura de
BLASTp (src/engines/blast_engine.py).

Cubre las funciones puras: enrutamiento por longitud (``_select_task``,
``_select_evalue``) y el calculo de identidad maxima con filtro de cobertura
(``_max_identity_by_query``). ``run_blastp_filter`` en si depende del binario
real 'blastp' y de una base de datos indexada, fuera del alcance de un test
unitario (ver README.md - Seccion de tests, para la justificacion).
``filter_self_tolerant`` SI se testea (logica de re-mapeo de columnas/orden es
pura), mockeando ``run_blastp_filter`` via monkeypatch para no depender de
BLASTp real.
"""

import pandas as pd

import src.engines.blast_engine as blast_engine_module
from src.config.settings import Settings
from src.engines.blast_engine import _max_identity_by_query, _select_evalue, _select_task, filter_self_tolerant


def test_select_task_peptido_corto_usa_blastp_short():
    assert _select_task(9) == "blastp-short"
    assert _select_task(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN) == "blastp-short"  # limite inclusive


def test_select_task_peptido_largo_usa_blastp():
    assert _select_task(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN + 1) == "blastp"
    assert _select_task(200) == "blastp"


def test_select_evalue_tramo_corto():
    assert _select_evalue(9) == Settings.BLAST_EVALUE_SHORT
    assert _select_evalue(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN) == Settings.BLAST_EVALUE_SHORT


def test_select_evalue_tramo_medio():
    assert _select_evalue(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN + 1) == Settings.BLAST_EVALUE_MEDIUM
    assert _select_evalue(Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN) == Settings.BLAST_EVALUE_MEDIUM  # limite inclusive


def test_select_evalue_tramo_largo():
    assert _select_evalue(Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN + 1) == Settings.BLAST_EVALUE_LONG
    assert _select_evalue(500) == Settings.BLAST_EVALUE_LONG


# --- _max_identity_by_query: filtro de cobertura de consulta -------------------------

def _hit(qidx, pident, length):
    return {
        "qseqid": f"peptide_{qidx}", "sseqid": "sp|X|FAKE_HUMAN", "pident": pident,
        "length": length, "mismatch": 0, "gapopen": 0, "qstart": 1, "qend": length,
        "sstart": 1, "send": length, "evalue": 1.0, "bitscore": 10.0,
    }


def test_hit_vacio_devuelve_serie_vacia():
    result = _max_identity_by_query(pd.DataFrame(), pd.Series(dtype=int), min_query_coverage=0.9)
    assert result.empty


def test_fragmento_minusculo_100pct_identico_no_cuenta_si_cobertura_insuficiente():
    # Regresion: un hit de 5 aa 100% identico dentro de un peptido de 14 aa
    # (cobertura ~36%) es estadisticamente esperable por azar contra un
    # proteoma completo, NO una homologia real -- no debe contar hacia
    # max_pident.
    hits = pd.DataFrame([_hit(0, pident=100.0, length=5)])
    query_lengths = pd.Series({0: 14})

    result = _max_identity_by_query(hits, query_lengths, min_query_coverage=0.9)

    assert result.empty  # el unico hit no alcanza el 90% de cobertura (5/14 ~ 0.36)


def test_hit_de_longitud_completa_si_cuenta():
    hits = pd.DataFrame([_hit(0, pident=100.0, length=14)])
    query_lengths = pd.Series({0: 14})

    result = _max_identity_by_query(hits, query_lengths, min_query_coverage=0.9)

    assert result["peptide_0"] == 100.0


def test_toma_el_maximo_solo_entre_hits_con_cobertura_suficiente():
    hits = pd.DataFrame([
        _hit(0, pident=100.0, length=5),   # cobertura insuficiente (5/20), se ignora
        _hit(0, pident=60.0, length=19),   # cobertura suficiente (19/20 = 0.95), cuenta
        _hit(0, pident=40.0, length=20),   # cobertura completa, cuenta, pero pident menor
    ])
    query_lengths = pd.Series({0: 20})

    result = _max_identity_by_query(hits, query_lengths, min_query_coverage=0.9)

    assert result["peptide_0"] == 60.0  # el de 100% se descarta por cobertura


def test_umbral_de_cobertura_es_configurable():
    hits = pd.DataFrame([_hit(0, pident=100.0, length=5)])
    query_lengths = pd.Series({0: 14})

    # Con un umbral mas laxo, el mismo hit corto si cuenta.
    result = _max_identity_by_query(hits, query_lengths, min_query_coverage=0.3)

    assert result["peptide_0"] == 100.0


def test_multiples_queries_independientes():
    hits = pd.DataFrame([
        _hit(0, pident=95.0, length=10),
        _hit(1, pident=100.0, length=3),  # cobertura insuficiente para query 1
    ])
    query_lengths = pd.Series({0: 10, 1: 15})

    result = _max_identity_by_query(hits, query_lengths, min_query_coverage=0.9)

    assert result["peptide_0"] == 95.0
    assert "peptide_1" not in result.index


# --- filter_self_tolerant: re-chequeo sobre la secuencia FINAL de largo real -----------


def _fake_run_blastp_filter(status_by_sequence, pident_by_sequence=None):
    """Mock de 'run_blastp_filter': asigna 'status'/'max_pident' segun los dicts dados.

    Incluye SIEMPRE 'blast_task'/'blast_evalue'/'max_pident' (ademas de
    'status'), igual que la funcion real -- un mock que solo devolviera
    'status' no reproduciria el contrato real y dejaria pasar sin detectar
    el bug de 'max_pident' obsoleto que corrigio 'filter_self_tolerant'
    (ver tests de esa funcion mas abajo).
    """
    pident_by_sequence = pident_by_sequence or {}

    def _fake(epitopes_df, db_path=None, identity_threshold=None):
        result = epitopes_df.copy()
        result["status"] = result["sequence"].map(status_by_sequence)
        result["blast_task"] = "blastp-short"
        result["blast_evalue"] = 50.0
        result["max_pident"] = result["sequence"].map(pident_by_sequence).fillna(0.0)
        return result
    return _fake


def test_filter_self_tolerant_descarta_solo_las_filas_marcadas_autoinmunidad(monkeypatch):
    monkeypatch.setattr(
        blast_engine_module, "run_blastp_filter",
        _fake_run_blastp_filter({"AAAKKKAAA": "Segura", "PELIGROSA": "Autoinmunidad"}),
    )
    df = pd.DataFrame({
        "accession": ["P1", "P1"],
        "sequence_f5": ["AAAKKKAAA", "PELIGROSA"],
        "n_alelos_promiscuos": [5, 8],
    })

    result = filter_self_tolerant(df, "sequence_f5")

    assert list(result["sequence_f5"]) == ["AAAKKKAAA"]
    assert list(result["n_alelos_promiscuos"]) == [5]  # resto de columnas preservado


def test_filter_self_tolerant_preserva_todas_las_columnas_si_nadie_se_descarta(monkeypatch):
    monkeypatch.setattr(
        blast_engine_module, "run_blastp_filter",
        _fake_run_blastp_filter({"AAAKKKAAA": "Segura", "BBBCCCBBB": "Segura"}),
    )
    df = pd.DataFrame({"sequence": ["AAAKKKAAA", "BBBCCCBBB"], "extra_col": [1, 2]})

    result = filter_self_tolerant(df, "sequence")

    assert list(result["sequence"]) == ["AAAKKKAAA", "BBBCCCBBB"]
    assert list(result["extra_col"]) == [1, 2]


def test_filter_self_tolerant_df_vacio_no_invoca_blast(monkeypatch):
    calls = []
    monkeypatch.setattr(
        blast_engine_module, "run_blastp_filter", lambda *a, **k: calls.append(1)
    )

    result = filter_self_tolerant(pd.DataFrame(columns=["sequence"]), "sequence")

    assert result.empty
    assert not calls


def test_filter_self_tolerant_sobreescribe_max_pident_obsoleto_de_la_region_padre(monkeypatch):
    """Regresion del bug real corregido (ver docstring de 'filter_self_tolerant').

    'df' ya trae 'max_pident'/'blast_task'/'blast_evalue' calculados sobre la
    region PADRE mas ancha (como ocurre de verdad en el pipeline: 'safe_df'
    hereda estas columnas de la Fase 4). El re-chequeo interno sobre la
    secuencia FINAL calcula un 'max_pident' distinto (57.9, simulando el
    caso real encontrado con BLAST contra el proteoma humano) -- ese valor
    fresco debe reemplazar al viejo (0.0) en el resultado, no coexistir con
    el ni descartarse.
    """
    monkeypatch.setattr(
        blast_engine_module, "run_blastp_filter",
        _fake_run_blastp_filter(
            status_by_sequence={"VENTANAFINAL": "Segura"},
            pident_by_sequence={"VENTANAFINAL": 57.9},
        ),
    )
    df = pd.DataFrame({
        "accession": ["P1"],
        "sequence": ["VENTANAFINAL"],
        # Valores OBSOLETOS de la region padre de Fase 4 (mas ancha, ya recortada
        # a 'sequence' antes de llegar aqui -- mismo escenario que 'safe_df' real).
        "blast_task": ["blastp"],
        "blast_evalue": [0.05],
        "max_pident": [0.0],
    })

    result = filter_self_tolerant(df, "sequence")

    assert result["max_pident"].iloc[0] == 57.9  # fresco, NO el 0.0 obsoleto de la region padre
    assert result["blast_task"].iloc[0] == "blastp-short"  # tambien refrescado
    assert result["blast_evalue"].iloc[0] == 50.0  # tambien refrescado


def test_filter_self_tolerant_anade_max_pident_si_df_no_lo_traia(monkeypatch):
    """Camino HTL/CTL (Fase 5/5b): 'df' no viene de un BLAST anterior, no trae estas
    columnas todavia -- deben aparecer como nuevas (no hay valor obsoleto que
    sobreescribir, pero antes del fix se perdian igual porque solo se usaba 'status')."""
    monkeypatch.setattr(
        blast_engine_module, "run_blastp_filter",
        _fake_run_blastp_filter(
            status_by_sequence={"AAAKKKAAA": "Segura"},
            pident_by_sequence={"AAAKKKAAA": 12.3},
        ),
    )
    df = pd.DataFrame({"accession": ["P1"], "sequence_f5": ["AAAKKKAAA"], "n_alelos_promiscuos": [5]})

    result = filter_self_tolerant(df, "sequence_f5")

    assert result["max_pident"].iloc[0] == 12.3
    assert result["n_alelos_promiscuos"].iloc[0] == 5  # el resto de columnas originales se preserva
