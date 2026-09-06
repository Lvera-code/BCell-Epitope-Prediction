"""Tests de la Fase 6b (src/engines/conservation_engine.py): amplitud de conservacion.

Cubre las funciones puras: amplitud por query (``_panel_breadth_by_query``,
analoga a ``_max_identity_by_query`` de blast_engine pero contando secuencias
DISTINTAS del panel matcheadas, no solo el mejor hit), conteo de secuencias
del panel (``_count_panel_sequences``) y hash de cache por contenido
(``_panel_cache_dir``). ``ensure_panel_db``/``run_conservation_filter`` en si
dependen de los binarios reales 'makeblastdb'/'blastp', fuera del alcance de
un test unitario (mismo criterio que ``run_blastp_filter`` en
test_blast_engine.py).
"""

from pathlib import Path

import pandas as pd
import pytest

import numpy as np

import src.engines.conservation_engine as conservation_engine_module
from src.engines.conservation_engine import (
    _count_panel_sequences,
    _detect_terminal_trim,
    _panel_breadth_by_query,
    _panel_cache_dir,
    _query_position_coverage,
    run_conservation_filter,
)


# --- _panel_breadth_by_query: amplitud (secuencias distintas del panel) --------------


def _hit(qidx, sseqid, pident, length):
    return {
        "qseqid": f"peptide_{qidx}", "sseqid": sseqid, "pident": pident,
        "length": length, "mismatch": 0, "gapopen": 0, "qstart": 1, "qend": length,
        "sstart": 1, "send": length, "evalue": 1.0, "bitscore": 10.0,
    }


def test_hits_vacio_devuelve_serie_vacia():
    result = _panel_breadth_by_query(pd.DataFrame(), pd.Series(dtype=int), identity_threshold=90.0, min_query_coverage=0.9)
    assert result.empty


def test_cuenta_secuencias_distintas_no_solo_el_mejor_hit():
    # 3 cepas del panel matchean el mismo candidato (mismo qseqid), cada una
    # con un sseqid distinto -- a diferencia de max_pident, aca importa
    # CUANTAS cepas distintas matchean, no solo la mejor.
    hits = pd.DataFrame([
        _hit(0, "cepa_A", pident=95.0, length=20),
        _hit(0, "cepa_B", pident=92.0, length=20),
        _hit(0, "cepa_C", pident=100.0, length=20),
    ])
    query_lengths = pd.Series({0: 20})

    result = _panel_breadth_by_query(hits, query_lengths, identity_threshold=90.0, min_query_coverage=0.9)

    assert result["peptide_0"] == 3


def test_hits_bajo_el_umbral_de_identidad_no_cuentan():
    hits = pd.DataFrame([
        _hit(0, "cepa_A", pident=95.0, length=20),
        _hit(0, "cepa_B", pident=50.0, length=20),  # bajo el umbral, no cuenta
    ])
    query_lengths = pd.Series({0: 20})

    result = _panel_breadth_by_query(hits, query_lengths, identity_threshold=90.0, min_query_coverage=0.9)

    assert result["peptide_0"] == 1


def test_hits_con_cobertura_insuficiente_no_cuentan():
    # Mismo criterio anti-ruido que _max_identity_by_query: un fragmento
    # minusculo 100% identico no cuenta si no cubre lo suficiente del candidato.
    hits = pd.DataFrame([_hit(0, "cepa_A", pident=100.0, length=5)])
    query_lengths = pd.Series({0: 20})  # cobertura 5/20 = 0.25

    result = _panel_breadth_by_query(hits, query_lengths, identity_threshold=90.0, min_query_coverage=0.9)

    assert result.empty


def test_mismo_sseqid_repetido_cuenta_una_sola_vez():
    # Dos alineamientos distintos contra la MISMA secuencia del panel
    # (p. ej. dos regiones locales) no deben inflar la amplitud.
    hits = pd.DataFrame([
        _hit(0, "cepa_A", pident=95.0, length=20),
        _hit(0, "cepa_A", pident=98.0, length=20),
    ])
    query_lengths = pd.Series({0: 20})

    result = _panel_breadth_by_query(hits, query_lengths, identity_threshold=90.0, min_query_coverage=0.9)

    assert result["peptide_0"] == 1


def test_multiples_queries_independientes():
    hits = pd.DataFrame([
        _hit(0, "cepa_A", pident=95.0, length=10),
        _hit(0, "cepa_B", pident=91.0, length=10),
        _hit(1, "cepa_A", pident=100.0, length=3),  # cobertura insuficiente para query 1
    ])
    query_lengths = pd.Series({0: 10, 1: 15})

    result = _panel_breadth_by_query(hits, query_lengths, identity_threshold=90.0, min_query_coverage=0.9)

    assert result["peptide_0"] == 2
    assert "peptide_1" not in result.index


# --- _count_panel_sequences -----------------------------------------------------------


def test_count_panel_sequences(tmp_path):
    panel = tmp_path / "panel.fasta"
    panel.write_text(">cepa_A\nMKV\n>cepa_B\nMKL\n>cepa_C\nMKT\n")

    assert _count_panel_sequences(panel) == 3


def test_count_panel_sequences_vacio(tmp_path):
    panel = tmp_path / "vacio.fasta"
    panel.write_text("")

    assert _count_panel_sequences(panel) == 0


# --- _panel_cache_dir: hash por CONTENIDO, no por ruta --------------------------------


def test_mismo_contenido_mismo_cache_dir(tmp_path):
    panel_a = tmp_path / "a.fasta"
    panel_b = tmp_path / "b.fasta"
    panel_a.write_text(">cepa_A\nMKV\n")
    panel_b.write_text(">cepa_A\nMKV\n")  # mismo contenido, ruta distinta

    assert _panel_cache_dir(panel_a) == _panel_cache_dir(panel_b)


def test_contenido_distinto_cache_dir_distinto(tmp_path):
    panel_a = tmp_path / "a.fasta"
    panel_b = tmp_path / "b.fasta"
    panel_a.write_text(">cepa_A\nMKV\n")
    panel_b.write_text(">cepa_A\nMKW\n")  # un residuo distinto

    assert _panel_cache_dir(panel_a) != _panel_cache_dir(panel_b)


# --- _query_position_coverage / _detect_terminal_trim: acantilado de cobertura terminal --


def test_query_position_coverage_calcula_fraccion_por_posicion():
    # 2 hits: uno cubre todo (0-13), otro deja fuera los ultimos 2 residuos (0-11).
    hits = pd.DataFrame([_hit(0, "cepa_A", pident=95.0, length=14), _hit(0, "cepa_B", pident=95.0, length=14)])
    hits.loc[1, "qend"] = 12  # 1-indexado inclusive -> cubre 0-indexado 0-11

    coverage = _query_position_coverage(hits, length=14)

    assert coverage[:12] == pytest.approx(np.ones(12))
    assert coverage[12] == pytest.approx(0.5)
    assert coverage[13] == pytest.approx(0.5)


def test_detect_terminal_trim_detecta_cola_no_nativa():
    # Nucleo solido (12 residuos al 100%) + cola de 2 residuos casi sin cobertura,
    # exactamente el patron real verificado en 4XAW (gp41 MPER + cola de cristalizacion).
    coverage = np.array([1.0] * 12 + [0.01, 0.0])

    trim = _detect_terminal_trim(coverage)

    assert trim == (0, 11)


def test_detect_terminal_trim_ignora_caida_gradual_sin_acantilado():
    # Declive gradual (variabilidad normal de un panel de cepas real), sin
    # acantilado limpio: no debe dispararse.
    coverage = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.4, 0.3, 0.2])

    assert _detect_terminal_trim(coverage) is None


def test_detect_terminal_trim_ignora_un_unico_residuo_ruidoso():
    # Un solo residuo con cobertura baja en el extremo no basta (se exige un
    # tramo contiguo de al menos 2), para no sobre-disparar por ruido puntual.
    coverage = np.array([1.0] * 10 + [0.0])

    assert _detect_terminal_trim(coverage) is None


def test_detect_terminal_trim_ignora_nucleo_interior_mediocre():
    # Extremo con caida real, pero el "nucleo" tampoco tiene cobertura alta:
    # no es el patron de cola no nativa, es una señal debil en general.
    coverage = np.array([0.5] * 10 + [0.0, 0.0])

    assert _detect_terminal_trim(coverage) is None


def test_run_conservation_filter_detect_terminal_trim_es_opt_in(tmp_path, monkeypatch):
    # Mismo patron de cobertura que test_detect_terminal_trim_detecta_cola_no_nativa
    # (nucleo solido de 12 + cola de 2 residuos sin cobertura), pero end-to-end a
    # traves de run_conservation_filter: por defecto (detect_terminal_trim=False,
    # el valor que usan los bloques HTL/CTL en pipeline.py) NO debe anotar nada,
    # aunque el patron dispararia. Solo detect_terminal_trim=True (bloque B-cell)
    # debe poblar las columnas.
    seq = "WFDITNWLWYIKKK"  # 14 aa
    # 19 hits solo cubren el nucleo (0-11); 1 unico hit cubre la cola tambien
    # (ruido residual, igual que el patron real de 4XAW) -- cobertura en la
    # cola = 1/20 = 5%, por debajo de _TERMINAL_TRIM_LOW_COVERAGE.
    hits = pd.DataFrame([_hit(0, f"cepa_{i}", pident=95.0, length=12 if i < 19 else 14) for i in range(20)])
    hits.loc[hits.index < 19, "qend"] = 12
    hits.loc[hits.index >= 19, "qend"] = 14

    monkeypatch.setattr(conservation_engine_module, "ensure_panel_db", lambda path: (Path("fake_db"), 2994))
    monkeypatch.setattr(conservation_engine_module, "_run_blastp_batch", lambda *a, **k: hits.copy())

    candidates = pd.DataFrame({"sequence": [seq]})

    default_result = run_conservation_filter(candidates, "fake_panel.fasta")
    assert default_result["conservation_query_trim_start"].isna().all()
    assert default_result["conservation_pct_native_core"].isna().all()

    trimmed_result = run_conservation_filter(candidates, "fake_panel.fasta", detect_terminal_trim=True)
    assert trimmed_result.loc[0, "conservation_query_trim_start"] == 1
    assert trimmed_result.loc[0, "conservation_query_trim_end"] == 12
    assert trimmed_result.loc[0, "conservation_pct_native_core"] > trimmed_result.loc[0, "conservation_pct"]


def test_detect_terminal_trim_candidato_nativo_no_dispara():
    # Control negativo real del panel (candidato B-cell nativo de gp120):
    # cobertura uniformemente alta en todo el candidato.
    coverage = np.array([0.95] * 13)

    assert _detect_terminal_trim(coverage) is None


def test_detect_terminal_trim_ignora_region_hipervariable_larga():
    # Regresion real encontrada al verificar contra el panel LANL/CATNAP:
    # candidato nativo 312-324 de gp120 (3NGB), region hipervariable del
    # CD4bs. 5 residuos casi universales + 8 con cobertura uniformemente baja
    # (~5-6%, no ~0%): un tramo de baja cobertura MAS LARGO que
    # _TERMINAL_TRIM_MAX_RUN es variabilidad biologica real, no una cola
    # corta de cristalizacion/clonaje, y no debe recortarse.
    coverage = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05])

    assert _detect_terminal_trim(coverage) is None


def test_detect_terminal_trim_respeta_el_tope_superior_en_el_extremo_correcto():
    # Cola de baja cobertura de exactamente _TERMINAL_TRIM_MAX_RUN (5) SI
    # cuenta (limite inclusive), pero un residuo mas ya no.
    coverage_ok = np.array([1.0] * 10 + [0.0] * 5)
    assert _detect_terminal_trim(coverage_ok) == (0, 9)

    coverage_too_long = np.array([1.0] * 10 + [0.0] * 6)
    assert _detect_terminal_trim(coverage_too_long) is None
