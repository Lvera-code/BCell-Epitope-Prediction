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

from src.engines.conservation_engine import (
    _count_panel_sequences,
    _panel_breadth_by_query,
    _panel_cache_dir,
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
