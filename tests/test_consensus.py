"""Tests de la Fase 3: union logica anotada BepiPred U EpiDope (src/engines/consensus.py).

Logica 100% pura (pandas), sin subprocess ni binarios externos.
"""

import pandas as pd
import pytest

from src.engines.consensus import (
    MIN_FINAL_PEPTIDE_LENGTH,
    accession_id,
    build_annotated_union_table,
)


def _epitope_row(accession, start, end, mean_score=0.5):
    return {"accession": accession, "start": start, "end": end, "mean_score": mean_score}


def _epitope_df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["accession", "start", "end", "mean_score"])


FULL_SEQ = "M" * 30 + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "M" * 50  # posiciones 31-56 = A..Z


def test_accession_id_normaliza_al_primer_token():
    assert accession_id("P03377 env polyprotein [HIV]") == "P03377"
    assert accession_id("P03377") == "P03377"
    assert accession_id("") == ""


def test_region_solo_bepipred_queda_etiquetada_como_tal():
    bepipred_df = _epitope_df([_epitope_row("ACC1", 31, 45)])  # 15 aa >= min_length
    epidope_df = _epitope_df([])
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(bepipred_df, epidope_df, lookup)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["origen"] == "BepiPred"
    assert row["start"] == 31 and row["end"] == 45
    assert row["sequence"] == FULL_SEQ[30:45]
    assert pd.isna(row["epidope_score"])


def test_regiones_solapadas_se_fusionan_como_consenso():
    # BepiPred 31-45 y EpiDope 40-55 comparten residuos 40-45 -> deben fusionarse.
    bepipred_df = _epitope_df([_epitope_row("ACC1", 31, 45, mean_score=0.6)])
    epidope_df = _epitope_df([_epitope_row("ACC1", 40, 55, mean_score=0.9)])
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(bepipred_df, epidope_df, lookup)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["origen"] == "Consenso"
    assert row["start"] == 31 and row["end"] == 55  # start minimo, end maximo
    assert row["sequence"] == FULL_SEQ[30:55]
    assert row["bepipred_score"] == pytest.approx(0.6)
    assert row["epidope_score"] == pytest.approx(0.9)


def test_fusion_transitiva_encadena_tres_regiones():
    # A (31-40) solapa con B (38-48) solapa con C (46-56), pero A y C solas NO se solapan.
    bepipred_df = _epitope_df(
        [_epitope_row("ACC1", 31, 40), _epitope_row("ACC1", 46, 56)]
    )
    epidope_df = _epitope_df([_epitope_row("ACC1", 38, 48)])
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(bepipred_df, epidope_df, lookup)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["start"] == 31 and row["end"] == 56
    assert row["origen"] == "Consenso"


def test_regiones_no_solapadas_quedan_separadas():
    bepipred_df = _epitope_df([_epitope_row("ACC1", 31, 45)])
    epidope_df = _epitope_df([_epitope_row("ACC1", 60, 75)])
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(bepipred_df, epidope_df, lookup)

    assert len(result) == 2
    assert set(result["origen"]) == {"BepiPred", "EpiDope"}


def test_filtro_de_longitud_minima_descarta_regiones_cortas():
    short_len = MIN_FINAL_PEPTIDE_LENGTH - 1
    bepipred_df = _epitope_df([_epitope_row("ACC1", 31, 31 + short_len - 1)])
    epidope_df = _epitope_df([])
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(bepipred_df, epidope_df, lookup)

    assert result.empty


def test_accessions_con_cabecera_completa_se_normalizan_para_cruzar_motores():
    # BepiPred conserva la cabecera FASTA completa; EpiDope solo el primer token.
    bepipred_df = _epitope_df([_epitope_row("ACC1 descripcion larga", 31, 45)])
    epidope_df = _epitope_df([_epitope_row("ACC1", 40, 55)])
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(bepipred_df, epidope_df, lookup)

    assert len(result) == 1
    assert result.iloc[0]["origen"] == "Consenso"
    assert result.iloc[0]["accession"] == "ACC1"


def test_ambos_motores_vacios_devuelve_dataframe_vacio():
    result = build_annotated_union_table(_epitope_df([]), _epitope_df([]), {})
    assert result.empty
