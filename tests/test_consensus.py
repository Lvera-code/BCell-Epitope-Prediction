"""Tests de la Fase 3: union logica anotada de N motores (src/engines/consensus.py).

Logica 100% pura (pandas), sin subprocess ni binarios externos. Cubre los 3
escenarios de motores contribuyentes que soporta el pipeline: solo motores de
secuencia (Camino 1: bepipred+epidope), solo motores estructurales (Camino 2:
discotope+scannet), y los 4 juntos (Camino 3).
"""

import pandas as pd
import pytest

from src.engines.consensus import (
    MIN_FINAL_PEPTIDE_LENGTH,
    _origen_label,
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


def test_engine_dfs_vacio_lanza_value_error():
    with pytest.raises(ValueError):
        build_annotated_union_table({}, {})


# --- Escenario 1: solo motores de secuencia (Camino 1: bepipred + epidope) ---

def test_camino1_region_solo_bepipred_queda_etiquetada_como_tal():
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("ACC1", 31, 45)]),  # 15 aa >= min_length
        "epidope": _epitope_df([]),
    }
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    assert len(result) == 1
    assert result.loc[0, "origen"] == "Bp"
    assert result.loc[0, "bepipred_score"] == 0.5
    assert pd.isna(result.loc[0, "epidope_score"])
    assert result.loc[0, "sequence"] == FULL_SEQ[30:45]


def test_camino1_regiones_solapadas_se_fusionan_y_listan_ambos_motores():
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("ACC1", 31, 45, mean_score=0.6)]),
        "epidope": _epitope_df([_epitope_row("ACC1", 40, 56, mean_score=0.9)]),
    }
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["start"] == 31 and row["end"] == 56  # min start, max end (fusion, no interseccion)
    assert row["origen"] == "Bp+Ed"
    assert row["bepipred_score"] == 0.6
    assert row["epidope_score"] == 0.9
    assert row["sequence"] == FULL_SEQ[30:56]


def test_camino1_regiones_no_solapadas_quedan_separadas():
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("ACC1", 31, 45)]),
        "epidope": _epitope_df([_epitope_row("ACC1", 47, 56)]),  # gap de 1 residuo, no solapa
    }
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    assert len(result) == 2
    assert list(result["origen"]) == ["Bp", "Ed"]


def test_camino1_filtro_de_longitud_minima_descarta_regiones_cortas():
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("ACC1", 31, 35)]),  # 5 aa < 9
        "epidope": _epitope_df([]),
    }
    lookup = {"ACC1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup, min_length=MIN_FINAL_PEPTIDE_LENGTH)

    assert result.empty


# --- Escenario 2: solo motores estructurales (Camino 2: discotope + scannet) ---

def test_camino2_solo_estructurales_no_asume_motor_de_secuencia():
    engine_dfs = {
        "discotope": _epitope_df([_epitope_row("PDB1", 31, 45, mean_score=0.92)]),
        "scannet": _epitope_df([_epitope_row("PDB1", 40, 56, mean_score=0.75)]),
    }
    lookup = {"PDB1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["origen"] == "Dt+Sn"
    assert row["discotope_score"] == 0.92
    assert row["scannet_score"] == 0.75
    assert "bepipred_score" not in result.columns
    assert "epidope_score" not in result.columns


def test_camino2_region_solo_discotope():
    engine_dfs = {
        "discotope": _epitope_df([_epitope_row("PDB1", 31, 45, mean_score=0.95)]),
        "scannet": _epitope_df([]),
    }
    lookup = {"PDB1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    assert len(result) == 1
    assert result.loc[0, "origen"] == "Dt"
    assert pd.isna(result.loc[0, "scannet_score"])


def test_camino2_verifica_limites_con_position_mapping_sin_lanzar(caplog):
    # position_mapping se usa solo como verificacion (ver ADR del modulo):
    # con coordenadas dentro de rango, no debe alterar el resultado ni fallar.
    engine_dfs = {
        "discotope": _epitope_df([_epitope_row("PDB1", 31, 45, mean_score=0.92)]),
        "scannet": _epitope_df([]),
    }
    lookup = {"PDB1": FULL_SEQ}
    position_mapping = pd.DataFrame({
        "accession": ["PDB1"] * len(FULL_SEQ),
        "fasta_position": range(1, len(FULL_SEQ) + 1),
    })

    result = build_annotated_union_table(engine_dfs, lookup, position_mapping=position_mapping)

    assert len(result) == 1
    assert not any("mas alla de la longitud" in msg for msg in caplog.messages)


def test_camino2_coordenadas_fuera_de_rango_loguea_warning(caplog):
    engine_dfs = {
        "discotope": _epitope_df([_epitope_row("PDB1", 1, 9, mean_score=0.92)]),
        "scannet": _epitope_df([]),
    }
    short_seq = "M" * 5  # mas corta que el 'end'=9 reportado por discotope
    lookup = {"PDB1": short_seq}
    position_mapping = pd.DataFrame({"accession": ["PDB1"] * 5, "fasta_position": range(1, 6)})

    build_annotated_union_table(engine_dfs, lookup, position_mapping=position_mapping)

    assert any("mas alla de la longitud" in msg for msg in caplog.messages)


# --- Escenario 3: los 4 motores juntos (Camino 3) ---

def test_camino3_los_4_motores_contribuyen_a_una_region_fusionada():
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("PDB1", 31, 40, mean_score=0.2)]),
        "epidope": _epitope_df([_epitope_row("PDB1", 38, 46, mean_score=0.85)]),
        "discotope": _epitope_df([_epitope_row("PDB1", 44, 50, mean_score=0.93)]),
        "scannet": _epitope_df([_epitope_row("PDB1", 48, 56, mean_score=0.6)]),
    }
    lookup = {"PDB1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["start"] == 31 and row["end"] == 56  # fusion transitiva de los 4 intervalos encadenados
    assert row["origen"] == "Consenso total"
    for key in ("bepipred", "epidope", "discotope", "scannet"):
        assert result.loc[0, f"{key}_score"] == engine_dfs[key].iloc[0]["mean_score"]


def test_camino3_motor_ausente_para_una_accession_no_rompe_las_demas():
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("ACC1", 31, 45), _epitope_row("ACC2", 31, 45)]),
        "epidope": _epitope_df([_epitope_row("ACC1", 31, 45)]),  # no corrio (o no detecto nada) para ACC2
        "discotope": _epitope_df([]),
        "scannet": _epitope_df([]),
    }
    lookup = {"ACC1": FULL_SEQ, "ACC2": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    acc1_row = result[result["accession"] == "ACC1"].iloc[0]
    acc2_row = result[result["accession"] == "ACC2"].iloc[0]
    assert acc1_row["origen"] == "Bp+Ed"
    assert acc2_row["origen"] == "Bp"


def test_columnas_de_salida_incluyen_score_y_region_por_cada_motor_de_engine_dfs():
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("ACC1", 31, 45)]),
        "discotope": _epitope_df([]),
    }
    result = build_annotated_union_table(engine_dfs, {"ACC1": FULL_SEQ})

    for key in ("bepipred", "discotope"):
        assert f"{key}_score" in result.columns
        assert f"{key}_region" in result.columns


# --- Etiquetas de 'origen': abreviaturas de 2 letras + 'Consenso total' ---------------

def test_origen_label_motor_unico_es_solo_la_abreviatura():
    assert _origen_label(["bepipred"]) == "Bp"
    assert _origen_label(["epidope"]) == "Ed"
    assert _origen_label(["discotope"]) == "Dt"
    assert _origen_label(["scannet"]) == "Sn"


def test_origen_label_dos_motores_abreviaturas_unidas_por_mas():
    assert _origen_label(["bepipred", "epidope"]) == "Bp+Ed"
    assert _origen_label(["discotope", "scannet"]) == "Dt+Sn"
    # Cualquier combinacion, no solo los pares "naturales" -- esto es
    # justamente lo que permite distinguirlas sin ambiguedad.
    assert _origen_label(["bepipred", "discotope"]) == "Bp+Dt"
    assert _origen_label(["epidope", "scannet"]) == "Ed+Sn"
    assert _origen_label(["bepipred", "scannet"]) == "Bp+Sn"
    assert _origen_label(["epidope", "discotope"]) == "Ed+Dt"


def test_origen_label_orden_sigue_el_de_contributing_keys():
    assert _origen_label(["epidope", "bepipred"]) == "Ed+Bp"
    assert _origen_label(["scannet", "discotope"]) == "Sn+Dt"


def test_origen_label_tres_motores_abreviaturas_unidas_por_mas():
    assert _origen_label(["bepipred", "epidope", "discotope"]) == "Bp+Ed+Dt"
    assert _origen_label(["bepipred", "discotope", "scannet"]) == "Bp+Dt+Sn"


def test_origen_label_los_4_motores_es_consenso_total():
    assert _origen_label(["bepipred", "epidope", "discotope", "scannet"]) == "Consenso total"
    # El orden de las claves no cambia el resultado: siempre exactamente los 4.
    assert _origen_label(["scannet", "bepipred", "discotope", "epidope"]) == "Consenso total"


def test_camino3_combinacion_parcial_de_3_motores_no_es_consenso_total(caplog):
    # Fusion end-to-end donde contribuyen 3 de los 4 motores (falta scannet):
    # NO debe etiquetarse 'Consenso total' (reservado a los 4 exactos).
    engine_dfs = {
        "bepipred": _epitope_df([_epitope_row("PDB1", 31, 45, mean_score=0.2)]),
        "epidope": _epitope_df([_epitope_row("PDB1", 40, 56, mean_score=0.85)]),
        "discotope": _epitope_df([_epitope_row("PDB1", 44, 56, mean_score=0.93)]),
        "scannet": _epitope_df([]),
    }
    lookup = {"PDB1": FULL_SEQ}

    result = build_annotated_union_table(engine_dfs, lookup)

    assert len(result) == 1
    assert result.loc[0, "origen"] == "Bp+Ed+Dt"
