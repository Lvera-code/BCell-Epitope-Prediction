"""Tests de la Fase 7 (src/engines/construct_assembly.py): seleccion top-N por clase,
deduplicacion por nucleo de union, y ensamblaje con linkers.

Logica 100% pura (sin subprocess), se prueba de punta a punta con DataFrames sinteticos
que replican el formato real de safe_df/algpred_df/stackgly_df/htl_df/ctl_df.
"""

import pandas as pd
import pytest

from src.engines.construct_assembly import assemble_construct
from src.config.settings import Settings


def _safe_df(rows):
    """rows: lista de dicts con al menos accession/start/end/sequence + columnas '{motor}_score'."""
    return pd.DataFrame(rows)


def _algpred_df(rows):
    return pd.DataFrame(rows, columns=["sequence", "algpred_score", "algpred_veredicto"])


def _stackgly_df(rows):
    return pd.DataFrame(rows, columns=["sequence", "sequon_position", "stackglyembed_veredicto", "stackglyembed_score"])


def _htl_ctl_row(accession, sequence_f5, core_9aa, start, end, n_prom, min_rank, netcleave_match=None, netcleave_score=None):
    row = {
        "accession": accession, "sequence_f5": sequence_f5, "core_9aa": core_9aa,
        "start": start, "end": end, "origen": "Ed",
        "n_alelos_promiscuos": n_prom, "n_alelos_evaluados": 27, "min_rank_el": min_rank,
    }
    if netcleave_match is not None:
        row["netcleave_c_term_match"] = netcleave_match
        row["netcleave_c_term_score"] = netcleave_score
    return row


# --- Caso vacio -----------------------------------------------------------------------


def test_todo_vacio_no_ensambla_nada():
    empty = pd.DataFrame()
    seq, meta = assemble_construct(empty, empty, empty, empty, empty)
    assert seq == ""
    assert meta.empty


# --- Seleccion B-cell: filtro Non-Allergen + sin glyco riesgoso ------------------------


def test_bcell_excluye_allergen():
    safe = _safe_df([
        {"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9},
        {"accession": "A", "start": 20, "end": 29, "sequence": "BBBBBBBBBB", "bepipred_score": 0.8},
    ])
    algpred = _algpred_df([
        ["AAAAAAAAAA", 0.9, "Allergen"],
        ["BBBBBBBBBB", 0.2, "Non-Allergen"],
    ])
    stackgly = _stackgly_df([])

    seq, meta = assemble_construct(safe, algpred, stackgly, pd.DataFrame(), pd.DataFrame())

    assert seq == "BBBBBBBBBB"
    assert list(meta["block"]) == ["B-cell"]


def test_bcell_excluye_con_sequon_glicosilado():
    safe = _safe_df([
        {"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9},
        {"accession": "A", "start": 20, "end": 29, "sequence": "BBBBBBBBBB", "bepipred_score": 0.8},
    ])
    algpred = _algpred_df([
        ["AAAAAAAAAA", 0.1, "Non-Allergen"],
        ["BBBBBBBBBB", 0.1, "Non-Allergen"],
    ])
    stackgly = _stackgly_df([["AAAAAAAAAA", 3, "Glicosilado", 0.8]])

    seq, meta = assemble_construct(safe, algpred, stackgly, pd.DataFrame(), pd.DataFrame())

    assert seq == "BBBBBBBBBB"


def test_bcell_sin_sequon_en_absoluto_no_se_excluye():
    # Un peptido que NUNCA aparece en stackgly_df (0 sequones) no debe tratarse
    # como riesgoso -- Fase 4c solo produce filas para sequones reales.
    safe = _safe_df([{"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9}])
    algpred = _algpred_df([["AAAAAAAAAA", 0.1, "Non-Allergen"]])
    stackgly = _stackgly_df([])  # vacio: ningun peptido tenia sequon

    seq, meta = assemble_construct(safe, algpred, stackgly, pd.DataFrame(), pd.DataFrame())

    assert seq == "AAAAAAAAAA"


def test_bcell_rankea_por_mejor_score_disponible_y_respeta_top_n():
    rows = []
    algpred_rows = []
    for i, score in enumerate([0.9, 0.5, 0.7, 0.3]):
        seq = f"{'X' * i}SEQUENCE{i}"
        rows.append({"accession": "A", "start": i * 10, "end": i * 10 + 9, "sequence": seq, "bepipred_score": score})
        algpred_rows.append([seq, 0.1, "Non-Allergen"])
    safe = _safe_df(rows)
    algpred = _algpred_df(algpred_rows)

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(), top_n_per_class=2)

    bcell_rows = meta[meta["block"] == "B-cell"]
    assert len(bcell_rows) == 2
    # Los 2 de mayor bepipred_score (0.9 y 0.7) deben ser los elegidos, en ese orden.
    assert list(bcell_rows["sequence"]) == ["SEQUENCE0", "XXSEQUENCE2"]


# --- Seleccion HTL/CTL: dedup por core_9aa + top-N -------------------------------------


def test_htl_dedup_por_core_se_queda_con_mejor_fila():
    htl = pd.DataFrame([
        _htl_ctl_row("A", "WINDOW1XXXXXXX", "CORE9AAXX", 100, 114, 3, 2.0),
        _htl_ctl_row("A", "WINDOW2XXXXXXX", "CORE9AAXX", 101, 115, 5, 0.5),  # mismo core, mejor promiscuidad
    ])

    seq, meta = assemble_construct(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), htl, pd.DataFrame())

    htl_rows = meta[meta["block"] == "HTL"]
    assert len(htl_rows) == 1
    assert htl_rows.iloc[0]["sequence"] == "CORE9AAXX"
    assert "n_alelos_promiscuos=5" in htl_rows.iloc[0]["source_score_note"]


def test_ctl_prioriza_netcleave_match_sobre_promiscuidad():
    ctl = pd.DataFrame([
        _htl_ctl_row("A", "W1", "COREAAAAA", 1, 9, 10, 0.1, netcleave_match=False, netcleave_score=None),
        _htl_ctl_row("A", "W2", "COREBBBBB", 1, 9, 3, 2.0, netcleave_match=True, netcleave_score=0.5),
    ])

    seq, meta = assemble_construct(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ctl, top_n_per_class=1
    )

    ctl_rows = meta[meta["block"] == "CTL"]
    assert len(ctl_rows) == 1
    # COREBBBBB tiene peor promiscuidad/%Rank pero SI tiene corte confirmado -> gana.
    assert ctl_rows.iloc[0]["sequence"] == "COREBBBBB"


# --- Linkers y orden de bloques ---------------------------------------------------------


def test_linkers_intra_e_inter_bloque_correctos():
    safe = _safe_df([
        {"accession": "A", "start": 1, "end": 3, "sequence": "BCL1", "bepipred_score": 0.9},
        {"accession": "A", "start": 10, "end": 13, "sequence": "BCL2", "bepipred_score": 0.8},
    ])
    algpred = _algpred_df([["BCL1", 0.1, "Non-Allergen"], ["BCL2", 0.1, "Non-Allergen"]])
    htl = pd.DataFrame([_htl_ctl_row("A", "W1", "HTLCORE", 20, 28, 5, 0.5)])
    ctl = pd.DataFrame([_htl_ctl_row("A", "W2", "CTLCORE", 30, 38, 4, 0.3, netcleave_match=True, netcleave_score=0.9)])

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), htl, ctl)

    expected = (
        "BCL1" + Settings.CONSTRUCT_LINKER_BCELL + "BCL2"
        + Settings.CONSTRUCT_LINKER_INTERBLOQUE
        + "HTLCORE"
        + Settings.CONSTRUCT_LINKER_INTERBLOQUE
        + "CTLCORE"
    )
    assert seq == expected


def test_clase_vacia_se_omite_sin_linker_colgante():
    # Sin HTL: B-cell debe unirse directo a CTL con un unico linker inter-bloque,
    # no dos (uno "hacia" HTL vacio y otro "desde" HTL vacio).
    safe = _safe_df([{"accession": "A", "start": 1, "end": 3, "sequence": "BCL1", "bepipred_score": 0.9}])
    algpred = _algpred_df([["BCL1", 0.1, "Non-Allergen"]])
    ctl = pd.DataFrame([_htl_ctl_row("A", "W2", "CTLCORE", 30, 38, 4, 0.3, netcleave_match=True, netcleave_score=0.9)])

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), pd.DataFrame(), ctl)

    assert seq == "BCL1" + Settings.CONSTRUCT_LINKER_INTERBLOQUE + "CTLCORE"
    assert "HTL" not in set(meta["block"])


def test_solo_htl_y_ctl_sin_bcell():
    # Caso real observado con un PDB (7c4s): 0 candidatos B-cell sobreviven
    # pero HTL/CTL si. El constructo debe arrancar directo con HTL, sin
    # ningun linker/hueco donde "deberia" ir el bloque B-cell.
    htl = pd.DataFrame([_htl_ctl_row("A", "W1", "HTLCORE", 20, 28, 5, 0.5)])
    ctl = pd.DataFrame([_htl_ctl_row("A", "W2", "CTLCORE", 30, 38, 4, 0.3, netcleave_match=True, netcleave_score=0.9)])

    seq, meta = assemble_construct(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), htl, ctl)

    assert seq == "HTLCORE" + Settings.CONSTRUCT_LINKER_INTERBLOQUE + "CTLCORE"
    assert meta.iloc[0]["block"] == "HTL"
    assert "B-cell" not in set(meta["block"])


def test_solo_una_clase_con_un_unico_candidato_sin_linker_intra():
    # Un solo B-cell candidato, sin HTL/CTL: no debe insertar ningun linker
    # intra-bloque (no hay "siguiente" candidato con quien unirse).
    safe = _safe_df([{"accession": "A", "start": 1, "end": 3, "sequence": "UNICO", "bepipred_score": 0.9}])
    algpred = _algpred_df([["UNICO", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame())

    assert seq == "UNICO"
    assert len(meta) == 1


def test_adjuvante_opcional_antepuesto_con_linker_rigido():
    safe = _safe_df([{"accession": "A", "start": 1, "end": 3, "sequence": "BCL1", "bepipred_score": 0.9}])
    algpred = _algpred_df([["BCL1", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(), adjuvant_sequence="ADJUVANT"
    )

    assert seq == "ADJUVANT" + Settings.CONSTRUCT_LINKER_ADJUVANTE + "BCL1"
    assert meta.iloc[0]["block"] == "Adjuvante"


# --- Invariante de trazabilidad -----------------------------------------------------------


def test_metadata_reconstruye_la_secuencia_exacta_siempre():
    safe = _safe_df([
        {"accession": "A", "start": 1, "end": 3, "sequence": "BCL1", "bepipred_score": 0.9},
        {"accession": "A", "start": 10, "end": 13, "sequence": "BCL2", "bepipred_score": 0.8},
    ])
    algpred = _algpred_df([["BCL1", 0.1, "Non-Allergen"], ["BCL2", 0.1, "Non-Allergen"]])
    htl = pd.DataFrame([
        _htl_ctl_row("A", "W1", "HTLCORE1", 20, 28, 5, 0.5),
        _htl_ctl_row("A", "W2", "HTLCORE2", 40, 48, 4, 0.6),
    ])
    ctl = pd.DataFrame([_htl_ctl_row("A", "W3", "CTLCORE", 30, 38, 4, 0.3, netcleave_match=True, netcleave_score=0.9)])

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), htl, ctl)

    assert "".join(meta["sequence"]) == seq
    # start/end de cada segmento deben ser contiguos y consistentes con su longitud.
    for row in meta.itertuples(index=False):
        assert row.end - row.start + 1 == len(row.sequence)
    for i in range(1, len(meta)):
        assert meta.iloc[i]["start"] == meta.iloc[i - 1]["end"] + 1
