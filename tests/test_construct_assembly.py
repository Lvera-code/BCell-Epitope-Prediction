"""Tests de la Fase 7 (src/engines/construct_assembly.py): seleccion top-N por clase,
deduplicacion por nucleo de union, y ensamblaje con linkers.

Logica CASI 100% pura (sin subprocess), se prueba de punta a punta con DataFrames
sinteticos que replican el formato real de safe_df/algpred_df/stackgly_df/htl_df/ctl_df.
2 excepciones, ambas mockeadas por defecto via monkeypatch (autouse):
``predict_allergenicity``/``predict_nglycosylation`` (invocadas por
``_pad_short_bcell_candidates``) y ``filter_self_tolerant`` (invocada al final de
``_select_bcell_candidates``, SI corre BLASTp real contra el proteoma humano en
produccion -- sin mockear, secuencias sinteticas cortas como poly-A pueden matchear
por azar y romper tests que no buscan probar ESTE mecanismo especificamente). Los
tests dedicados a cada uno de esos 2 mecanismos anulan el mock por defecto.
"""

import pandas as pd
import pytest

import src.engines.construct_assembly as construct_assembly_module
from src.engines.construct_assembly import assemble_construct
from src.config.settings import Settings


@pytest.fixture(autouse=True)
def _bypass_self_tolerance_reblast(monkeypatch):
    """Por defecto, 'filter_self_tolerant' es un pass-through (nadie se descarta).

    Evita que TODOS los tests que ejercitan `_select_bcell_candidates` con
    output_dir/input_stem reales terminen invocando BLASTp de verdad -- lento y no
    determinista (depende del proteoma humano real). Los tests que SI quieren
    probar el re-chequeo de autotolerancia (ver seccion dedicada mas abajo)
    sobreescriben este mock con uno que descarta secuencias especificas.
    """
    monkeypatch.setattr(
        construct_assembly_module, "filter_self_tolerant", lambda df, sequence_col, **kwargs: df
    )


def _safe_df(rows):
    """rows: lista de dicts con al menos accession/start/end/sequence + columnas '{motor}_score'."""
    return pd.DataFrame(rows)


def _algpred_df(rows):
    return pd.DataFrame(rows, columns=["sequence", "algpred_score", "algpred_veredicto"])


def _stackgly_df(rows):
    return pd.DataFrame(rows, columns=["sequence", "sequon_position", "stackglyembed_veredicto", "stackglyembed_score"])


def _conservation_df(rows):
    """rows: lista de dicts con 'sequence'/'conservation_pct' (mismo formato que Fase 6b)."""
    return pd.DataFrame(rows, columns=["sequence", "conservation_pct"])


def _iedb_df(sequences):
    """rows: lista de secuencias con >=1 match IEDB (mismo formato minimo que Fase 6c)."""
    return pd.DataFrame({"sequence": sequences, "source_organism": ["X"] * len(sequences)})


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


# --- Seleccion B-cell: ya no filtra por Allergen/glyco, solo anota ---------------------


def test_bcell_no_excluye_allergen_pero_lo_anota():
    # DECISION 2026-08-13: la validacion de publicacion encontro que AlgPred2
    # marcaba 'Allergen' 7 de 9 candidatos evaluados sin correlato clinico
    # real -- se dejo de excluir, mismo tratamiento que la glicosilacion.
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

    bcell_rows = meta[meta["block"] == "B-cell"]
    assert set(bcell_rows["sequence"]) == {"AAAAAAAAAA", "BBBBBBBBBB"}
    allergen_row = bcell_rows[bcell_rows["sequence"] == "AAAAAAAAAA"].iloc[0]
    non_allergen_row = bcell_rows[bcell_rows["sequence"] == "BBBBBBBBBB"].iloc[0]
    assert "allergen=True" in allergen_row["source_score_note"]
    assert "allergen=False" in non_allergen_row["source_score_note"]


def test_bcell_no_excluye_con_sequon_glicosilado_pero_lo_anota():
    # Existen anticuerpos descritos contra regiones glicosiladas (feedback de
    # Carmen Elena Gomez, ver vault) -- ya no se descarta, solo se anota.
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

    bcell_rows = meta[meta["block"] == "B-cell"]
    assert set(bcell_rows["sequence"]) == {"AAAAAAAAAA", "BBBBBBBBBB"}
    glyco_row = bcell_rows[bcell_rows["sequence"] == "AAAAAAAAAA"].iloc[0]
    non_glyco_row = bcell_rows[bcell_rows["sequence"] == "BBBBBBBBBB"].iloc[0]
    assert "glycosylated=True" in glyco_row["source_score_note"]
    assert "glycosylated=False" in non_glyco_row["source_score_note"]


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


# --- Conservacion (Fase 6b, OPCIONAL): anota, no filtra ni rankea ----------------------


def test_bcell_anota_conservacion_si_se_provee():
    safe = _safe_df([{"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9}])
    algpred = _algpred_df([["AAAAAAAAAA", 0.1, "Non-Allergen"]])
    conservation = _conservation_df([["AAAAAAAAAA", 66.67]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(), conservation_df=conservation
    )

    assert "conservation_pct=66.67" in meta.iloc[0]["source_score_note"]


def test_htl_ctl_anota_conservacion_por_sequence_f5():
    htl = pd.DataFrame([_htl_ctl_row("A", "W1", "HTLCORE", 20, 28, 5, 0.5)])
    ctl = pd.DataFrame([_htl_ctl_row("A", "W2", "CTLCORE", 30, 38, 4, 0.3, netcleave_match=True, netcleave_score=0.9)])
    conservation = _conservation_df([["W1", 100.0], ["W2", 33.33]])

    seq, meta = assemble_construct(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), htl, ctl, conservation_df=conservation
    )

    htl_row = meta[meta["block"] == "HTL"].iloc[0]
    ctl_row = meta[meta["block"] == "CTL"].iloc[0]
    assert "conservation_pct=100.0" in htl_row["source_score_note"]
    assert "conservation_pct=33.33" in ctl_row["source_score_note"]


def test_sin_panel_conservacion_no_agrega_columna_ni_nota():
    # conservation_df=None (default): comportamiento identico a antes de
    # este parametro, sin ningun rastro de 'conservation_pct' en la nota.
    safe = _safe_df([{"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9}])
    algpred = _algpred_df([["AAAAAAAAAA", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame())

    assert "conservation_pct" not in meta.iloc[0]["source_score_note"]


# --- Regiones documentadas (Fase 6c, IEDB): anota, no filtra ni rankea -- solo B-cell -----


def test_bcell_anota_documented_region_si_hay_match():
    safe = _safe_df([
        {"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9},
        {"accession": "A", "start": 20, "end": 29, "sequence": "BBBBBBBBBB", "bepipred_score": 0.8},
    ])
    algpred = _algpred_df([
        ["AAAAAAAAAA", 0.1, "Non-Allergen"],
        ["BBBBBBBBBB", 0.1, "Non-Allergen"],
    ])
    iedb = _iedb_df(["AAAAAAAAAA"])  # solo AAAAAAAAAA tiene match documentado

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(), iedb_df=iedb)

    bcell_rows = meta[meta["block"] == "B-cell"]
    matched_row = bcell_rows[bcell_rows["sequence"] == "AAAAAAAAAA"].iloc[0]
    unmatched_row = bcell_rows[bcell_rows["sequence"] == "BBBBBBBBBB"].iloc[0]
    assert "documented_region=True" in matched_row["source_score_note"]
    assert "documented_region=False" in unmatched_row["source_score_note"]


def test_sin_iedb_df_no_agrega_columna_ni_nota():
    # iedb_df=None (default): comportamiento identico a antes de este
    # parametro, sin ningun rastro de 'documented_region' en la nota.
    safe = _safe_df([{"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9}])
    algpred = _algpred_df([["AAAAAAAAAA", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame())

    assert "documented_region" not in meta.iloc[0]["source_score_note"]


def test_iedb_df_vacio_se_trata_igual_que_none():
    safe = _safe_df([{"accession": "A", "start": 1, "end": 10, "sequence": "AAAAAAAAAA", "bepipred_score": 0.9}])
    algpred = _algpred_df([["AAAAAAAAAA", 0.1, "Non-Allergen"]])
    iedb_vacio = pd.DataFrame(columns=["sequence", "source_organism"])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(), iedb_df=iedb_vacio
    )

    assert "documented_region" not in meta.iloc[0]["source_score_note"]


def test_documented_region_no_se_anota_en_htl_ctl():
    # IEDB (Fase 6c) es especificamente ensayos B-cell -- no aplica al
    # mecanismo de reconocimiento MHC de HTL/CTL (ver docstring de
    # iedb_engine), a diferencia de conservation_pct que si cubre las 3 clases.
    htl = pd.DataFrame([_htl_ctl_row("A", "W1", "HTLCORE", 20, 28, 5, 0.5)])
    iedb = _iedb_df(["W1"])  # aunque "matchee" por casualidad, no debe anotarse en HTL

    seq, meta = assemble_construct(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), htl, pd.DataFrame(), iedb_df=iedb
    )

    htl_row = meta[meta["block"] == "HTL"].iloc[0]
    assert "documented_region" not in htl_row["source_score_note"]


# --- Seleccion HTL/CTL: dedup por core_9aa + top-N -------------------------------------


def test_htl_dedup_por_core_se_queda_con_mejor_fila():
    htl = pd.DataFrame([
        _htl_ctl_row("A", "WINDOW1XXXXXXX", "CORE9AAXX", 100, 114, 3, 2.0),
        _htl_ctl_row("A", "WINDOW2XXXXXXX", "CORE9AAXX", 101, 115, 5, 0.5),  # mismo core, mejor promiscuidad
    ])

    seq, meta = assemble_construct(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), htl, pd.DataFrame())

    htl_rows = meta[meta["block"] == "HTL"]
    assert len(htl_rows) == 1
    # sequence_f5 (ventana completa con flancos) de la fila ganadora, no el core.
    assert htl_rows.iloc[0]["sequence"] == "WINDOW2XXXXXXX"
    assert "n_alelos_promiscuos=5" in htl_rows.iloc[0]["source_score_note"]
    assert "glycosylated=False" in htl_rows.iloc[0]["source_score_note"]


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
    # W2/COREBBBBB tiene peor promiscuidad/%Rank pero SI tiene corte confirmado -> gana.
    # La secuencia insertada es sequence_f5 (W2), no el core.
    assert ctl_rows.iloc[0]["sequence"] == "W2"


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

    # Los bloques HTL/CTL insertan sequence_f5 (ventana completa con flancos:
    # "W1"/"W2"), no core_9aa.
    expected = (
        "BCL1" + Settings.CONSTRUCT_LINKER_BCELL + "BCL2"
        + Settings.CONSTRUCT_LINKER_INTERBLOQUE
        + "W1"
        + Settings.CONSTRUCT_LINKER_INTERBLOQUE
        + "W2"
    )
    assert seq == expected


def test_clase_vacia_se_omite_sin_linker_colgante():
    # Sin HTL: B-cell debe unirse directo a CTL con un unico linker inter-bloque,
    # no dos (uno "hacia" HTL vacio y otro "desde" HTL vacio).
    safe = _safe_df([{"accession": "A", "start": 1, "end": 3, "sequence": "BCL1", "bepipred_score": 0.9}])
    algpred = _algpred_df([["BCL1", 0.1, "Non-Allergen"]])
    ctl = pd.DataFrame([_htl_ctl_row("A", "W2", "CTLCORE", 30, 38, 4, 0.3, netcleave_match=True, netcleave_score=0.9)])

    seq, meta = assemble_construct(safe, algpred, _stackgly_df([]), pd.DataFrame(), ctl)

    assert seq == "BCL1" + Settings.CONSTRUCT_LINKER_INTERBLOQUE + "W2"
    assert "HTL" not in set(meta["block"])


def test_solo_htl_y_ctl_sin_bcell():
    # Caso real observado con un PDB (7c4s): 0 candidatos B-cell sobreviven
    # pero HTL/CTL si. El constructo debe arrancar directo con HTL, sin
    # ningun linker/hueco donde "deberia" ir el bloque B-cell.
    htl = pd.DataFrame([_htl_ctl_row("A", "W1", "HTLCORE", 20, 28, 5, 0.5)])
    ctl = pd.DataFrame([_htl_ctl_row("A", "W2", "CTLCORE", 30, 38, 4, 0.3, netcleave_match=True, netcleave_score=0.9)])

    seq, meta = assemble_construct(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), htl, ctl)

    assert seq == "W1" + Settings.CONSTRUCT_LINKER_INTERBLOQUE + "W2"
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


# --- Ranking B-cell por consenso, no por escala de un solo motor -----------------------


def test_bcell_rankea_por_consenso_no_por_escala_de_un_solo_motor():
    """DiscoTope-3.0 es un score calibrado que puede superar 1 (a diferencia de
    BepiPred/EpiDope/ScanNet, acotados a [0,1]). Antes del fix, max() de scores crudos
    dejaba que esa escala mas grande dominara el ranking. El fix usa el percentil de cada
    candidato DENTRO de su propia columna, promediado entre motores: el candidato fuerte en
    AMBOS motores (SEQC) le gana al que domina en uno solo por pura escala (SEQB), pese a
    que SEQB tiene el score crudo mas alto de toda la tabla.
    """
    rows = [
        {"accession": "A", "start": 1, "end": 4, "sequence": "SEQA", "bepipred_score": 0.95, "discotope_score": 0.1},
        {"accession": "A", "start": 10, "end": 13, "sequence": "SEQB", "bepipred_score": 0.30, "discotope_score": 45.0},
        {"accession": "A", "start": 20, "end": 23, "sequence": "SEQC", "bepipred_score": 0.60, "discotope_score": 20.0},
        {"accession": "A", "start": 30, "end": 33, "sequence": "SEQD", "bepipred_score": 0.50, "discotope_score": 10.0},
    ]
    safe = _safe_df(rows)
    algpred = _algpred_df([[r["sequence"], 0.1, "Non-Allergen"] for r in rows])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(), top_n_per_class=1
    )

    assert seq == "SEQC"


# --- Recorte de candidatos B-cell largos a su mejor sub-ventana (Settings.CONSTRUCT_BCELL_MAX_LENGTH) ---


def test_bcell_candidato_corto_no_se_recorta(tmp_path):
    safe = _safe_df([{"accession": "A", "start": 1, "end": 10, "sequence": "SHORTSEQAA", "bepipred_score": 0.9}])
    algpred = _algpred_df([["SHORTSEQAA", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="X", bcell_max_length=20,
    )

    assert seq == "SHORTSEQAA"
    note = meta.loc[meta["block"] == "B-cell", "source_score_note"].iloc[0]
    assert "trimmed_from_length" not in note


def test_bcell_candidato_largo_se_recorta_a_la_mejor_subventana(tmp_path):
    # Region fusionada de 30 aa; el tramo de mayor score por-residuo (BepiPred) esta en
    # posiciones 11-15 (1-indexado, absoluto) -- 5 residuos con score 0.9, el resto 0.1.
    full_sequence = "A" * 30
    per_residue_scores = [0.1] * 30
    for pos in range(11, 16):
        per_residue_scores[pos - 1] = 0.9

    raw_df = pd.DataFrame({
        "Accession": ["P1"] * 30,
        "Residue": list(full_sequence),
        "BepiPred-3.0 score": per_residue_scores,
    })
    raw_df.to_csv(tmp_path / "GP1_bepipred_raw.csv", index=False)

    safe = _safe_df([{"accession": "P1", "start": 1, "end": 30, "sequence": full_sequence, "bepipred_score": 0.5}])
    algpred = _algpred_df([[full_sequence, 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="GP1", bcell_max_length=5,
    )

    bcell_row = meta[meta["block"] == "B-cell"].iloc[0]
    assert len(seq) == 5
    assert bcell_row["source_start"] == 11
    assert bcell_row["source_end"] == 15
    assert "trimmed_from_length=30" in bcell_row["source_score_note"]


def test_bcell_sin_raw_cacheado_recorta_centrado_como_fallback(tmp_path):
    full_sequence = "A" * 10
    safe = _safe_df([{"accession": "P1", "start": 1, "end": 10, "sequence": full_sequence, "bepipred_score": 0.5}])
    algpred = _algpred_df([[full_sequence, 0.1, "Non-Allergen"]])

    # output_dir existe pero sin ningun '{input_stem}_{motor}_raw.csv' cacheado.
    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="NOCACHE", bcell_max_length=4,
    )

    assert len(seq) == 4
    bcell_row = meta[meta["block"] == "B-cell"].iloc[0]
    # recorte centrado como fallback: offset = (10-4)//2 = 3 -> posiciones 4-7 (1-indexado)
    assert bcell_row["source_start"] == 4
    assert bcell_row["source_end"] == 7


# --- Padding de flancos nativos en candidatos B-cell cortos (Settings.CONSTRUCT_BCELL_FLANK_THRESHOLD) ---


def _write_bepipred_raw(tmp_path, input_stem, accession, full_sequence):
    """Escribe un raw CSV minimo (Accession/Residue/score) -- suficiente para
    '_load_native_residues', el score en si no importa en estos tests de padding."""
    raw_df = pd.DataFrame({
        "Accession": [accession] * len(full_sequence),
        "Residue": list(full_sequence),
        "BepiPred-3.0 score": [0.5] * len(full_sequence),
    })
    raw_df.to_csv(tmp_path / f"{input_stem}_bepipred_raw.csv", index=False)


def _mock_predict_allergenicity(allergen_seqs):
    def _fake(sequences, output_dir, filename_prefix=""):
        return pd.DataFrame({
            "sequence": sequences,
            "algpred_score": [0.9 if s in allergen_seqs else 0.1 for s in sequences],
            "algpred_veredicto": ["Allergen" if s in allergen_seqs else "Non-Allergen" for s in sequences],
        })
    return _fake


def _mock_predict_nglycosylation(glyco_seqs):
    def _fake(sequences, output_dir, filename_prefix=""):
        rows = [s for s in sequences if s in glyco_seqs]
        return pd.DataFrame({
            "sequence": rows,
            "sequon_position": [1] * len(rows),
            "stackglyembed_veredicto": ["Glicosilado"] * len(rows),
            "stackglyembed_score": [0.9] * len(rows),
        })
    return _fake


def test_bcell_candidato_corto_se_extiende_con_flancos_nativos(monkeypatch, tmp_path):
    # Proteina de 20 aa; el candidato corto ocupa las posiciones 9-12 (1-indexado, "SHRT").
    full_sequence = "AAAAAAAASHRTAAAAAAAA"
    assert len(full_sequence) == 20
    _write_bepipred_raw(tmp_path, "GP1", "P1", full_sequence)

    monkeypatch.setattr(construct_assembly_module, "predict_allergenicity", _mock_predict_allergenicity(set()))
    monkeypatch.setattr(construct_assembly_module, "predict_nglycosylation", _mock_predict_nglycosylation(set()))

    safe = _safe_df([{"accession": "P1", "start": 9, "end": 12, "sequence": "SHRT", "bepipred_score": 0.5}])
    algpred = _algpred_df([["SHRT", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="GP1", bcell_flank_threshold=15, bcell_flank_padding=3,
    )

    bcell_row = meta[meta["block"] == "B-cell"].iloc[0]
    assert seq == "AAASHRTAAA"  # 3 aa nativos de margen a cada lado
    assert bcell_row["source_start"] == 6
    assert bcell_row["source_end"] == 15
    assert "flanked_from_length=4" in bcell_row["source_score_note"]


def test_bcell_candidato_en_el_borde_de_la_proteina_solo_extiende_hacia_adentro(monkeypatch, tmp_path):
    # El candidato ya arranca en la posicion 1 -- no hay hacia donde extender a la izquierda.
    full_sequence = "SHRTAAAAAAAAAAAAAAAA"
    assert len(full_sequence) == 20
    _write_bepipred_raw(tmp_path, "GP1", "P1", full_sequence)

    monkeypatch.setattr(construct_assembly_module, "predict_allergenicity", _mock_predict_allergenicity(set()))
    monkeypatch.setattr(construct_assembly_module, "predict_nglycosylation", _mock_predict_nglycosylation(set()))

    safe = _safe_df([{"accession": "P1", "start": 1, "end": 4, "sequence": "SHRT", "bepipred_score": 0.5}])
    algpred = _algpred_df([["SHRT", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="GP1", bcell_flank_threshold=15, bcell_flank_padding=3,
    )

    bcell_row = meta[meta["block"] == "B-cell"].iloc[0]
    assert seq == "SHRTAAA"  # sin margen a la izquierda (ya en la posicion 1), 3 aa a la derecha
    assert bcell_row["source_start"] == 1
    assert bcell_row["source_end"] == 7


def test_bcell_padding_se_aplica_igual_si_la_version_extendida_es_alergeno_pero_lo_anota(monkeypatch, tmp_path):
    full_sequence = "AAAAAAAASHRTAAAAAAAA"
    _write_bepipred_raw(tmp_path, "GP1", "P1", full_sequence)

    # DECISION 2026-08-13: la version extendida "AAASHRTAAA" es alergena
    # segun el motor (mockeado) -- el padding ya NO se descarta por eso, se
    # aplica igual y el veredicto de alergenicidad de la version YA extendida
    # queda anotado en 'allergen' (mismo tratamiento que la glicosilacion).
    monkeypatch.setattr(
        construct_assembly_module, "predict_allergenicity", _mock_predict_allergenicity({"AAASHRTAAA"})
    )
    monkeypatch.setattr(construct_assembly_module, "predict_nglycosylation", _mock_predict_nglycosylation(set()))

    safe = _safe_df([{"accession": "P1", "start": 9, "end": 12, "sequence": "SHRT", "bepipred_score": 0.5}])
    algpred = _algpred_df([["SHRT", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="GP1", bcell_flank_threshold=15, bcell_flank_padding=3,
    )

    bcell_row = meta[meta["block"] == "B-cell"].iloc[0]
    assert seq == "AAASHRTAAA"  # padding aplicado igual
    assert bcell_row["source_start"] == 9 - 3
    assert bcell_row["source_end"] == 12 + 3
    assert "allergen=True" in bcell_row["source_score_note"]
    assert "flanked_from_length=4" in bcell_row["source_score_note"]


def test_bcell_candidato_ya_largo_no_se_extiende(monkeypatch, tmp_path):
    full_sequence = "A" * 30
    _write_bepipred_raw(tmp_path, "GP1", "P1", full_sequence)

    calls = []
    monkeypatch.setattr(
        construct_assembly_module, "predict_allergenicity",
        lambda *a, **k: calls.append(1) or pd.DataFrame(columns=["sequence", "algpred_score", "algpred_veredicto"]),
    )

    long_sequence = "A" * 16  # >= threshold (15), no deberia disparar padding
    safe = _safe_df([{"accession": "P1", "start": 1, "end": 16, "sequence": long_sequence, "bepipred_score": 0.5}])
    algpred = _algpred_df([[long_sequence, 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="GP1", bcell_flank_threshold=15, bcell_flank_padding=3,
    )

    assert seq == long_sequence
    assert not calls  # predict_allergenicity nunca deberia invocarse


# --- Re-chequeo de autotolerancia sobre la secuencia FINAL (filter_self_tolerant) ------


def test_bcell_candidato_final_con_homologia_humana_se_excluye(monkeypatch, tmp_path):
    """Hallazgo de sesion: Fase 4 corre sobre la region padre (potencialmente mas larga
    que 20 aa), asi que un motivo corto peligroso enterrado dentro puede pasar su filtro
    de cobertura sin ser detectado. '_select_bcell_candidates' debe re-chequear la
    secuencia FINAL (ya recortada/extendida) y descartar la que de verdad resulte
    homologa al proteoma humano a esa escala."""
    monkeypatch.setattr(
        construct_assembly_module, "filter_self_tolerant",
        lambda df, sequence_col, **kwargs: df[df[sequence_col] != "PELIGROSA"],
    )

    safe = _safe_df([
        {"accession": "P1", "start": 1, "end": 9, "sequence": "PELIGROSA", "bepipred_score": 0.9},
        {"accession": "P1", "start": 20, "end": 28, "sequence": "INOFENSIV", "bepipred_score": 0.5},
    ])
    algpred = _algpred_df([["PELIGROSA", 0.1, "Non-Allergen"], ["INOFENSIV", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="GP1",
    )

    bcell_rows = meta[meta["block"] == "B-cell"]
    assert list(bcell_rows["sequence"]) == ["INOFENSIV"]
    assert "PELIGROSA" not in seq


def test_bcell_todos_los_candidatos_con_homologia_humana_deja_bloque_vacio(monkeypatch, tmp_path):
    monkeypatch.setattr(
        construct_assembly_module, "filter_self_tolerant",
        lambda df, sequence_col, **kwargs: df.iloc[0:0],
    )

    safe = _safe_df([{"accession": "P1", "start": 1, "end": 9, "sequence": "PELIGROSA", "bepipred_score": 0.9}])
    algpred = _algpred_df([["PELIGROSA", 0.1, "Non-Allergen"]])

    seq, meta = assemble_construct(
        safe, algpred, _stackgly_df([]), pd.DataFrame(), pd.DataFrame(),
        output_dir=tmp_path, input_stem="GP1",
    )

    assert seq == ""
    assert meta.empty
