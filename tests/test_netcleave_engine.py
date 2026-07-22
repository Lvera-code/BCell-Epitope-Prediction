"""Tests de la anotacion de Fase 5b (src/engines/netcleave_engine.py): coincidencia de
corte proteasomal C-terminal exacto entre un traceback de MHC-I/II y la salida de
NetCleave.

Cubre ``annotate_cterm_cleavage`` (funcion pura, sin subprocess): dado un candidato y una
o mas ventanas de corte candidatas, verifica que solo cuenta un match cuando
``cleavage_position`` cae EXACTO un residuo despues del ultimo residuo del candidato dentro
de su secuencia origen -- no "hay algun corte en la region" en sentido laxo. ``predict_cleavage``
en si depende del binario real (venv dedicado), fuera del alcance de un test unitario.
"""

import pandas as pd

from src.engines.netcleave_engine import annotate_cterm_cleavage


def _traceback(sequence_f5):
    return pd.DataFrame({"accession": ["ACC1"], "sequence_f5": [sequence_f5]})


def _cleavage_row(source_sequence, cleavage_position, cleavage_score=0.5):
    return {
        "sequence_window": "XXX|XXX",
        "cleavage_position": cleavage_position,
        "cleavage_residue": "A",
        "cleavage_score": cleavage_score,
        "source_sequence": source_sequence,
    }


def test_traceback_vacio_devuelve_columnas_nuevas_sin_matches():
    result = annotate_cterm_cleavage(pd.DataFrame(columns=["accession", "sequence_f5"]), pd.DataFrame())

    assert list(result.columns) == ["accession", "sequence_f5", "netcleave_c_term_match", "netcleave_c_term_score"]
    assert result.empty


def test_cleavage_vacio_no_matchea_nada():
    traceback_df = _traceback("DEFGHIJK")

    result = annotate_cterm_cleavage(traceback_df, pd.DataFrame())

    assert result.iloc[0]["netcleave_c_term_match"] == False  # noqa: E712 (numpy bool, no Python bool)
    assert pd.isna(result.iloc[0]["netcleave_c_term_score"])


def test_corte_exacto_en_el_limite_c_terminal_matchea():
    # source_sequence = 'ABCDEFGHIJKLMNOP', candidato 'DEFGHIJK' termina en
    # offset 3 (0-idx) + len 8 = 11 (0-idx exclusivo) -> el corte 1-indexado
    # INMEDIATAMENTE DESPUES cae en la posicion 3 + 8 + 1 = 12.
    traceback_df = _traceback("DEFGHIJK")
    cleavage_df = pd.DataFrame([_cleavage_row("ABCDEFGHIJKLMNOP", cleavage_position=12, cleavage_score=0.87)])

    result = annotate_cterm_cleavage(traceback_df, cleavage_df)

    assert result.iloc[0]["netcleave_c_term_match"] == True  # noqa: E712 (numpy bool, no Python bool)
    assert result.iloc[0]["netcleave_c_term_score"] == 0.87


def test_corte_desplazado_un_residuo_no_matchea():
    traceback_df = _traceback("DEFGHIJK")
    # Un residuo antes (11) y uno despues (13) del limite exacto (12): ninguno cuenta.
    cleavage_df = pd.DataFrame(
        [
            _cleavage_row("ABCDEFGHIJKLMNOP", cleavage_position=11),
            _cleavage_row("ABCDEFGHIJKLMNOP", cleavage_position=13),
        ]
    )

    result = annotate_cterm_cleavage(traceback_df, cleavage_df)

    assert result.iloc[0]["netcleave_c_term_match"] == False  # noqa: E712 (numpy bool, no Python bool)


def test_candidato_sin_secuencia_origen_correspondiente_no_matchea():
    traceback_df = _traceback("ZZZZZZZZ")  # no es substring de ninguna source_sequence
    cleavage_df = pd.DataFrame([_cleavage_row("ABCDEFGHIJKLMNOP", cleavage_position=12)])

    result = annotate_cterm_cleavage(traceback_df, cleavage_df)

    assert result.iloc[0]["netcleave_c_term_match"] == False  # noqa: E712 (numpy bool, no Python bool)


def test_toma_el_mejor_score_entre_multiples_matches_exactos():
    traceback_df = _traceback("DEFGHIJK")
    # Dos secuencias origen distintas, ambas contienen el candidato con
    # corte exacto en el limite -- debe quedarse con el score mas alto.
    cleavage_df = pd.DataFrame(
        [
            _cleavage_row("ABCDEFGHIJKLMNOP", cleavage_position=12, cleavage_score=0.30),
            # 'DEFGHIJK' arranca en offset 2 de 'XXDEFGHIJKYYY' -> limite exacto = 2+8+1 = 11.
            _cleavage_row("XXDEFGHIJKYYY", cleavage_position=11, cleavage_score=0.95),
        ]
    )

    result = annotate_cterm_cleavage(traceback_df, cleavage_df)

    assert result.iloc[0]["netcleave_c_term_match"] == True  # noqa: E712 (numpy bool, no Python bool)
    assert result.iloc[0]["netcleave_c_term_score"] == 0.95


def test_multiples_candidatos_independientes():
    traceback_df = pd.DataFrame(
        {"accession": ["ACC1", "ACC2"], "sequence_f5": ["DEFGHIJK", "NOMATCH"]}
    )
    cleavage_df = pd.DataFrame([_cleavage_row("ABCDEFGHIJKLMNOP", cleavage_position=12, cleavage_score=0.87)])

    result = annotate_cterm_cleavage(traceback_df, cleavage_df)

    assert result.iloc[0]["netcleave_c_term_match"] == True  # noqa: E712 (numpy bool, no Python bool)
    assert result.iloc[1]["netcleave_c_term_match"] == False  # noqa: E712 (numpy bool, no Python bool)
    assert result.iloc[0]["accession"] == "ACC1"
    assert result.iloc[1]["accession"] == "ACC2"


def test_no_muta_el_traceback_original():
    traceback_df = _traceback("DEFGHIJK")
    original_columns = list(traceback_df.columns)

    annotate_cterm_cleavage(traceback_df, pd.DataFrame())

    assert list(traceback_df.columns) == original_columns
