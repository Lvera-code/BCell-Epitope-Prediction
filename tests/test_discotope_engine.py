"""Tests de ``_load_raw_scores`` (src/engines/discotope_engine.py).

CSVs sinteticos minimos construidos aqui mismo (``_load_raw_scores`` solo lee
un CSV de disco, sin subprocess ni red) -- mismo criterio que
``test_structure_parser.py``.
"""

import pandas as pd

from src.engines.discotope_engine import (
    ACCESSION_COLUMN,
    ALPHAFOLD_STRUC_FLAG_COLUMN,
    DiscoTopeEngine,
    PLDDT_COLUMN,
    RESIDUE_COLUMN,
    RSA_COLUMN,
    SCORE_COLUMN,
)


def _write_csv(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_retiene_rsa_plddt_y_alphafold_flag_del_csv_real(tmp_path):
    """Reproduce el esquema real confirmado empiricamente (ver ADR del modulo):
    'pdb,chain,res_id,residue,DiscoTope-3.0_score,calibrated_score,epitope,
    rsa,pLDDTs,length,alphafold_struc_flag'."""
    _write_csv(
        tmp_path, "demo_A_discotope3.csv",
        "pdb,chain,res_id,residue,DiscoTope-3.0_score,calibrated_score,epitope,"
        "rsa,pLDDTs,length,alphafold_struc_flag\n"
        "demo_A,A,1,M,0.10,0.42,False,0.64,100,845,0\n"
        "demo_A,A,2,P,0.20,1.55,True,0.05,100,845,0\n",
    )

    df = DiscoTopeEngine._load_raw_scores(tmp_path, accession="demo")

    assert list(df.columns) == [
        ACCESSION_COLUMN, RESIDUE_COLUMN, SCORE_COLUMN,
        RSA_COLUMN, PLDDT_COLUMN, ALPHAFOLD_STRUC_FLAG_COLUMN,
    ]
    assert df[SCORE_COLUMN].tolist() == [0.42, 1.55]  # calibrated_score, NO la cruda
    assert df[RSA_COLUMN].tolist() == [0.64, 0.05]
    assert df[PLDDT_COLUMN].tolist() == [100, 100]
    assert df[ALPHAFOLD_STRUC_FLAG_COLUMN].tolist() == [0, 0]
    assert (df[ACCESSION_COLUMN] == "demo").all()


def test_degrada_a_na_sin_romper_si_el_csv_no_trae_las_columnas_nuevas(tmp_path):
    """Robustez real: si una version futura (u otra instalacion) de DiscoTope-3.0
    no emite rsa/pLDDTs/alphafold_struc_flag, no debe fallar -- ningun consumidor
    de Project 1 las requeria antes de que existieran, no deben convertirse en
    un nuevo punto de fallo."""
    _write_csv(
        tmp_path, "demo_A_discotope3.csv",
        "residue,calibrated_score\nM,0.42\nP,1.55\n",
    )

    df = DiscoTopeEngine._load_raw_scores(tmp_path, accession="demo")

    assert df[SCORE_COLUMN].tolist() == [0.42, 1.55]
    assert df[RSA_COLUMN].isna().all()
    assert df[PLDDT_COLUMN].isna().all()
    assert df[ALPHAFOLD_STRUC_FLAG_COLUMN].isna().all()
