"""Tests de parse_args (pipeline.py): validacion temprana de --alelo-extra.

No corre ninguna fase del pipeline, solo el parseo de argumentos via argparse.
"""

import pytest

from pipeline import parse_args


def test_alelo_extra_ausente_no_falla():
    args = parse_args(["--input", "x.fasta"])
    assert args.alelo_extra is None


def test_alelo_extra_valido_se_conserva_tal_cual():
    args = parse_args(["--input", "x.fasta", "--alelo-extra", "DRB1_1602"])
    assert args.alelo_extra == "DRB1_1602"


def test_alelo_extra_con_espacio_falla_al_parsear(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--input", "x.fasta", "--alelo-extra", "DRB1_1602, DRB1_1301"])
    err = capsys.readouterr().err
    assert "--alelo-extra" in err
    assert "espacio" in err.lower()


def test_alelo_extra_formato_invalido_falla_al_parsear(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--input", "x.fasta", "--alelo-extra", "DRB1_16O2"])
    err = capsys.readouterr().err
    assert "--alelo-extra" in err
    assert "formato invalido" in err.lower()
