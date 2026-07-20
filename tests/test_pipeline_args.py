"""Tests de parse_args (pipeline.py): validacion temprana de --alelo-extra y --pdb-mode.

No corre ninguna fase del pipeline, solo el parseo de argumentos via argparse.
"""

import pytest

from pipeline import parse_args
from src.config.settings import Settings


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


def test_pdb_mode_ausente_por_defecto_none():
    # None -> _resolve_active_engines_and_inputs cae a Settings.PDB_PROCESSING_MODE.
    args = parse_args(["--input", "x.pdb"])
    assert args.pdb_mode is None


def test_pdb_mode_acepta_structure_only():
    args = parse_args(["--input", "x.pdb", "--pdb-mode", "structure_only"])
    assert args.pdb_mode == "structure_only"


def test_pdb_mode_acepta_structure_and_sequence():
    args = parse_args(["--input", "x.pdb", "--pdb-mode", "structure_and_sequence"])
    assert args.pdb_mode == "structure_and_sequence"


def test_pdb_mode_valor_invalido_falla_al_parsear(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--input", "x.pdb", "--pdb-mode", "modo_inventado"])
    err = capsys.readouterr().err
    assert "--pdb-mode" in err


def test_discotope_threshold_default_es_el_valor_oficial():
    args = parse_args(["--input", "x.pdb"])
    assert args.discotope_threshold == Settings.DISCOTOPE_THRESHOLD == 0.90


def test_discotope_threshold_y_min_length_configurables():
    args = parse_args(["--input", "x.pdb", "--discotope-threshold", "0.40", "--discotope-min-length", "12"])
    assert args.discotope_threshold == 0.40
    assert args.discotope_min_length == 12


def test_scannet_threshold_ausente_por_defecto_none():
    # None -> extract_epitopes usa el umbral ADAPTATIVO por percentil, no un
    # numero fijo (ScanNet no publica un umbral absoluto oficial).
    args = parse_args(["--input", "x.pdb"])
    assert args.scannet_threshold is None


def test_scannet_threshold_explicito_desactiva_el_modo_adaptativo():
    args = parse_args(["--input", "x.pdb", "--scannet-threshold", "0.15"])
    assert args.scannet_threshold == 0.15


def test_scannet_min_length_configurable():
    args = parse_args(["--input", "x.pdb", "--scannet-min-length", "10"])
    assert args.scannet_min_length == 10
