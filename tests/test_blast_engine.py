"""Tests de la Fase 4: seleccion dinamica de task/E-value de BLASTp (src/engines/blast_engine.py).

Solo cubre las funciones puras de enrutamiento por longitud (``_select_task``,
``_select_evalue``): ``run_blastp_filter`` en si depende del binario real
'blastp' y de una base de datos indexada, fuera del alcance de un test
unitario (ver README.md - Seccion de tests, para la justificacion).
"""

from src.config.settings import Settings
from src.engines.blast_engine import _select_evalue, _select_task


def test_select_task_peptido_corto_usa_blastp_short():
    assert _select_task(9) == "blastp-short"
    assert _select_task(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN) == "blastp-short"  # limite inclusive


def test_select_task_peptido_largo_usa_blastp():
    assert _select_task(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN + 1) == "blastp"
    assert _select_task(200) == "blastp"


def test_select_evalue_tramo_corto():
    assert _select_evalue(9) == Settings.BLAST_EVALUE_SHORT
    assert _select_evalue(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN) == Settings.BLAST_EVALUE_SHORT


def test_select_evalue_tramo_medio():
    assert _select_evalue(Settings.BLAST_SHORT_PEPTIDE_MAX_LEN + 1) == Settings.BLAST_EVALUE_MEDIUM
    assert _select_evalue(Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN) == Settings.BLAST_EVALUE_MEDIUM  # limite inclusive


def test_select_evalue_tramo_largo():
    assert _select_evalue(Settings.BLAST_MEDIUM_PEPTIDE_MAX_LEN + 1) == Settings.BLAST_EVALUE_LONG
    assert _select_evalue(500) == Settings.BLAST_EVALUE_LONG
