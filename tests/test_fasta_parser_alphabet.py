"""Tests del alfabeto de compatibilidad con BepiPred/EpiDope (src/utils/fasta_parser.py).

Cubre `is_bepipred_compatible` (Camino 3: gate no-fatal para el FASTA
derivado de una estructura) y confirma que `load_and_sanitize` (Camino 1:
FASTA subido por el usuario) sigue rechazando no-canonicos de forma fatal,
sin ningun cambio de comportamiento -- la alternativa de relajar
CANONICAL_AMINOACIDS se descarto tras confirmar empiricamente que
BepiPred-3.0 sigue rechazando 'X' (exit code 1) en su instalacion local real.
"""

import pytest

from src.utils.exceptions import FastaFormatError
from src.utils.fasta_parser import CANONICAL_AMINOACIDS, is_bepipred_compatible, load_and_sanitize


@pytest.mark.parametrize("residue", ["X", "B", "Z", "U", "O", "J", "-", "*"])
def test_no_canonicos_marcan_incompatible_con_bepipred(residue):
    compatible, invalid = is_bepipred_compatible(f"MKT{residue}AYIAKQ")
    assert compatible is False
    assert invalid == [residue]


def test_secuencia_100pct_canonica_es_compatible():
    compatible, invalid = is_bepipred_compatible("MKTAYIAKQRQISFVKSHFSRQ")
    assert compatible is True
    assert invalid == []


def test_reporta_todos_los_no_canonicos_unicos_y_ordenados():
    compatible, invalid = is_bepipred_compatible("MXKTBXAYU")
    assert compatible is False
    assert invalid == sorted({"X", "B", "U"})


def test_canonical_aminoacidos_no_se_relajo():
    # CANONICAL_AMINOACIDS debe seguir siendo exactamente los 20 estandar:
    # la alternativa de ampliarlo a IUPAC extendido (X/B/Z/U/O) se descarto
    # porque BepiPred-3.0 los rechaza igual (ver docstring del modulo).
    assert CANONICAL_AMINOACIDS == set("ACDEFGHIKLMNPQRSTVWY")


def test_load_and_sanitize_sigue_siendo_fatal_ante_no_canonicos(tmp_path):
    # Camino 1 (FASTA subido por el usuario) no cambia de comportamiento.
    fasta_path = tmp_path / "bad.fasta"
    fasta_path.write_text(">ACC1\nMKTXAYIAKQ\n")

    with pytest.raises(FastaFormatError):
        load_and_sanitize(fasta_path)
