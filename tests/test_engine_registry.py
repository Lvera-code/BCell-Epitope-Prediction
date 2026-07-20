"""Tests de src/engines/engine_registry.py: resolucion del subconjunto de motores activo."""

import pytest

from src.engines.engine_registry import ENGINE_REGISTRY, active_engines_for
from src.utils.exceptions import InputRoutingError


def test_camino1_fasta_activa_solo_motores_de_secuencia():
    assert active_engines_for("fasta", None) == ["bepipred", "epidope"]


def test_camino2_structure_only_activa_solo_motores_estructurales():
    assert active_engines_for("structure", "structure_only") == ["discotope", "scannet"]


def test_camino3_structure_and_sequence_activa_los_4():
    assert active_engines_for("structure", "structure_and_sequence") == [
        "bepipred", "epidope", "discotope", "scannet",
    ]


def test_structure_sin_pdb_mode_reconocido_lanza_error():
    with pytest.raises(InputRoutingError):
        active_engines_for("structure", "modo_inventado")
    with pytest.raises(InputRoutingError):
        active_engines_for("structure", None)


def test_input_type_no_reconocido_lanza_error():
    with pytest.raises(InputRoutingError):
        active_engines_for("no_existe", None)


def test_engine_registry_declara_tipo_de_input_correcto_por_motor():
    assert ENGINE_REGISTRY["bepipred"][1] == "fasta"
    assert ENGINE_REGISTRY["epidope"][1] == "fasta"
    assert ENGINE_REGISTRY["discotope"][1] == "pdb"
    assert ENGINE_REGISTRY["scannet"][1] == "pdb"
