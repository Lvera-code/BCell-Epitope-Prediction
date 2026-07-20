"""Tests del enrutador de input (src/utils/input_router.py).

Logica 100% pura (I/O local sobre archivos de texto minimos escritos por el
propio test via tmp_path), sin subprocess ni binarios externos.
"""

import pytest

from src.utils.exceptions import InputRoutingError
from src.utils.input_router import route_input, route_inputs

FASTA_CONTENT = ">ACC1 some protein\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV\n"

PDB_CONTENT = (
    "HEADER    HYDROLASE                              01-JAN-00   1ABC\n"
    "ATOM      1  N   MET A   1      11.104  13.207   2.100  1.00 20.00           N\n"
    "ATOM      2  CA  MET A   1      12.560  13.207   2.100  1.00 20.00           C\n"
    "END\n"
)

MMCIF_CONTENT = (
    "data_1ABC\n"
    "_entry.id 1ABC\n"
    "loop_\n"
    "_atom_site.group_PDB\n"
    "ATOM 1 N MET A 1 11.104 13.207 2.100 1.00 20.00 N\n"
)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_fasta_correcto_por_extension_y_contenido(tmp_path):
    p = _write(tmp_path, "seq.fasta", FASTA_CONTENT)
    result = route_input(p)
    assert result.input_type == "fasta"
    assert result.path == p


def test_pdb_correcto_por_extension_y_contenido(tmp_path):
    p = _write(tmp_path, "structure.pdb", PDB_CONTENT)
    result = route_input(p)
    assert result.input_type == "structure"


def test_mmcif_correcto_por_extension_y_contenido(tmp_path):
    p = _write(tmp_path, "structure.cif", MMCIF_CONTENT)
    result = route_input(p)
    assert result.input_type == "structure"


def test_extension_incorrecta_pero_contenido_reconocible_fasta(tmp_path):
    # Extension .txt (no reconocida), pero el contenido es FASTA valido.
    p = _write(tmp_path, "seq.txt", FASTA_CONTENT)
    result = route_input(p)
    assert result.input_type == "fasta"


def test_extension_incorrecta_pero_contenido_reconocible_estructura(tmp_path):
    p = _write(tmp_path, "structure.dat", PDB_CONTENT)
    result = route_input(p)
    assert result.input_type == "structure"


def test_extension_y_contenido_en_conflicto_prioriza_contenido(tmp_path):
    # Extension dice FASTA, contenido real es PDB: debe ganar el contenido.
    p = _write(tmp_path, "mislabeled.fasta", PDB_CONTENT)
    result = route_input(p)
    assert result.input_type == "structure"


def test_archivo_no_reconocible_lanza_input_routing_error(tmp_path):
    p = _write(tmp_path, "mystery.dat", "esto no es ni FASTA ni una estructura\nlinea 2\n")
    with pytest.raises(InputRoutingError):
        route_input(p)


def test_archivo_vacio_lanza_input_routing_error(tmp_path):
    p = _write(tmp_path, "empty.dat", "")
    with pytest.raises(InputRoutingError):
        route_input(p)


def test_archivo_inexistente_lanza_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        route_input(tmp_path / "no_existe.fasta")


def test_route_inputs_preserva_orden(tmp_path):
    p1 = _write(tmp_path, "a.fasta", FASTA_CONTENT)
    p2 = _write(tmp_path, "b.pdb", PDB_CONTENT)

    results = route_inputs([p1, p2])

    assert [r.input_type for r in results] == ["fasta", "structure"]
    assert [r.path for r in results] == [p1, p2]
