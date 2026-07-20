"""Tests de Fase 1.5 (src/utils/structure_parser.py).

PDBs sintéticos minimos (2-4 residuos) construidos aqui mismo, sin descargas
de red ni dependencia de ningun binario externo (gemmi es una libreria pura,
sin subprocess).
"""

import pytest

from src.utils.exceptions import StructureParsingError
from src.utils.structure_parser import parse_structure

PDB_WITH_MODIFIED_RESIDUES = (
    "HEADER    TEST\n"
    "ATOM      1  N   MET A   1      11.104  13.207   2.100  1.00 20.00           N\n"
    "ATOM      2  CA  MET A   1      12.560  13.207   2.100  1.00 20.00           C\n"
    "ATOM      3  C   MET A   1      13.100  14.600   2.100  1.00 20.00           C\n"
    "HETATM    4  N   MSE A   2      14.500  14.700   2.100  1.00 20.00           N\n"
    "HETATM    5  CA  MSE A   2      15.000  15.700   2.100  1.00 20.00           C\n"
    "HETATM    6  SE  MSE A   2      15.500  16.200   2.500  1.00 20.00          SE\n"
    "HETATM    7  N   SEP A   3      16.500  16.700   2.100  1.00 20.00           N\n"
    "HETATM    8  CA  SEP A   3      17.000  17.700   2.100  1.00 20.00           C\n"
    "HETATM    9  OG  SEP A   3      17.500  18.200   2.500  1.00 20.00           O\n"
    "ATOM     10  N   GLY A   4      18.500  18.700   2.100  1.00 20.00           N\n"
    "ATOM     11  CA  GLY A   4      19.000  19.700   2.100  1.00 20.00           C\n"
    "END\n"
)

PDB_MULTI_CHAIN = (
    "HEADER    TEST\n"
    "ATOM      1  N   VAL A   1      11.104  13.207   2.100  1.00 20.00           N\n"
    "ATOM      2  CA  VAL A   1      12.560  13.207   2.100  1.00 20.00           C\n"
    "ATOM      3  N   ALA A   2      14.500  14.700   2.100  1.00 20.00           N\n"
    "ATOM      4  CA  ALA A   2      15.000  15.700   2.100  1.00 20.00           C\n"
    "ATOM      5  N   MET B   1      21.000  21.700   2.100  1.00 20.00           N\n"
    "ATOM      6  CA  MET B   1      22.000  22.700   2.100  1.00 20.00           C\n"
    "ATOM      7  N   ALA B   2      23.000  23.700   2.100  1.00 20.00           N\n"
    "ATOM      8  CA  ALA B   2      24.000  24.700   2.100  1.00 20.00           C\n"
    "ATOM      9  N   GLY B   3      25.000  25.700   2.100  1.00 20.00           N\n"
    "ATOM     10  CA  GLY B   3      26.000  26.700   2.100  1.00 20.00           C\n"
    "END\n"
)

PDB_ONLY_WATERS = (
    "HEADER    TEST\n"
    "HETATM    1  O   HOH A   1      11.104  13.207   2.100  1.00 20.00           O\n"
    "HETATM    2  O   HOH A   2      14.500  14.700   2.100  1.00 20.00           O\n"
    "END\n"
)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_residuos_modificados_se_resuelven_via_ccd(tmp_path):
    pdb_path = _write(tmp_path, "modres.pdb", PDB_WITH_MODIFIED_RESIDUES)
    output_dir = tmp_path / "out"

    record = parse_structure(pdb_path, output_dir)

    assert record.sequence == "MMSG"  # MET, MSE->M, SEP->S, GLY
    assert len(record.sequence) == len(record.position_mapping)
    assert record.fasta_path.is_file()
    assert record.fasta_path.read_text().splitlines()[0] == f">{record.accession}"
    assert record.chain_pdb_path.is_file()
    assert record.pdb_path == pdb_path  # original sin modificar


def test_position_mapping_columnas_y_numeracion(tmp_path):
    pdb_path = _write(tmp_path, "modres.pdb", PDB_WITH_MODIFIED_RESIDUES)
    record = parse_structure(pdb_path, tmp_path / "out")

    mapping = record.position_mapping
    assert list(mapping.columns) == [
        "accession", "chain_id", "pdb_seqid", "insertion_code", "fasta_position", "residue_letter",
    ]
    assert mapping["fasta_position"].tolist() == [1, 2, 3, 4]
    assert mapping["pdb_seqid"].tolist() == [1, 2, 3, 4]
    assert mapping["residue_letter"].tolist() == ["M", "M", "S", "G"]
    assert (mapping["accession"] == record.accession).all()
    assert (mapping["chain_id"] == "A").all()


def test_multi_cadena_estrategia_longest_elige_la_mas_larga(tmp_path):
    pdb_path = _write(tmp_path, "multi.pdb", PDB_MULTI_CHAIN)

    record = parse_structure(pdb_path, tmp_path / "out", chain_selection_strategy="longest")

    assert record.chain_id == "B"  # B tiene 3 residuos, A tiene 2
    assert record.sequence == "MAG"

    # El PDB de una sola cadena debe contener SOLO la cadena elegida (B), no la A.
    chain_pdb_text = record.chain_pdb_path.read_text()
    atom_chain_ids = {line[21] for line in chain_pdb_text.splitlines() if line.startswith("ATOM")}
    assert atom_chain_ids == {"B"}


def test_multi_cadena_estrategia_explicit_respeta_chain_id(tmp_path):
    pdb_path = _write(tmp_path, "multi.pdb", PDB_MULTI_CHAIN)

    record = parse_structure(
        pdb_path, tmp_path / "out", chain_selection_strategy="explicit", explicit_chain_id="A"
    )

    assert record.chain_id == "A"
    assert record.sequence == "VA"


def test_estrategia_explicit_sin_chain_id_lanza_error(tmp_path):
    pdb_path = _write(tmp_path, "multi.pdb", PDB_MULTI_CHAIN)

    with pytest.raises(StructureParsingError):
        parse_structure(pdb_path, tmp_path / "out", chain_selection_strategy="explicit", explicit_chain_id="")


def test_estrategia_explicit_con_chain_id_inexistente_lanza_error(tmp_path):
    pdb_path = _write(tmp_path, "multi.pdb", PDB_MULTI_CHAIN)

    with pytest.raises(StructureParsingError):
        parse_structure(
            pdb_path, tmp_path / "out", chain_selection_strategy="explicit", explicit_chain_id="Z"
        )


def test_sin_cadena_proteica_valida_lanza_structure_parsing_error(tmp_path):
    pdb_path = _write(tmp_path, "waters.pdb", PDB_ONLY_WATERS)

    with pytest.raises(StructureParsingError):
        parse_structure(pdb_path, tmp_path / "out")


def test_archivo_inexistente_lanza_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_structure(tmp_path / "no_existe.pdb", tmp_path / "out")


def test_accession_deriva_del_stem_del_archivo(tmp_path):
    pdb_path = _write(tmp_path, "mi_estructura_custom.pdb", PDB_WITH_MODIFIED_RESIDUES)
    record = parse_structure(pdb_path, tmp_path / "out")
    assert record.accession == "mi_estructura_custom"
