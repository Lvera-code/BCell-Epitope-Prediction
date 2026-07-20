"""Test de integracion liviano de los 3 caminos de entrada (pipeline.py), motores mockeados.

Verifica que `_resolve_active_engines_and_inputs` + `fase_2_antigenicidad` +
`fase_3_mapeo_y_union` invoquen exactamente el subconjunto de motores
esperado por camino, y que el resto del flujo (extraccion de epitopos, union
anotada) componga sin errores con esos motores mockeados. Ningun binario
externo real (BepiPred/EpiDope/DiscoTope-3.0/ScanNet) se ejecuta: se
monkeypatchea el metodo `run` de cada clase de motor.
"""

import pathlib

import pandas as pd
import pytest

from pipeline import (
    _cached_raw_scores,
    _cached_structural_raw_scores,
    _resolve_active_engines_and_inputs,
    fase_2_antigenicidad,
    fase_3_mapeo_y_union,
)
from src.engines.bepipred_engine import BepiPredEngine
from src.engines.discotope_engine import DiscoTopeEngine
from src.engines.discotope_engine import SCORE_COLUMN as DISCOTOPE_SCORE_COLUMN
from src.engines.epidope_engine import EpidopeEngine
from src.engines.scannet_engine import ScanNetEngine

FASTA_CONTENT = ">ACC1 test protein\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL\n"

PDB_CONTENT = (
    "HEADER    TEST\n"
    "ATOM      1  N   MET A   1      11.104  13.207   2.100  1.00 20.00           N\n"
    "ATOM      2  CA  MET A   1      12.560  13.207   2.100  1.00 20.00           C\n"
    "ATOM      3  C   MET A   1      13.100  14.600   2.100  1.00 20.00           C\n"
    "ATOM      4  N   ALA A   2      14.500  14.700   2.100  1.00 20.00           N\n"
    "ATOM      5  CA  ALA A   2      15.000  15.700   2.100  1.00 20.00           C\n"
    "ATOM      6  C   ALA A   2      15.500  16.700   2.100  1.00 20.00           C\n"
    "ATOM      7  N   GLY A   3      18.500  18.700   2.100  1.00 20.00           N\n"
    "ATOM      8  CA  GLY A   3      19.000  19.700   2.100  1.00 20.00           C\n"
    "ATOM      9  C   GLY A   3      19.500  20.700   2.100  1.00 20.00           C\n"
    "END\n"
)

# Contiene 'ZZZ', un codigo de residuo no reconocido por el CCD (no mapeable
# -> 'X' en Fase 1.5), para ejercitar el gate no-fatal de Camino 3.
PDB_CONTENT_UNMAPPABLE = (
    "HEADER    TEST\n"
    "ATOM      1  N   MET A   1      11.104  13.207   2.100  1.00 20.00           N\n"
    "ATOM      2  CA  MET A   1      12.560  13.207   2.100  1.00 20.00           C\n"
    "ATOM      3  C   MET A   1      13.100  14.600   2.100  1.00 20.00           C\n"
    "ATOM      4  N   ZZZ A   2      14.500  14.700   2.100  1.00 20.00           N\n"
    "ATOM      5  CA  ZZZ A   2      15.000  15.700   2.100  1.00 20.00           C\n"
    "ATOM      6  C   ZZZ A   2      15.500  16.700   2.100  1.00 20.00           C\n"
    "ATOM      7  N   GLY A   3      18.500  18.700   2.100  1.00 20.00           N\n"
    "ATOM      8  CA  GLY A   3      19.000  19.700   2.100  1.00 20.00           C\n"
    "ATOM      9  C   GLY A   3      19.500  20.700   2.100  1.00 20.00           C\n"
    "END\n"
)


def _fake_run_fasta(accession_col, residue_col, score_col, n=20):
    def _run(self, items, output_dir=None):
        results = []
        for item in items:
            # Mismo criterio que BepiPred/EpiDope reales: el accession sale
            # del header del FASTA que se le paso, no de un valor fijo (en
            # Camino 3 el FASTA derivado tiene como header
            # structure_record.accession, no el accession de Camino 1).
            header = pathlib.Path(item).read_text().splitlines()[0]
            accession = header.lstrip(">").split()[0]
            results.append(pd.DataFrame({
                accession_col: [accession] * n,
                residue_col: ["M"] * n,
                score_col: [1.0] * n,
            }))
        return results
    return _run


def _fake_run_pdb(accession_col, residue_col, score_col, n=20):
    def _run(self, items, output_dir=None):
        return [
            pd.DataFrame({
                accession_col: [pathlib.Path(item).stem] * n,
                residue_col: ["M"] * n,
                score_col: [1.0] * n,
            })
            for item in items
        ]
    return _run


@pytest.fixture
def mock_all_engines(monkeypatch):
    monkeypatch.setattr(BepiPredEngine, "run", _fake_run_fasta("Accession", "Residue", "BepiPred-3.0 score"))
    monkeypatch.setattr(EpidopeEngine, "run", _fake_run_fasta("Accession", "Residue", "EpiDope score"))
    monkeypatch.setattr(DiscoTopeEngine, "run", _fake_run_pdb("Accession", "Residue", DISCOTOPE_SCORE_COLUMN))
    monkeypatch.setattr(ScanNetEngine, "run", _fake_run_pdb("Accession", "Residue", "ScanNet score"))


def test_camino1_fasta_invoca_solo_bepipred_y_epidope(tmp_path, mock_all_engines):
    fasta_path = tmp_path / "seq.fasta"
    fasta_path.write_text(FASTA_CONTENT)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    active_engines, clean_fasta, structure_record = _resolve_active_engines_and_inputs(
        fasta_path, output_dir, pdb_mode_override=None
    )
    assert active_engines == ["bepipred", "epidope"]
    assert structure_record is None
    assert clean_fasta is not None

    raw_dfs = fase_2_antigenicidad(active_engines, fasta_path.stem, clean_fasta, structure_record, output_dir)
    assert set(raw_dfs.keys()) == {"bepipred", "epidope"}

    union_df = fase_3_mapeo_y_union(raw_dfs, structure_record, 0.5, 9, 0.5, 9, output_dir, fasta_path.stem)
    assert not union_df.empty
    assert set(union_df["origen"]) == {"Bp+Ed"}


def test_camino2_structure_only_invoca_solo_motores_estructurales(tmp_path, mock_all_engines):
    pdb_path = tmp_path / "structure.pdb"
    pdb_path.write_text(PDB_CONTENT)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    active_engines, clean_fasta, structure_record = _resolve_active_engines_and_inputs(
        pdb_path, output_dir, pdb_mode_override="structure_only"
    )
    assert active_engines == ["discotope", "scannet"]
    assert clean_fasta is None
    assert structure_record is not None

    raw_dfs = fase_2_antigenicidad(active_engines, pdb_path.stem, clean_fasta, structure_record, output_dir)
    assert set(raw_dfs.keys()) == {"discotope", "scannet"}
    # BepiPred/EpiDope NUNCA se invocan en este camino (ver mock: si se
    # hubieran llamado, apareceria su clave en raw_dfs).
    assert "bepipred" not in raw_dfs
    assert "epidope" not in raw_dfs

    union_df = fase_3_mapeo_y_union(raw_dfs, structure_record, 0.5, 9, 0.5, 9, output_dir, pdb_path.stem)
    assert not union_df.empty
    assert set(union_df["origen"]) == {"Dt+Sn"}
    assert "bepipred_score" not in union_df.columns
    assert "epidope_score" not in union_df.columns


def test_camino3_structure_and_sequence_invoca_los_4_motores(tmp_path, mock_all_engines):
    pdb_path = tmp_path / "structure.pdb"
    pdb_path.write_text(PDB_CONTENT)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    active_engines, clean_fasta, structure_record = _resolve_active_engines_and_inputs(
        pdb_path, output_dir, pdb_mode_override="structure_and_sequence"
    )
    assert active_engines == ["bepipred", "epidope", "discotope", "scannet"]
    assert clean_fasta == structure_record.fasta_path

    raw_dfs = fase_2_antigenicidad(active_engines, pdb_path.stem, clean_fasta, structure_record, output_dir)
    assert set(raw_dfs.keys()) == {"bepipred", "epidope", "discotope", "scannet"}

    union_df = fase_3_mapeo_y_union(raw_dfs, structure_record, 0.5, 9, 0.5, 9, output_dir, pdb_path.stem)
    assert not union_df.empty
    assert set(union_df["origen"]) == {"Consenso total"}

    # Las posiciones de motores estructurales y de secuencia coinciden: ambas
    # provienen del mismo sequence_lookup (StructureRecord.sequence), que a
    # su vez coincide con position_mapping.fasta_position (ver ADR de
    # consensus.py).
    assert len(structure_record.sequence) == len(structure_record.position_mapping)


def test_camino3_con_residuo_no_mapeable_excluye_bepipred_epidope(tmp_path, mock_all_engines, caplog):
    pdb_path = tmp_path / "structure_unmappable.pdb"
    pdb_path.write_text(PDB_CONTENT_UNMAPPABLE)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    active_engines, clean_fasta, structure_record = _resolve_active_engines_and_inputs(
        pdb_path, output_dir, pdb_mode_override="structure_and_sequence"
    )

    assert "X" in structure_record.sequence  # ZZZ no se pudo mapear
    assert active_engines == ["discotope", "scannet"]  # bepipred/epidope excluidos
    assert clean_fasta is None
    assert any("no canonico" in msg for msg in caplog.messages)

    raw_dfs = fase_2_antigenicidad(active_engines, pdb_path.stem, clean_fasta, structure_record, output_dir)
    assert set(raw_dfs.keys()) == {"discotope", "scannet"}


def test_desfase_de_filas_de_motor_estructural_se_loguea(tmp_path, monkeypatch, caplog):
    # Reproduce el hallazgo real (2026-07-20, PDB sintetico con residuo no
    # mapeable): DiscoTope-3.0 descarto en silencio 1 de 3 residuos (2 filas
    # de salida en vez de 3). _cached_structural_raw_scores debe detectar
    # ese desfase apenas recibe los scores crudos, sin depender de que se
    # llegue a extraer una region de epitopo.
    pdb_path = tmp_path / "structure.pdb"
    pdb_path.write_text(PDB_CONTENT)  # 3 residuos: M, A, G
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _, _, structure_record = _resolve_active_engines_and_inputs(pdb_path, output_dir, pdb_mode_override="structure_only")
    assert len(structure_record.sequence) == 3

    def _short_run(self, items, output_dir=None):
        # Simula que el motor devolvio 1 fila menos de las esperadas.
        return [pd.DataFrame({
            "Accession": ["whatever"] * 2,
            "Residue": ["M", "G"],
            DISCOTOPE_SCORE_COLUMN: [0.5, 0.5],
        })]

    monkeypatch.setattr(DiscoTopeEngine, "run", _short_run)

    df = _cached_structural_raw_scores(
        engine_name="DiscoTope-3.0",
        cache_path=output_dir / "raw.csv",
        raw_artifacts_dir=output_dir / "_raw",
        structure_record=structure_record,
        engine=DiscoTopeEngine(),
    )

    assert len(df) == 2
    assert any("posible" in msg.lower() and "desalineacion" in msg.lower() for msg in caplog.messages) or any(
        "devolvio" in msg for msg in caplog.messages
    )


def test_cache_se_invalida_cuando_cambia_el_contenido_del_input(tmp_path, monkeypatch):
    # Regresion real (2026-07-20, 6xc2.pdb corrido primero con estrategia
    # 'longest' -pica el Fab del anticuerpo- y despues con 'explicit' -pica
    # el antigeno real-): el cache anterior (keyed solo por input_stem, sin
    # mirar el contenido real del FASTA/PDB) se reusaba en silencio pese a
    # que el input real habia cambiado de cadena, mezclando datos de una
    # cadena con resultados que decian ser de otra.
    call_count = {"n": 0}

    def _counting_run(self, items, output_dir=None):
        call_count["n"] += 1
        text = pathlib.Path(items[0]).read_text()
        return [pd.DataFrame({
            "Accession": ["acc"], "Residue": ["M"], "BepiPred-3.0 score": [float(len(text))],
        })]

    monkeypatch.setattr(BepiPredEngine, "run", _counting_run)

    fasta_v1 = tmp_path / "input.fasta"
    fasta_v1.write_text(">acc\nMKTAY\n")
    cache_path = tmp_path / "raw.csv"

    df1 = _cached_raw_scores("BepiPred-3.0", cache_path, tmp_path / "_raw", fasta_v1, BepiPredEngine())
    assert call_count["n"] == 1

    # Misma ruta de cache_path, pero el CONTENIDO del input cambio (simula
    # una re-corrida con distinta cadena/estrategia de seleccion bajo el
    # mismo input_stem): debe re-ejecutar, NO servir el cache viejo.
    fasta_v1.write_text(">acc\nMKTAYIAKQRQISFVKSHFSRQ\n")
    df2 = _cached_raw_scores("BepiPred-3.0", cache_path, tmp_path / "_raw", fasta_v1, BepiPredEngine())

    assert call_count["n"] == 2
    assert df1["BepiPred-3.0 score"].iloc[0] != df2["BepiPred-3.0 score"].iloc[0]

    # Sin cambios de contenido, la tercera llamada SI debe servir el cache.
    df3 = _cached_raw_scores("BepiPred-3.0", cache_path, tmp_path / "_raw", fasta_v1, BepiPredEngine())
    assert call_count["n"] == 2
    assert df3["BepiPred-3.0 score"].iloc[0] == df2["BepiPred-3.0 score"].iloc[0]
