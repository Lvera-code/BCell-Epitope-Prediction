"""Tests de src/engines/scannet_engine.py: construccion pura del comando por runtime.

Regresion (2026-07-20, hallado corriendo ./run.sh sin variables de entorno
exportadas): el runtime 'venv' pasaba 'predict_bindingsites.py' PRECEDIDO de
install_path (p. ej. 'ScanNet/predict_bindingsites.py') Y ADEMAS fijaba
cwd=install_path en el subprocess -- la combinacion resolvia el argumento
relativo al NUEVO cwd, buscando 'ScanNet/ScanNet/predict_bindingsites.py'
(exit code 2, 'No such file or directory'). El runtime 'docker' nunca tuvo
este bug porque ya usaba el nombre pelado (el WORKDIR del contenedor cumple
el mismo rol que cwd en el runtime 'venv').
"""

from pathlib import Path

import pandas as pd

from src.engines.scannet_engine import (
    ACCESSION_COLUMN,
    RESIDUE_COLUMN,
    SCORE_COLUMN,
    ScanNetEngine,
    extract_epitopes,
)


def test_venv_runtime_no_antepone_install_path_al_script():
    engine = ScanNetEngine(runtime="venv", install_path=Path("ScanNet"), python_bin="/fake/python")

    cmd = engine._build_command(Path("/tmp/chain_A.pdb"), Path("/tmp/out"), "acc1")

    assert "predict_bindingsites.py" in cmd
    assert "ScanNet/predict_bindingsites.py" not in cmd
    # El script debe pasarse pelado (relativo al cwd que fija _run_single),
    # nunca con install_path como prefijo.
    script_idx = cmd.index("predict_bindingsites.py")
    assert cmd[script_idx] == "predict_bindingsites.py"


def test_docker_runtime_tambien_usa_script_pelado():
    engine = ScanNetEngine(runtime="docker", install_path=Path("ScanNet"), docker_workdir="/ScanNet")

    cmd = engine._build_command(Path("/tmp/chain_A.pdb"), Path("/tmp/out"), "acc1")

    assert "predict_bindingsites.py" in cmd
    assert "ScanNet/predict_bindingsites.py" not in cmd


def test_venv_runtime_incluye_flags_esperados():
    engine = ScanNetEngine(runtime="venv", install_path=Path("ScanNet"), python_bin="/fake/python")

    cmd = engine._build_command(Path("/tmp/chain_A.pdb"), Path("/tmp/out"), "acc1")

    assert "--mode" in cmd and "epitope" in cmd
    assert "--noMSA" in cmd
    assert "--pdb" in cmd
    assert "--name" in cmd and "acc1" in cmd
    assert "--predictions_folder" in cmd


# --- extract_epitopes: umbral adaptativo por percentil (default) vs fijo -------------

def _raw_df(accession, scores):
    return pd.DataFrame({
        ACCESSION_COLUMN: [accession] * len(scores),
        RESIDUE_COLUMN: ["M"] * len(scores),
        SCORE_COLUMN: scores,
    })


def test_threshold_none_usa_percentil_adaptativo_por_accession():
    # Cadena de baja senal general (max=0.29, como el ejemplo real): un
    # umbral fijo tipico (p. ej. 0.5) nunca encontraria nada, pero el
    # percentil 90 SI debe encontrar la zona alta real de esta cadena.
    scores = [0.03, 0.03, 0.03, 0.25, 0.26, 0.27, 0.28, 0.29, 0.28, 0.03, 0.03, 0.03]
    df = _raw_df("ACC1", scores)

    result = extract_epitopes(df, threshold=None, min_length=3, window_size=3, max_gap_residues=0)

    assert not result.empty
    assert result.iloc[0]["accession"] == "ACC1"


def test_threshold_fijo_bypassa_el_modo_adaptativo():
    scores = [0.03] * 12
    df = _raw_df("ACC1", scores)

    # Con un umbral fijo bajo, hasta una cadena "plana" y de baja senal
    # produce una region (todo el mundo supera 0.01).
    result = extract_epitopes(df, threshold=0.01, min_length=3, window_size=3, max_gap_residues=0)

    assert not result.empty


def test_cada_accession_calcula_su_propio_umbral_adaptativo():
    # ACC1: senal alta en general (percentil 90 ~ alto en escala absoluta).
    # ACC2: senal baja en general (percentil 90 ~ bajo en escala absoluta).
    # Ambas deben encontrar SU propia zona relativamente alta, pese a que
    # las escalas absolutas de las dos cadenas son muy distintas.
    acc1 = _raw_df("ACC1", [0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5])
    acc2 = _raw_df("ACC2", [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
    df = pd.concat([acc1, acc2], ignore_index=True)

    result = extract_epitopes(df, threshold=None, min_length=3, window_size=3, max_gap_residues=0)

    accessions_found = set(result["accession"])
    assert accessions_found == {"ACC1", "ACC2"}


def test_dataframe_vacio_no_rompe_modo_adaptativo():
    df = pd.DataFrame(columns=[ACCESSION_COLUMN, RESIDUE_COLUMN, SCORE_COLUMN])

    result = extract_epitopes(df, threshold=None, min_length=9, window_size=9, max_gap_residues=2)

    assert result.empty
