"""Tests de la Fase 3b (src/engines/tmbed_engine.py): parseo del formato de 3 lineas
por proteina de 'tmbed predict --out-format 1', colapsado de clases en regiones de
enmascarado, filtro de solapamiento contra la union anotada, y propagacion de errores
del subproceso.
"""

import subprocess

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.tmbed_engine import filter_overlapping_regions, predict_tm_signal_regions
from src.utils.exceptions import EngineExecutionError


@pytest.fixture(autouse=True)
def _fake_binary(monkeypatch, tmp_path):
    fake_bin_dir = tmp_path / "fake_venv" / "bin"
    fake_bin_dir.mkdir(parents=True)
    (fake_bin_dir / "python").write_text("fake")
    (fake_bin_dir / Settings.TMBED_BINARY_NAME).write_text("fake")

    model_dir = tmp_path / "tmbed_models" / "t5"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    monkeypatch.setattr(Settings, "TMBED_PYTHON_BIN", str(fake_bin_dir / "python"))
    monkeypatch.setattr(Settings, "TMBED_MODEL_DIR", str(model_dir))


def _mock_run_writing(records):
    """records: lista de tuplas (header, sequence, classes)."""

    def _fake_run(cmd, **kwargs):
        pred_path = cmd[cmd.index("--predictions") + 1]
        with open(pred_path, "w") as f:
            for header, sequence, classes in records:
                f.write(f">{header}\n{sequence}\n{classes}\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _fake_run


def test_sequences_vacio_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_tm_signal_regions({}, tmp_path)

    assert result.empty
    assert called == []


def test_sin_ninguna_region_devuelve_dataframe_vacio(monkeypatch, tmp_path):
    # Proteina toda 'i' (no-membrana, adentro): no hay nada que enmascarar.
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([("acc1", "MKTAY", "iiiii")]))

    result = predict_tm_signal_regions({"acc1": "MKTAY"}, tmp_path)

    assert result.empty


def test_colapsa_residuos_consecutivos_de_la_misma_clase_en_una_region(monkeypatch, tmp_path):
    # 'S' (senal) en 1-3, 'i' en 4-5, 'H' (helice TM) en 6-9.
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([("acc1", "MKTAYIAKQ", "SSSiiHHHH")]))

    result = predict_tm_signal_regions({"acc1": "MKTAYIAKQ"}, tmp_path)

    rows = {(r.accession, r.start, r.end, r.type) for r in result.itertuples(index=False)}
    assert rows == {
        ("acc1", 1, 3, "signal_peptide"),
        ("acc1", 6, 9, "TM_alpha_helix"),
    }


def test_multiples_accessions_se_reportan_por_separado(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess,
        "run",
        _mock_run_writing([("acc1", "MKTAY", "BBBBB"), ("acc2", "IAKQR", "ooooo")]),
    )

    result = predict_tm_signal_regions({"acc1": "MKTAY", "acc2": "IAKQR"}, tmp_path)

    assert set(result["accession"]) == {"acc1"}
    assert result.iloc[0]["type"] == "TM_beta_strand"


def test_region_minima_descarta_regiones_cortas(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "TMBED_MIN_REGION_LENGTH", 5)
    # Region 'H' de solo 3 residuos (< 5): se descarta.
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([("acc1", "MKTAYIAKQ", "iiHHHiiii")]))

    result = predict_tm_signal_regions({"acc1": "MKTAYIAKQ"}, tmp_path)

    assert result.empty


def test_desfase_secuencia_prediccion_lanza_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        pred_path = cmd[cmd.index("--predictions") + 1]
        with open(pred_path, "w") as f:
            f.write(">acc1\nMKTAY\nSSS\n")  # 5 residuos, 3 letras de clase
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="Desfase secuencia/prediccion"):
        predict_tm_signal_regions({"acc1": "MKTAY"}, tmp_path)


def test_exit_code_distinto_de_cero_propaga_engine_execution_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="exit code 1"):
        predict_tm_signal_regions({"acc1": "MKTAY"}, tmp_path)


def test_pesos_ausentes_lanza_error_accionable(monkeypatch, tmp_path):
    empty_model_dir = tmp_path / "sin_pesos"
    empty_model_dir.mkdir()
    monkeypatch.setattr(Settings, "TMBED_MODEL_DIR", str(empty_model_dir))

    with pytest.raises(EngineExecutionError, match="config.json"):
        predict_tm_signal_regions({"acc1": "MKTAY"}, tmp_path)


def test_filter_overlapping_regions_descarta_solo_filas_solapadas():
    union_df = pd.DataFrame({
        "accession": ["acc1", "acc1", "acc2"],
        "start": [1, 20, 5],
        "end": [9, 28, 13],
        "sequence": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
    })
    regions_df = pd.DataFrame({
        "accession": ["acc1"],
        "start": [5],
        "end": [15],
        "type": ["TM_alpha_helix"],
    })

    kept, n_discarded = filter_overlapping_regions(union_df, regions_df)

    assert n_discarded == 1
    assert list(kept["accession"]) == ["acc1", "acc2"]
    assert list(kept["start"]) == [20, 5]


def test_filter_overlapping_regions_sin_regiones_no_descarta_nada():
    union_df = pd.DataFrame({"accession": ["acc1"], "start": [1], "end": [9], "sequence": ["AAAAAAAAA"]})

    kept, n_discarded = filter_overlapping_regions(union_df, pd.DataFrame(columns=["accession", "start", "end", "type"]))

    assert n_discarded == 0
    assert kept.equals(union_df)
