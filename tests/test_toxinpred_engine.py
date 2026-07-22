"""Tests de la Fase 8a (src/engines/toxinpred_engine.py): workaround del bug de batch
de tamano 1 (el camino NORMAL en Fase 8, que evalua un unico constructo por corrida) y
propagacion de errores del subproceso. Mismo criterio que ``test_algpred_engine.py``:
mockea ``subprocess.run`` en vez de invocar el venv real.
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.toxinpred_engine import predict_toxicity
from src.utils.exceptions import EngineExecutionError


@pytest.fixture(autouse=True)
def _fake_binary(monkeypatch, tmp_path):
    fake_bin_dir = tmp_path / "fake_venv" / "bin"
    fake_bin_dir.mkdir(parents=True)
    (fake_bin_dir / "python").write_text("fake")
    (fake_bin_dir / Settings.TOXINPRED2_BINARY_NAME).write_text("fake")

    monkeypatch.setattr(Settings, "TOXINPRED2_PYTHON_BIN", str(fake_bin_dir / "python"))


def _mock_run_writing(raw_csv_rows):
    def _fake_run(cmd, **kwargs):
        out_path = cmd[cmd.index("-o") + 1]
        pd.DataFrame(raw_csv_rows).to_csv(out_path, index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _fake_run


def test_sequences_vacio_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_toxicity([], tmp_path)

    assert result.empty
    assert called == []


def test_batch_de_una_sola_secuencia_duplica_y_descarta_fila_extra(monkeypatch, tmp_path):
    # Bug verificado empiricamente: el modelo ONNX espera rank 2, con batch=1
    # el pipeline de features produce rank 1 y revienta -- workaround: duplicar.
    rows = [
        {"Sequence": "SOLA", "ML_Score": 0.24, "Prediction": "Non-Toxin"},
        {"Sequence": "SOLA", "ML_Score": 0.24, "Prediction": "Non-Toxin"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_toxicity(["SOLA"], tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["sequence"] == "SOLA"
    assert result.iloc[0]["toxinpred_veredicto"] == "Non-Toxin"


def test_batch_normal_no_duplica_secuencias(monkeypatch, tmp_path):
    rows = [
        {"Sequence": "AAAA", "ML_Score": 0.1, "Prediction": "Non-Toxin"},
        {"Sequence": "BBBB", "ML_Score": 0.9, "Prediction": "Toxin"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_toxicity(["AAAA", "BBBB"], tmp_path, filename_prefix="x_")

    assert len(result) == 2
    assert (tmp_path / "x_toxinpred_raw.csv").is_file()


def test_exit_code_distinto_de_cero_propaga_engine_execution_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="exit code 1"):
        predict_toxicity(["AAAA", "BBBB"], tmp_path)


def test_output_dir_relativo_se_resuelve_a_ruta_absoluta(monkeypatch, tmp_path):
    # Regresion real (2026-07-22, ver el mismo test en test_algpred_engine.py):
    # el subprocess de ToxinPred2 corre con 'cwd=tmp' (el directorio temporal
    # del batch), asi que un 'output_dir' relativo se resolveria contra ESE
    # directorio, no el de pipeline.py, si no se resolviera a absoluto antes.
    monkeypatch.chdir(tmp_path)
    rows = [{"Sequence": "AAAA", "ML_Score": 0.1, "Prediction": "Non-Toxin"},
            {"Sequence": "BBBB", "ML_Score": 0.9, "Prediction": "Toxin"}]
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out_path = cmd[cmd.index("-o") + 1]
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    predict_toxicity(["AAAA", "BBBB"], Path("relative_out_dir"))

    out_arg = captured["cmd"][captured["cmd"].index("-o") + 1]
    assert Path(out_arg).is_absolute()


def test_binario_ausente_lanza_error_accionable(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "TOXINPRED2_PYTHON_BIN", str(tmp_path / "no_existe"))

    with pytest.raises(EngineExecutionError, match="No se encontro el interprete"):
        predict_toxicity(["AAAA"], tmp_path)
