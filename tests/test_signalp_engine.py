"""Tests de la Fase 8d (src/engines/signalp_engine.py): parseo de 'prediction_results.txt'
(bug real verificado: 2 lineas de comentario '#', no 1 -- un 'skiprows' fijo leia la
segunda como fila de datos) y propagacion de errores del subproceso.
"""

import subprocess

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.signalp_engine import predict_signal_peptide
from src.utils.exceptions import EngineExecutionError

# Formato real verificado empiricamente (2 lineas de comentario, no 1).
_REAL_HEADER = (
    "# SignalP-6.0\tOrganism: Other\tTimestamp: 20260722123855\n"
    "# ID\tPrediction\tOTHER\tSP(Sec/SPI)\tLIPO(Sec/SPII)\tTAT(Tat/SPI)\tTATLIPO(Tat/SPII)\tPILIN(Sec/SPIII)\tCS Position\n"
)


@pytest.fixture(autouse=True)
def _fake_binary(monkeypatch, tmp_path):
    fake_bin_dir = tmp_path / "fake_venv" / "bin"
    fake_bin_dir.mkdir(parents=True)
    (fake_bin_dir / "python").write_text("fake")
    (fake_bin_dir / Settings.SIGNALP_BINARY_NAME).write_text("fake")

    model_dir = tmp_path / "signalp-6.0" / "models"
    (model_dir / "sequential_models_signalp6").mkdir(parents=True)

    monkeypatch.setattr(Settings, "SIGNALP_PYTHON_BIN", str(fake_bin_dir / "python"))
    monkeypatch.setattr(Settings, "SIGNALP_MODEL_DIR", str(model_dir))


def _mock_run_writing(data_lines):
    def _fake_run(cmd, **kwargs):
        out_dir = cmd[cmd.index("--output_dir") + 1]
        import os

        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/prediction_results.txt", "w") as f:
            f.write(_REAL_HEADER)
            f.write("\n".join(data_lines) + "\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _fake_run


def test_sequences_vacio_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_signal_peptide([], tmp_path)

    assert result.empty
    assert called == []


def test_sin_peptido_senal_no_lee_la_segunda_linea_de_comentario_como_dato(monkeypatch, tmp_path):
    # Regresion del bug real: con skiprows=1 fijo, esta fila se leia con
    # 'Prediction' == 'Prediction' (texto de la 2da cabecera), no 'OTHER'.
    line = "candidato_0\tOTHER\t1.000000\t0.000000\t0.000000\t0.000000\t0.000000\t0.000000\t"
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([line]))

    result = predict_signal_peptide(["MKTAYIAKQRQ"], tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["signalp_prediction"] == "OTHER"
    assert result.iloc[0]["signalp_prob_other"] == pytest.approx(1.0)


def test_con_peptido_senal_detectado(monkeypatch, tmp_path):
    line = "candidato_0\tSP\t0.000161\t0.999315\t0.000128\t0.000149\t0.000121\t0.000118\tCS pos: 24-25. Pr: 0.9771"
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([line]))

    result = predict_signal_peptide(["MALWMRLLPLL"], tmp_path)

    assert result.iloc[0]["signalp_prediction"] == "SP"
    assert result.iloc[0]["signalp_prob_sp"] == pytest.approx(0.999315)
    assert "CS pos" in result.iloc[0]["signalp_cs_position"]


def test_numero_de_filas_no_coincide_lanza_error(monkeypatch, tmp_path):
    line = "candidato_0\tOTHER\t1.000000\t0.000000\t0.000000\t0.000000\t0.000000\t0.000000\t"
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([line]))

    with pytest.raises(EngineExecutionError, match="se esperaban 2"):
        predict_signal_peptide(["AAAA", "BBBB"], tmp_path)


def test_exit_code_distinto_de_cero_propaga_engine_execution_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="exit code 1"):
        predict_signal_peptide(["AAAA"], tmp_path)


def test_pesos_ausentes_lanza_error_accionable(monkeypatch, tmp_path):
    empty_model_dir = tmp_path / "sin_pesos"
    empty_model_dir.mkdir()
    monkeypatch.setattr(Settings, "SIGNALP_MODEL_DIR", str(empty_model_dir))

    with pytest.raises(EngineExecutionError, match="sequential_models_signalp6"):
        predict_signal_peptide(["AAAA"], tmp_path)
