"""Tests de la Fase 8b (src/engines/iapred_engine.py): validacion de formato de salida
y propagacion de errores del subproceso. Mockea ``subprocess.run`` (mismo criterio que
``test_algpred_engine.py``): IApred en si no tiene el bug de batch=1 que si tienen
AlgPred2/ToxinPred2, asi que no hace falta probar ese caso aqui.
"""

import subprocess

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.iapred_engine import predict_intrinsic_antigenicity
from src.utils.exceptions import EngineExecutionError


@pytest.fixture(autouse=True)
def _fake_binary(monkeypatch, tmp_path):
    home = tmp_path / "IApred"
    (home / "models").mkdir(parents=True)
    (home / Settings.IAPRED_SCRIPT_NAME).write_text("fake")

    monkeypatch.setattr(Settings, "IAPRED_PYTHON_BIN", __file__)
    monkeypatch.setattr(Settings, "IAPRED_HOME", str(home))


def _mock_run_writing(raw_csv_rows):
    def _fake_run(cmd, **kwargs):
        out_path = cmd[3]  # [python, script, fasta, output_csv, "-v"]
        pd.DataFrame(raw_csv_rows).to_csv(out_path, index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _fake_run


def test_sequences_vacio_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_intrinsic_antigenicity([], tmp_path)

    assert result.empty
    assert called == []


def test_secuencia_menor_a_20aa_no_rompe_y_devuelve_nan_no_string(monkeypatch, tmp_path):
    # Bug real verificado empiricamente (IApred.py linea ~116): para
    # secuencias de MENOS de 20 aa, IApred escribe el texto literal
    # 'Sequence too short' en la columna de score (no un numero) y 'N/A' en
    # la de categoria (que pandas ya lee como NaN, token de NA por defecto).
    # Sin coercion, un consumidor que asuma 'iapred_score' siempre numerico
    # (print_iapred_report formatea con ':.4f') revienta con TypeError.
    rows = [
        {"Header": "candidato_0", "Sequence_Length": 3, "Intrinsic_Antigenicity_Score": "Sequence too short", "Antigenicity_Category": "N/A"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_intrinsic_antigenicity(["ACD"], tmp_path)

    assert pd.isna(result.iloc[0]["iapred_score"])
    assert pd.api.types.is_float_dtype(result["iapred_score"])  # nunca deja un string mezclado en la columna
    assert result.iloc[0]["iapred_categoria"] == "No evaluado (secuencia < 20 aa)"


def test_batch_mixto_corto_y_normal(monkeypatch, tmp_path):
    rows = [
        {"Header": "candidato_0", "Sequence_Length": 3, "Intrinsic_Antigenicity_Score": "Sequence too short", "Antigenicity_Category": "N/A"},
        {"Header": "candidato_1", "Sequence_Length": 30, "Intrinsic_Antigenicity_Score": -1.02, "Antigenicity_Category": "Low"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_intrinsic_antigenicity(["ACD", "M" * 30], tmp_path)

    assert pd.isna(result.iloc[0]["iapred_score"])
    assert result.iloc[1]["iapred_score"] == pytest.approx(-1.02)
    assert result.iloc[1]["iapred_categoria"] == "Low"


def test_batch_normal(monkeypatch, tmp_path):
    rows = [
        {"Header": "candidato_0", "Sequence_Length": 4, "Intrinsic_Antigenicity_Score": 0.12, "Antigenicity_Category": "Moderate"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_intrinsic_antigenicity(["AAAA"], tmp_path, filename_prefix="x_")

    assert len(result) == 1
    assert result.iloc[0]["sequence"] == "AAAA"
    assert result.iloc[0]["iapred_score"] == 0.12
    assert result.iloc[0]["iapred_categoria"] == "Moderate"
    assert (tmp_path / "x_iapred_raw.csv").is_file()


def test_numero_de_filas_no_coincide_lanza_error(monkeypatch, tmp_path):
    rows = [
        {"Header": "candidato_0", "Sequence_Length": 4, "Intrinsic_Antigenicity_Score": 0.12, "Antigenicity_Category": "Moderate"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    with pytest.raises(EngineExecutionError, match="se esperaban 2"):
        predict_intrinsic_antigenicity(["AAAA", "BBBB"], tmp_path)


def test_csv_sin_columnas_esperadas_lanza_error(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([{"columna_rara": 1}]))

    with pytest.raises(EngineExecutionError, match="formato del CSV"):
        predict_intrinsic_antigenicity(["AAAA"], tmp_path)


def test_exit_code_distinto_de_cero_propaga_engine_execution_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="exit code 1"):
        predict_intrinsic_antigenicity(["AAAA"], tmp_path)


def test_carpeta_models_ausente_lanza_error_accionable(monkeypatch, tmp_path):
    home = tmp_path / "IApred_sin_models"
    home.mkdir()
    (home / Settings.IAPRED_SCRIPT_NAME).write_text("fake")
    monkeypatch.setattr(Settings, "IAPRED_HOME", str(home))

    with pytest.raises(EngineExecutionError, match="models"):
        predict_intrinsic_antigenicity(["AAAA"], tmp_path)


def test_print_report_no_revienta_con_score_nan():
    # Regresion directa del bug: antes de la coercion a NaN, esta llamada
    # reventaba con TypeError (f"{'Sequence too short':.4f}" no es valido).
    from src.engines.iapred_engine import print_iapred_report

    df = pd.DataFrame([
        {"sequence": "ACD", "iapred_score": float("nan"), "iapred_categoria": "No evaluado (secuencia < 20 aa)"},
    ])
    print_iapred_report(df)  # no debe lanzar ninguna excepcion
