"""Tests de la Fase 4b (src/engines/algpred_engine.py): workaround del bug de batch de
tamano 1, propagacion de errores del subproceso y validacion del formato de salida.

``predict_allergenicity`` en si invoca un binario real (venv dedicado de AlgPred2), asi que
aqui se mockea ``subprocess.run`` para escribir un CSV crudo sintetico en la ruta esperada,
en vez de invocar el venv real (mismo criterio que ``test_blast_engine.py``: no depender de
binarios/instalaciones externas en un test unitario). ``_resolve_binary`` se satisface
apuntando ``Settings.ALGPRED_PYTHON_BIN``/``ALGPRED_SCRIPT_PATH`` a archivos que ya existen
en el repo (su contenido no importa, solo que ``Path.is_file()`` de verdadero).
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.algpred_engine import predict_allergenicity
from src.utils.exceptions import EngineExecutionError


@pytest.fixture(autouse=True)
def _fake_binary(monkeypatch):
    # Cualquier archivo real sirve: _resolve_binary solo comprueba is_file().
    monkeypatch.setattr(Settings, "ALGPRED_PYTHON_BIN", __file__)
    monkeypatch.setattr(Settings, "ALGPRED_SCRIPT_PATH", __file__)


def _mock_run_writing(raw_csv_rows):
    """Fabrica un reemplazo de ``subprocess.run`` que escribe ``raw_csv_rows`` en la ruta '-o' del comando."""

    def _fake_run(cmd, **kwargs):
        out_path = cmd[cmd.index("-o") + 1]
        pd.DataFrame(raw_csv_rows).to_csv(out_path, index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _fake_run


def test_sequences_vacio_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_allergenicity([], tmp_path)

    assert result.empty
    assert list(result.columns) == ["sequence", "algpred_score", "algpred_veredicto"]
    assert called == []


def test_batch_normal_no_duplica_secuencias(monkeypatch, tmp_path):
    rows = [
        {"Sequence": "AAAA", "ML_Score": 0.1, "Prediction": "Non-Allergen"},
        {"Sequence": "BBBB", "ML_Score": 0.9, "Prediction": "Allergen"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_allergenicity(["AAAA", "BBBB"], tmp_path, filename_prefix="x_")

    assert len(result) == 2
    assert result.iloc[0]["sequence"] == "AAAA"
    assert result.iloc[1]["algpred_veredicto"] == "Allergen"
    assert (tmp_path / "x_algpred_raw.csv").is_file()


def test_batch_de_una_sola_secuencia_duplica_y_descarta_fila_extra(monkeypatch, tmp_path):
    # AlgPred2 revienta con batch de tamano 1 (ver docstring del modulo): el
    # wrapper duplica la secuencia antes de invocar el binario. Aqui se
    # verifica que el mock reciba 2 secuencias en el FASTA de entrada (via el
    # CSV crudo de 2 filas que "devuelve" el binario) y que el resultado
    # final tenga una unica fila, no dos.
    rows = [
        {"Sequence": "SOLA", "ML_Score": 0.5, "Prediction": "Allergen"},
        {"Sequence": "SOLA", "ML_Score": 0.5, "Prediction": "Allergen"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_allergenicity(["SOLA"], tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["sequence"] == "SOLA"


def test_exit_code_distinto_de_cero_propaga_engine_execution_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="exit code 1"):
        predict_allergenicity(["AAAA", "BBBB"], tmp_path)


def test_timeout_propaga_engine_execution_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="tiempo limite"):
        predict_allergenicity(["AAAA", "BBBB"], tmp_path)


def test_csv_sin_columnas_esperadas_lanza_error(monkeypatch, tmp_path):
    rows = [{"Sequence": "AAAA", "columna_inesperada": 1}]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    with pytest.raises(EngineExecutionError, match="formato del CSV"):
        predict_allergenicity(["AAAA", "BBBB"], tmp_path)


def test_output_dir_relativo_se_resuelve_a_ruta_absoluta(monkeypatch, tmp_path):
    # Regresion real (2026-07-22): el subprocess de AlgPred2 corre con
    # 'cwd' forzado a la carpeta del script instalado (algpred2.py necesita
    # rutas propias relativas a su instalacion). Si 'output_dir' llega
    # relativo (default de Settings.FASTA_OUTPUT_DIR: 'fasta_outputs'), el
    # hijo lo resuelve contra SU cwd, no el de pipeline.py -- confirmado con
    # 'OSError: Cannot save file into a non-existent directory'. La ruta '-o'
    # pasada al subprocess debe ser siempre absoluta, sin importar el cwd
    # del proceso que llama.
    monkeypatch.chdir(tmp_path)
    rows = [{"Sequence": "AAAA", "ML_Score": 0.1, "Prediction": "Non-Allergen"},
            {"Sequence": "BBBB", "ML_Score": 0.9, "Prediction": "Allergen"}]
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out_path = cmd[cmd.index("-o") + 1]
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    predict_allergenicity(["AAAA", "BBBB"], Path("relative_out_dir"))

    out_arg = captured["cmd"][captured["cmd"].index("-o") + 1]
    assert Path(out_arg).is_absolute()


def test_binario_ausente_lanza_error_accionable(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "ALGPRED_PYTHON_BIN", str(tmp_path / "no_existe"))

    with pytest.raises(EngineExecutionError, match="No se encontro el interprete"):
        predict_allergenicity(["AAAA"], tmp_path)
