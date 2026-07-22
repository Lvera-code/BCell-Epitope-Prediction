"""Tests de la Fase 4c (src/engines/stackglyembed_engine.py): scanner de secuones
N-X-[S/T] propio (el repo original de StackGlyEmbed no trae uno) y el wrapper de
subprocess que lo conecta con ``stackglyembed_predict_local.py``.

``scan_sequons`` es logica pura (regex), se prueba directamente. El resto (``predict_nglycosylation``)
invoca un binario real (venv dedicado con ProteinBERT/ESM-2/ProtT5), asi que se mockea
``subprocess.run`` para escribir un ``predicted_values.csv`` sintetico en la carpeta
``--output-dir`` que el propio wrapper crea, en vez de correr los modelos reales.
"""

import subprocess

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.stackglyembed_engine import predict_nglycosylation, scan_sequons
from src.utils.exceptions import EngineExecutionError


# --- scan_sequons: regla N-X-[S/T], X != Prolina, con solapamiento ----------------------


def test_sin_sequon_devuelve_lista_vacia():
    assert scan_sequons("AAAAAAAAAA") == []


def test_sequon_simple_reporta_posicion_1_indexada_de_la_n():
    # 'NQS' en el offset 6 (0-idx) -> posicion 1-indexada 7.
    assert scan_sequons("ACDEFGNQSHIKLMNPQRST") == [7]


def test_prolina_como_x_excluye_el_sequon():
    # 'NPS': X=Prolina, no es un secuon valido.
    assert scan_sequons("AAANPSAAA") == []


def test_tercer_residuo_debe_ser_serina_o_treonina():
    assert scan_sequons("AAANAGAAA") == []  # 'NAG', tercer residuo 'G' invalido
    assert scan_sequons("AAANASAAA") != []  # 'NAS', valido
    assert scan_sequons("AAANATAAA") != []  # 'NAT', valido


def test_sequones_solapados_se_reportan_todos():
    # 'NNSNST': N(0) N S -> valido (pos1); N(1) S N -> invalido (3er char no S/T);
    # N(3) S T -> valido (pos4). Ver docstring de _SEQUON_PATTERN para el porque
    # de la regex de lookahead (un finditer plano se saltaria el segundo).
    assert scan_sequons("NNSNST") == [1, 4]


def test_sequon_al_final_de_la_secuencia_sin_espacio_no_matchea():
    # Falta el 3er residuo del secuon: no debe intentar leer fuera de rango.
    assert scan_sequons("AAAAANS") == []


# --- predict_nglycosylation: wrapper de subprocess ---------------------------------------


@pytest.fixture(autouse=True)
def _fake_binary(monkeypatch, tmp_path):
    models_dir = tmp_path / "fake_models"
    (models_dir / "base_layer_pickle_files").mkdir(parents=True)
    (models_dir / "base_layer_pickle_files" / "SVM_meta_layer.sav").write_text("fake")

    monkeypatch.setattr(Settings, "STACKGLYEMBED_PYTHON_BIN", __file__)
    monkeypatch.setattr(Settings, "STACKGLYEMBED_SCRIPT_PATH", __file__)
    monkeypatch.setattr(Settings, "STACKGLYEMBED_MODELS_DIR", str(models_dir))


def _mock_run_writing(predictions):
    """Fabrica un reemplazo de ``subprocess.run`` que escribe ``predictions`` en '--output-dir'."""

    def _fake_run(cmd, **kwargs):
        out_dir = cmd[cmd.index("--output-dir") + 1]
        import os

        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(predictions).to_csv(f"{out_dir}/predicted_values.csv", index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _fake_run


def test_sequences_vacio_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_nglycosylation([], tmp_path)

    assert result.empty
    assert called == []


def test_peptidos_sin_ningun_sequon_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_nglycosylation(["AAAAAAAAAA", "CCCCCCCCCC"], tmp_path)

    assert result.empty
    assert called == []


def test_un_peptido_un_sequon(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([{"prediction": 1, "probability": 0.87}]))

    result = predict_nglycosylation(["ACDEFGNQSHIKLMNPQRST"], tmp_path, filename_prefix="x_")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["sequence"] == "ACDEFGNQSHIKLMNPQRST"
    assert row["sequon_position"] == 7
    assert row["stackglyembed_veredicto"] == "Glicosilado"
    assert row["stackglyembed_score"] == pytest.approx(0.87)
    assert (tmp_path / "x_stackglyembed_raw.csv").is_file()


def test_peptido_con_multiples_sequones_produce_una_fila_por_sitio(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, "run",
        _mock_run_writing([{"prediction": 0, "probability": 0.1}, {"prediction": 1, "probability": 0.9}]),
    )

    result = predict_nglycosylation(["NNSNSTAAAAAAAAAAAAAA"], tmp_path)

    assert len(result) == 2
    assert list(result["sequon_position"]) == [1, 4]
    assert list(result["stackglyembed_veredicto"]) == ["No glicosilado", "Glicosilado"]


def test_mezcla_de_peptidos_con_y_sin_sequon_omite_los_sin_sequon(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([{"prediction": 0, "probability": 0.2}]))

    result = predict_nglycosylation(["AAAAAAAAAA", "ACDEFGNQSHIKLMNPQRST"], tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["sequence"] == "ACDEFGNQSHIKLMNPQRST"


def test_numero_de_predicciones_no_coincide_con_sitios_lanza_error(monkeypatch, tmp_path):
    # El script devolvio 1 prediccion pero se esperaban 2 (2 sequones en el peptido).
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([{"prediction": 1, "probability": 0.5}]))

    with pytest.raises(EngineExecutionError, match="se esperaban 2"):
        predict_nglycosylation(["NNSNSTAAAAAAAAAAAAAA"], tmp_path)


def test_exit_code_distinto_de_cero_propaga_engine_execution_error(monkeypatch, tmp_path):
    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(EngineExecutionError, match="exit code 1"):
        predict_nglycosylation(["ACDEFGNQSHIKLMNPQRST"], tmp_path)


def test_pickles_del_clasificador_ausentes_lanza_error_accionable(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "STACKGLYEMBED_MODELS_DIR", str(tmp_path / "no_existe"))

    with pytest.raises(EngineExecutionError, match="pickles del clasificador"):
        predict_nglycosylation(["ACDEFGNQSHIKLMNPQRST"], tmp_path)
