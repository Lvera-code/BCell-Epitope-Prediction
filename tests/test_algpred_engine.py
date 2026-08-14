"""Tests de la Fase 4b (src/engines/algpred_engine.py): workaround del bug de batch de
tamano 1, propagacion de errores del subproceso y validacion del formato de salida.

``predict_allergenicity`` en si invoca un binario real (venv dedicado de AlgPred2), asi que
aqui se mockea ``subprocess.run`` para escribir un CSV crudo sintetico en la ruta esperada,
en vez de invocar el venv real (mismo criterio que ``test_blast_engine.py``: no depender de
binarios/instalaciones externas en un test unitario). ``_resolve_binary`` se satisface
apuntando ``Settings.ALGPRED_PYTHON_BIN``/``ALGPRED_SCRIPT_PATH`` a archivos que ya existen
en el repo (su contenido no importa, solo que ``Path.is_file()`` de verdadero).

``Settings.ALGPRED_MODEL`` por defecto es 2 (hibrido) desde 2026-08-14, asi que las filas
sinteticas de la mayoria de estos tests usan el esquema de salida del modelo 2
(``Subject,ML Score,MERCI Score,BLAST Score,Hybrid Score,Prediction``, sin columna
``Sequence`` -- la secuencia se reconstruye por posicion). Hay tests dedicados para el
esquema del modelo 1 (``ID,Sequence,ML_Score,Prediction``) pasando ``model=1`` explicito.
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.algpred_engine import predict_allergenicity
from src.utils.exceptions import EngineExecutionError

_OUTPUT_COLUMNS = [
    "sequence", "algpred_score", "algpred_veredicto",
    "algpred_ml_score", "algpred_merci_score", "algpred_blast_score",
]


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


def _hybrid_rows(subjects, ml_scores, merci_scores, blast_scores, predictions):
    hybrid_scores = [ml + me + bl for ml, me, bl in zip(ml_scores, merci_scores, blast_scores)]
    return [
        {
            "Subject": s, "ML Score": ml, "MERCI Score": me, "BLAST Score": bl,
            "Hybrid Score": hy, "Prediction": p,
        }
        for s, ml, me, bl, hy, p in zip(subjects, ml_scores, merci_scores, blast_scores, hybrid_scores, predictions)
    ]


def test_sequences_vacio_no_invoca_subprocess(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = predict_allergenicity([], tmp_path)

    assert result.empty
    assert list(result.columns) == _OUTPUT_COLUMNS
    assert called == []


def test_batch_normal_no_duplica_secuencias(monkeypatch, tmp_path):
    rows = _hybrid_rows(
        subjects=["candidato_0", "candidato_1"],
        ml_scores=[0.1, 0.35], merci_scores=[0.0, 0.0], blast_scores=[0.0, 0.5],
        predictions=["Non-Allergen", "Allergen"],
    )
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_allergenicity(["AAAA", "BBBB"], tmp_path, filename_prefix="x_")

    assert len(result) == 2
    assert result.iloc[0]["sequence"] == "AAAA"
    assert result.iloc[1]["sequence"] == "BBBB"
    assert result.iloc[1]["algpred_veredicto"] == "Allergen"
    assert result.iloc[1]["algpred_score"] == pytest.approx(0.85)  # Hybrid Score = ML+MERCI+BLAST
    assert result.iloc[1]["algpred_blast_score"] == pytest.approx(0.5)
    assert (tmp_path / "x_algpred_raw.csv").is_file()


def test_batch_de_una_sola_secuencia_duplica_y_descarta_fila_extra(monkeypatch, tmp_path):
    # AlgPred2 revienta con batch de tamano 1 (ver docstring del modulo): el
    # wrapper duplica la secuencia antes de invocar el binario. Aqui se
    # verifica que el resultado final tenga una unica fila, no dos, y que la
    # secuencia (reconstruida por posicion, modelo hibrido no trae 'Sequence')
    # siga siendo la correcta.
    rows = _hybrid_rows(
        subjects=["candidato_0", "candidato_1"],
        ml_scores=[0.5, 0.5], merci_scores=[0.0, 0.0], blast_scores=[0.0, 0.0],
        predictions=["Allergen", "Allergen"],
    )
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_allergenicity(["SOLA"], tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["sequence"] == "SOLA"


def test_modelo_hibrido_expone_desglose_de_evidencia(monkeypatch, tmp_path):
    # Caso real que motivo el cambio de default (ver investigacion 6B5M,
    # 2026-08-14): un candidato "Allergen" con evidencia BLAST/MERCI real
    # debe distinguirse de uno que solo lo es por el RF de composicion.
    rows = _hybrid_rows(
        subjects=["candidato_0", "candidato_1"],
        ml_scores=[0.371, 0.321], merci_scores=[0.0, 0.0], blast_scores=[0.0, 0.5],
        predictions=["Allergen", "Allergen"],
    )
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_allergenicity(["NPDPNANPNVDPN", "GTHNGQIGNDPNRDILIASNWYFNHLKDKTL"], tmp_path)

    sin_evidencia = result.iloc[0]
    con_evidencia = result.iloc[1]
    assert sin_evidencia["algpred_merci_score"] == 0.0 and sin_evidencia["algpred_blast_score"] == 0.0
    assert con_evidencia["algpred_blast_score"] == pytest.approx(0.5)
    assert con_evidencia["algpred_score"] == pytest.approx(0.821)


def test_modelo_hibrido_filas_no_coinciden_con_entrada_lanza_error(monkeypatch, tmp_path):
    # Salvaguarda: el modelo hibrido no trae columna 'Sequence', asi que la
    # reconstruccion por posicion exige que el numero de filas coincida
    # exactamente con las secuencias de entrada -- si no, mejor fallar fuerte
    # que asignar secuencias a la fila equivocada en silencio.
    rows = _hybrid_rows(
        subjects=["candidato_0"], ml_scores=[0.5], merci_scores=[0.0], blast_scores=[0.0],
        predictions=["Allergen"],
    )
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    with pytest.raises(EngineExecutionError, match="no se puede reconstruir la secuencia"):
        predict_allergenicity(["AAAA", "BBBB"], tmp_path)


def test_modelo1_explicito_usa_esquema_legado(monkeypatch, tmp_path):
    rows = [
        {"ID": "candidato_0", "Sequence": "AAAA", "ML_Score": 0.1, "Prediction": "Non-Allergen"},
        {"ID": "candidato_1", "Sequence": "BBBB", "ML_Score": 0.9, "Prediction": "Allergen"},
    ]
    monkeypatch.setattr(subprocess, "run", _mock_run_writing(rows))

    result = predict_allergenicity(["AAAA", "BBBB"], tmp_path, model=1)

    assert len(result) == 2
    assert result.iloc[1]["algpred_score"] == pytest.approx(0.9)
    assert result.iloc[1]["algpred_veredicto"] == "Allergen"
    # Modelo 1 no calcula MERCI/BLAST: siempre 0, nunca evidencia real.
    assert result.iloc[1]["algpred_merci_score"] == 0.0
    assert result.iloc[1]["algpred_blast_score"] == 0.0


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
    # Regresion real: el subprocess de AlgPred2 corre con
    # 'cwd' forzado a la carpeta del script instalado (algpred2.py necesita
    # rutas propias relativas a su instalacion). Si 'output_dir' llega
    # relativo (default de Settings.FASTA_OUTPUT_DIR: 'outputs'), el
    # hijo lo resuelve contra SU cwd, no el de pipeline.py -- confirmado con
    # 'OSError: Cannot save file into a non-existent directory'. La ruta '-o'
    # pasada al subprocess debe ser siempre absoluta, sin importar el cwd
    # del proceso que llama.
    monkeypatch.chdir(tmp_path)
    rows = _hybrid_rows(
        subjects=["candidato_0", "candidato_1"],
        ml_scores=[0.1, 0.9], merci_scores=[0.0, 0.0], blast_scores=[0.0, 0.0],
        predictions=["Non-Allergen", "Allergen"],
    )
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
    assert "-m" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-m") + 1] == "2"


def test_binario_ausente_lanza_error_accionable(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "ALGPRED_PYTHON_BIN", str(tmp_path / "no_existe"))

    with pytest.raises(EngineExecutionError, match="No se encontro el interprete"):
        predict_allergenicity(["AAAA"], tmp_path)
