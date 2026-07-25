"""Tests de la Fase 3b (src/engines/tmbed_engine.py): parseo del formato de 3 lineas
por proteina de 'tmbed predict --out-format 1', colapsado de clases (incluida 'i',
intracelular) en regiones de enmascarado, filtro de solapamiento contra la union
anotada -- incluyendo los casos de validacion PSMD7 (control negativo, proteina
100% citoplasmatica que debe quedar excluida por completo) y THBS2 (control
positivo, secretada, que debe pasar integra) --, y propagacion de errores del
subproceso.
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
    # Proteina toda 'o' (no-membrana, afuera): nada que enmascarar.
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([("acc1", "MKTAY", "ooooo")]))

    result = predict_tm_signal_regions({"acc1": "MKTAY"}, tmp_path)

    assert result.empty


def test_colapsa_residuos_consecutivos_de_la_misma_clase_en_una_region(monkeypatch, tmp_path):
    # 'S' (senal) en 1-3, 'i' (intracelular) en 4-5, 'H' (helice TM) en 6-9.
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([("acc1", "MKTAYIAKQ", "SSSiiHHHH")]))

    result = predict_tm_signal_regions({"acc1": "MKTAYIAKQ"}, tmp_path)

    rows = {(r.accession, r.start, r.end, r.type) for r in result.itertuples(index=False)}
    assert rows == {
        ("acc1", 1, 3, "signal_peptide"),
        ("acc1", 4, 5, "intracellular"),
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


def test_psmd7_intracelular_sin_tm_ni_senal_se_excluye_por_completo(monkeypatch, tmp_path):
    # Control negativo (subunidad del proteasoma 26S, ver fasta_outputs/PSMD7_P51665_AF_tmbed_raw.pred):
    # sin peptido senal, sin TM, clase 'i' de punta a punta -- 100% citoplasmatica.
    sequence = "MPELAVQKVVVHPLVLLSVVDHFNRIGKVGN"
    classes = "i" * len(sequence)
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([("PSMD7_P51665_AF", sequence, classes)]))

    regions = predict_tm_signal_regions({"PSMD7_P51665_AF": sequence}, tmp_path)

    assert list(regions.itertuples(index=False)) == [
        ("PSMD7_P51665_AF", 1, len(sequence), "intracellular"),
    ]

    # Region candidata (ej. BepiPred+ScanNet) que cae dentro de esa proteina: debe descartarse entera.
    union_df = pd.DataFrame({
        "accession": ["PSMD7_P51665_AF"],
        "start": [5],
        "end": [20],
        "sequence": [sequence[4:20]],
    })
    kept, discarded = filter_overlapping_regions(union_df, regions)

    assert kept.empty
    assert list(discarded["type"]) == ["intracellular"]


def test_thbs2_secretada_pasa_integra_sin_exclusion_indebida(monkeypatch, tmp_path):
    # Control positivo (trombospondina-2, secretada, ver fasta_outputs/THBS2_P35442_AF_tmbed_raw.pred):
    # peptido senal N-terminal (1-18) seguido de clase 'o' (extracelular) en el resto -- sin TM.
    sequence = "MVWRLVLLALWVWPSTQAGHQDKDTTFDLFSISNINRKTIGAKQFRGPDPGVPAYRFVRFDYIPPVNADD"
    classes = "S" * 18 + "o" * (len(sequence) - 18)
    monkeypatch.setattr(subprocess, "run", _mock_run_writing([("THBS2_P35442_AF", sequence, classes)]))

    regions = predict_tm_signal_regions({"THBS2_P35442_AF": sequence}, tmp_path)

    assert set(regions["type"]) == {"signal_peptide"}

    # Region candidata en la proteina madura (fuera del peptido senal): no debe descartarse.
    union_df = pd.DataFrame({
        "accession": ["THBS2_P35442_AF"],
        "start": [30],
        "end": [50],
        "sequence": [sequence[29:50]],
    })
    kept, discarded = filter_overlapping_regions(union_df, regions)

    assert len(kept) == 1
    assert discarded.empty


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

    kept, discarded = filter_overlapping_regions(union_df, regions_df)

    assert len(discarded) == 1
    assert list(discarded["accession"]) == ["acc1"]
    assert list(discarded["start"]) == [1]
    assert list(discarded["type"]) == ["TM_alpha_helix"]
    assert list(kept["accession"]) == ["acc1", "acc2"]
    assert list(kept["start"]) == [20, 5]


def test_filter_overlapping_regions_sin_regiones_no_descarta_nada():
    union_df = pd.DataFrame({"accession": ["acc1"], "start": [1], "end": [9], "sequence": ["AAAAAAAAA"]})

    kept, discarded = filter_overlapping_regions(union_df, pd.DataFrame(columns=["accession", "start", "end", "type"]))

    assert discarded.empty
    assert kept.equals(union_df)
