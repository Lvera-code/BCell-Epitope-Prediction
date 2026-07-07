"""Suite de estres inquebrantable del SOTA-B-Epitope-Pipeline.

Tres invariantes criticos que el pipeline NUNCA debe violar:

1. Tolerancia a fallos: caracteres ilegales en un FASTA se purgan sin abortar
   la ejecucion (codigo de salida 0).
2. Verdad biologica: una proteina de superficie viral notoriamente
   inmunogenica (Spike de SARS-CoV-2) debe puntuar por encima de una proteina
   humana intracelular "housekeeping" (GAPDH), la misma clase de proteina
   endogena no antigenica usada como control negativo en el entrenamiento.
3. Estres geometrico: el Sliding Window Stitcher cubre macromoleculas de
   longitud arbitraria (>3000 aa) con el numero exacto de ventanas que
   predice la aritmetica del solapamiento, sin truncar un solo residuo.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.engines.antigenicity_cnn import AntigenicityCNNEngine
from src.engines.epitope_engine import compute_sliding_windows
from src.models import SequenceRecord
from src.utils.fasta_parser import FastaParser
from src.validation.benchmark_suite import run_macromolecular_stress_test

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Secuencias de referencia (UniProt, Swiss-Prot reviewed) ---
# P04406: Gliceraldehido-3-fosfato deshidrogenasa (GAPDH) humana. Enzima
# metabolica intracelular "housekeeping" -- exactamente la clase de proteina
# endogena no antigenica curada como control negativo por
# ``src/training/dataset_prep.py`` (HOUSEKEEPING_ACCESSIONS). A diferencia de
# la Albumina (extracelular/secretada, rica en parches hidrofilicos de
# superficie y por tanto FUERA de la distribucion de negativos de
# entrenamiento), GAPDH es representativa de lo que la Fase 1 aprendio
# efectivamente a reconocer como "no antigenico".
HUMAN_GAPDH_P04406 = (
    "MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTV"
    "KAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVI"
    "ISAPSADAPMFVMGVNHEKYDNSLKIISNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHA"
    "ITATQKTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELNGKLTGMAFRVPTANV"
    "SVVDLTCRLEKPAKYDDIKKVVKQASEGPLKGILGYTEHQVVSSDFNSDTHSSTFDAGAG"
    "IALNDHFVKLISWYDNEFGYSNRVVDLMAHMASKE"
)

# P0DTC2: Glucoproteina Spike de SARS-CoV-2 (antigeno de superficie viral,
# blanco principal de la respuesta humoral; fuertemente inmunogenica).
SARS_COV2_SPIKE_P0DTC2 = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS"
    "NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV"
    "NNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLE"
    "GKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQT"
    "LLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETK"
    "CTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISN"
    "CVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD"
    "YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPC"
    "NGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVN"
    "FNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITP"
    "GTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSY"
    "ECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTI"
    "SVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQE"
    "VFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC"
    "LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAM"
    "QMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALN"
    "TLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRA"
    "SANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPA"
    "ICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDP"
    "LQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDL"
    "QELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDD"
    "SEPVLKGVKLHYT"
)


# ---------------------------------------------------------------------------
# 1. Tolerancia a fallos: purga de caracteres ilegales sin romper la ejecucion
# ---------------------------------------------------------------------------

_STRESS_FASTA_CONTENT = (
    ">VALID_SEQ una secuencia perfectamente canonica\n"
    "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQT\n"
    ">ILLEGAL_BANG_SEQ contiene un caracter ilegal '!'\n"
    "MKVLWAALL!VTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQT\n"
    ">AMBIGUOUS_Z_SEQ contiene residuos ambiguos 'Z'\n"
    "MKVLWAALLZVTFLAGZCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQT\n"
)


def test_fasta_sanitization_purges_illegal_without_crashing(tmp_path):
    """El modulo de aduana debe purgar sin abortar: descarta lo irrecuperable,
    excinde lo ambiguo, y conserva lo valido."""
    fasta_path = tmp_path / "stress.fasta"
    fasta_path.write_text(_STRESS_FASTA_CONTENT, encoding="utf-8")

    records = FastaParser.parse(fasta_path, min_length=9)
    ids = {record.id: record for record in records}

    assert "VALID_SEQ" in ids, "La secuencia perfectamente valida debe sobrevivir intacta."
    assert "ILLEGAL_BANG_SEQ" not in ids, (
        "Una secuencia con un caracter fuera del alfabeto canonico (p. ej. '!') "
        "debe descartarse por completo, no debe propagarse corrupta."
    )
    assert "AMBIGUOUS_Z_SEQ" in ids, (
        "Una secuencia con residuos ambiguos IUPAC ('Z') debe sobrevivir: se "
        "excinden solo esos residuos, no se descarta el registro completo."
    )
    assert "Z" not in ids["AMBIGUOUS_Z_SEQ"].sequence, "Los residuos 'Z' deben quedar excindidos."


def test_cli_survives_illegal_characters_exit_code_zero(tmp_path):
    """El pipeline completo (subprocess real) debe terminar con codigo 0 ante
    un FASTA con secuencias corruptas, purgandolas en lugar de abortar.

    Se fuerza un umbral > 1.0 para que ninguna secuencia supere la Fase 1: el
    test verifica la tolerancia a fallos del saneamiento y la finalizacion
    limpia del proceso, sin depender de la disponibilidad/velocidad de ESM-2.
    """
    fasta_path = tmp_path / "stress_cli.fasta"
    fasta_path.write_text(_STRESS_FASTA_CONTENT, encoding="utf-8")
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "-i",
            str(fasta_path),
            "-t",
            "1.1",
            "--offline",
            "-o",
            str(output_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"El CLI debe salir con codigo 0 pese a residuos ilegales en el FASTA de "
        f"entrada.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 2. Verdad biologica: Spike (antigenico) > GAPDH (housekeeping, no antigenico)
# ---------------------------------------------------------------------------

def test_gapdh_score_lower_than_spike_score():
    """El score de antigenicidad calibrado de Fase 1 debe reflejar biologia real:

    GAPDH humana (enzima metabolica intracelular "housekeeping", la misma
    clase de proteina curada como control negativo en el entrenamiento) debe
    puntuar estrictamente por debajo de la glucoproteina Spike de SARS-CoV-2
    (antigeno de superficie viral, blanco principal de anticuerpos).
    """
    engine = AntigenicityCNNEngine(threshold=Settings.ANTIGENICITY_THRESHOLD)

    gapdh_record = SequenceRecord(
        id="GAPDH_HUMAN_P04406", sequence=HUMAN_GAPDH_P04406, description="Human GAPDH (housekeeping)"
    )
    spike_record = SequenceRecord(
        id="SPIKE_SARS2_P0DTC2", sequence=SARS_COV2_SPIKE_P0DTC2, description="SARS-CoV-2 Spike glycoprotein"
    )

    results = engine.run([gapdh_record, spike_record])
    scores = {result.record.id: result.score for result in results}

    assert scores["GAPDH_HUMAN_P04406"] < scores["SPIKE_SARS2_P0DTC2"], (
        f"Verdad biologica violada: GAPDH ({scores['GAPDH_HUMAN_P04406']:.4f}) deberia "
        f"puntuar por debajo de la Spike de SARS-CoV-2 ({scores['SPIKE_SARS2_P0DTC2']:.4f})."
    )


# ---------------------------------------------------------------------------
# 3. Estres geometrico: aritmetica del Sliding Window Stitcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seq_len, window_size, overlap",
    [
        (3500, 1000, 200),  # configuracion por defecto del pipeline, >3000 aa
        (1273, 1000, 200),  # longitud real de la Spike de SARS-CoV-2
        (5000, 1000, 200),  # limite auditado en el historial del proyecto
        (900, 1000, 200),  # por debajo de la ventana: una unica ventana
    ],
)
def test_sliding_window_count_matches_overlap_math(seq_len, window_size, overlap):
    """El numero de ventanas debe coincidir EXACTAMENTE con la aritmetica del
    solapamiento: ``1 + ceil((L - W) / (W - overlap))`` para ``L > W``."""
    windows = compute_sliding_windows(seq_len, window_size, overlap)

    if seq_len <= window_size:
        expected_count = 1
    else:
        stride = window_size - overlap
        expected_count = 1 + -(-(seq_len - window_size) // stride)  # ceil division

    assert len(windows) == expected_count, (
        f"Para L={seq_len}, W={window_size}, overlap={overlap}: se esperaban "
        f"{expected_count} ventanas, se obtuvieron {len(windows)}."
    )

    # La union de las ventanas debe cubrir exactamente [0, seq_len), sin huecos.
    assert windows[0][0] == 0
    assert windows[-1][1] == seq_len
    for (_, end_a), (start_b, _) in zip(windows, windows[1:]):
        assert start_b <= end_a, "No debe haber huecos entre ventanas consecutivas."

    # El solapamiento real entre ventanas consecutivas debe ser el nominal,
    # salvo en el ultimo par (la ventana final puede ser mas corta que
    # window_size si seq_len no es multiplo exacto del stride).
    for idx, ((start_a, end_a), (start_b, end_b)) in enumerate(zip(windows, windows[1:])):
        is_last_pair = idx == len(windows) - 2
        if is_last_pair and end_b == seq_len and (end_b - start_b) < window_size:
            continue
        assert end_a - start_b == overlap


def test_sliding_window_stitcher_processes_massive_sequence():
    """El Sliding Window Stitcher debe ensamblar y procesar de extremo a
    extremo (Fase 1 + Fase 2) una macromolecula sintetica >3000 aa sin
    truncar un solo residuo y sin excepciones (OOM u otras)."""
    passed = run_macromolecular_stress_test(length=3100)
    assert passed, "El stress test macromolecular del Sliding Window Stitcher debe superar la cobertura completa."
