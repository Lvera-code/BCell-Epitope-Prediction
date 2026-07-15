"""Tests de la Fase 5 (src/engines/netmhciipan_engine.py): parseo de .xls, exclusion de
alelos invertidos, veredicto de promiscuidad, traceback a la Fase 3/4 y deduplicacion de
ventanas del modo proteina.

Logica 100% local: ``_parse_xls`` se prueba escribiendo un .xls sintetico con el mismo
formato de 2 filas de cabecera que produce el binario real (comentario + nombres de alelo
+ nombres de columna), sin invocar NetMHCIIpan ni ningun subprocess.
"""

import math

import pandas as pd
import pytest

from src.config.settings import Settings
from src.engines.netmhciipan_engine import (
    _MAX_PEPTIDE_MODE_LENGTH,
    _parse_xls,
    build_traceback_report,
    validate_allele_extra,
)


def _write_xls(tmp_path, alleles, rows):
    """Construye un .xls sintetico con el mismo formato que devuelve el binario real.

    Args:
        alleles: lista de nombres de alelo, en el orden del panel '-a'.
        rows: lista de (peptido, [(core, inverted, rank_el), ...]) -una tupla por alelo,
            mismo orden que ``alleles``-.
    """
    n = len(alleles)
    comment_line = "#/fake/NetMHCIIpan-4.3 -p -f peptides.pep -a " + ",".join(alleles)
    allele_header = "\t\t\t\t" + "\t\t\t\t".join(alleles) + "\t\t"
    col_header = "Pos\tPeptide\tID\tTarget\t" + "\t".join(["Core\tInverted\tScore_EL\tRank_EL"] * n) + "\tAve\tNB"

    lines = [comment_line, allele_header, col_header]
    for pos, (peptide, allele_values) in enumerate(rows, start=1):
        fields = [str(pos), peptide, "Sequence", "NA"]
        for core, inverted, rank_el in allele_values:
            fields += [core, str(int(inverted)), "0.5", str(rank_el)]
        fields += ["0.5", "1"]
        lines.append("\t".join(fields))

    path = tmp_path / "synthetic.xls"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- _parse_xls: exclusion de alelos invertidos desde el calculo -----------------------


def test_parse_xls_alelo_ganador_normal_no_se_modifica(tmp_path):
    alleles = ["A1", "A2", "A3"]
    rows = [("PEPTIDEONE9AA", [("PTIDEONE9", 0, 0.5), ("EONE9AAXX", 0, 3.0), ("XXX9AAYYY", 0, 8.0)])]
    xls_path = _write_xls(tmp_path, alleles, rows)

    result = _parse_xls(xls_path, n_alleles=3)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["core_9aa"] == "PTIDEONE9"  # el de menor Rank_EL (0.5)
    assert row["min_rank_el"] == pytest.approx(0.5)


def test_parse_xls_excluye_alelo_invertido_aunque_tenga_mejor_rank(tmp_path):
    # El alelo con MEJOR (menor) Rank_EL esta invertido: no debe ganar ni contar.
    alleles = ["A1", "A2", "A3"]
    rows = [
        (
            "PEPTIDEONE9AA",
            [
                ("INVERTEDCORE", 1, 0.1),  # mejor rank de todos, pero invertido -> se ignora
                ("PTIDEONE9NORMAL", 0, 2.0),  # mejor rank ENTRE LOS NORMALES
                ("XXXNORMALYYY", 0, 8.0),
            ],
        )
    ]
    xls_path = _write_xls(tmp_path, alleles, rows)

    result = _parse_xls(xls_path, n_alleles=3)

    row = result.iloc[0]
    assert row["core_9aa"] == "PTIDEONE9NORMAL"
    assert row["min_rank_el"] == pytest.approx(2.0)


def test_parse_xls_veredicto_usa_solo_promiscuidad_normal(tmp_path):
    weak = Settings.NETMHCIIPAN_RANK_WEAK
    min_alleles = Settings.NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES
    alleles = [f"A{i}" for i in range(1, 6)]  # 5 alelos
    # 2 normales bajo el umbral + 3 invertidos bajo el umbral: total=5 (pasaria
    # con la logica vieja de min_alleles=3), pero normal=2 < 3 -> debe rechazarse.
    rows = [
        (
            "PEPTIDEXXXXXXXXX",
            [
                ("CORENORMAL1", 0, weak - 1),
                ("CORENORMAL2", 0, weak - 1),
                ("COREINVERT1", 1, weak - 1),
                ("COREINVERT2", 1, weak - 1),
                ("COREINVERT3", 1, weak - 1),
            ],
        )
    ]
    xls_path = _write_xls(tmp_path, alleles, rows)

    result = _parse_xls(xls_path, n_alleles=5)

    row = result.iloc[0]
    assert row["n_alelos_promiscuos"] == 2
    assert min_alleles == 3  # supuesto del test: confirma el default de Settings
    assert row["veredicto"] == "Rechazado"


def test_parse_xls_veredicto_valido_con_suficientes_alelos_normales(tmp_path):
    weak = Settings.NETMHCIIPAN_RANK_WEAK
    alleles = [f"A{i}" for i in range(1, 4)]
    rows = [
        (
            "PEPTIDEXXXXXXXXX",
            [("CORE1", 0, weak - 1), ("CORE2", 0, weak - 1), ("CORE3", 0, weak - 1)],
        )
    ]
    xls_path = _write_xls(tmp_path, alleles, rows)

    result = _parse_xls(xls_path, n_alleles=3)

    assert result.iloc[0]["n_alelos_promiscuos"] == 3
    assert result.iloc[0]["veredicto"] == "Candidato Valido"


def test_parse_xls_todos_los_alelos_invertidos_no_inventa_core_normal(tmp_path):
    alleles = ["A1", "A2"]
    rows = [("PEPTIDEXXXXXXXXX", [("COREINV1", 1, 0.5), ("COREINV2", 1, 0.8)])]
    xls_path = _write_xls(tmp_path, alleles, rows)

    result = _parse_xls(xls_path, n_alleles=2)

    row = result.iloc[0]
    assert row["n_alelos_promiscuos"] == 0
    assert math.isinf(row["min_rank_el"])
    assert row["veredicto"] == "Rechazado"


def test_parse_xls_formato_inesperado_lanza_error(tmp_path):
    from src.utils.exceptions import ImmunogenicityExecutionError

    bad_path = tmp_path / "bad.xls"
    bad_path.write_text("linea1\nlinea2\ncolumna_rara\tsin\tel\tformato\tesperado\n")

    with pytest.raises(ImmunogenicityExecutionError):
        _parse_xls(bad_path, n_alleles=27)


# --- build_traceback_report: traceback de coordenadas -----------------------------------


def _parent_df(accession, start, sequence, origen="Consenso", bp=0.5, ed=0.5):
    return pd.DataFrame(
        {
            "accession": [accession],
            "start": [start],
            "end": [start + len(sequence) - 1],
            "sequence": [sequence],
            "origen": [origen],
            "bepipred_score": [bp],
            "epidope_score": [ed],
        }
    )


def _report_row(sequence, core, n_prom, min_rank, veredicto="Candidato Valido", n_eval=27):
    return {
        "sequence": sequence,
        "core_9aa": core,
        "n_alelos_evaluados": n_eval,
        "n_alelos_promiscuos": n_prom,
        "min_rank_el": min_rank,
        "veredicto": veredicto,
    }


def test_traceback_recalcula_coordenadas_absolutas():
    parent = _parent_df("ACC1", 100, "AAABBBCCCDDDEEEFFFGGGHHH")
    # 'CCCDDDEEE' esta en el offset 6 del padre -> start real = 100 + 6 = 106.
    report = pd.DataFrame([_report_row("CCCDDDEEE", "CCCDDDEEE", 3, 0.5)])

    result = build_traceback_report(report, parent)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["accession"] == "ACC1"
    assert row["start"] == 106
    assert row["end"] == 114


def test_traceback_candidato_sin_match_se_omite_con_warning(caplog):
    parent = _parent_df("ACC1", 1, "AAABBBCCC")
    report = pd.DataFrame([_report_row("ZZZZZZZZZ", "ZZZZZZZZZ", 3, 0.5)])

    result = build_traceback_report(report, parent)

    assert result.empty
    assert "no se pudo trazar" in caplog.text.lower()


def test_traceback_ignora_filas_rechazadas():
    parent = _parent_df("ACC1", 1, "AAABBBCCCDDD")
    report = pd.DataFrame(
        [
            _report_row("AAABBBCCC", "AAABBBCCC", 3, 0.5, veredicto="Candidato Valido"),
            _report_row("BBBCCCDDD", "BBBCCCDDD", 1, 5.0, veredicto="Rechazado"),
        ]
    )

    result = build_traceback_report(report, parent)

    assert len(result) == 1
    assert result.iloc[0]["sequence_f5"] == "AAABBBCCC"


# --- Deduplicacion de ventanas del modo proteina -----------------------------------------


def test_dedup_fusiona_solo_si_core_y_promiscuidad_coinciden_exacto():
    # Replica el ejemplo real acordado con el usuario (AGR38513.1, 66-83):
    # filas 1 y 2 comparten core pero distinta promiscuidad -> NO se fusionan.
    # filas 3 y 4 comparten core Y promiscuidad -> se fusionan, gana el de menor %Rank.
    parent = _parent_df("AGR38513.1", 66, "NEAAITDSAVAVAAASST")
    report = pd.DataFrame(
        [
            _report_row("NEAAITDSAVAVAAA", "ITDSAVAVA", 5, 0.326225),
            _report_row("EAAITDSAVAVAAAS", "ITDSAVAVA", 6, 0.157271),
            _report_row("AAITDSAVAVAAASS", "DSAVAVAAA", 3, 0.286541),
            _report_row("AITDSAVAVAAASST", "DSAVAVAAA", 3, 0.344177),
        ]
    )

    result = build_traceback_report(report, parent)

    assert len(result) == 3
    fused = result[result["core_9aa"] == "DSAVAVAAA"]
    assert len(fused) == 1
    assert fused.iloc[0]["sequence_f5"] == "AAITDSAVAVAAASS"  # el de menor min_rank_el
    assert fused.iloc[0]["min_rank_el"] == pytest.approx(0.286541)

    distinct_prom = result[result["core_9aa"] == "ITDSAVAVA"]
    assert len(distinct_prom) == 2  # NO se fusionan: promiscuidad distinta


def test_dedup_core_distinto_por_1aa_no_se_fusiona():
    parent = _parent_df("ACC1", 1, "AAABBBCCCDDDEEE")
    report = pd.DataFrame(
        [
            _report_row("AAABBBCCCDDDEEE", "AABBBCCCD", 3, 0.5),
            _report_row("AAABBBCCCDDDEEE", "ABBBCCCDD", 3, 0.4),  # core distinto por 1 aa, misma promiscuidad
        ]
    )

    result = build_traceback_report(report, parent)

    assert len(result) == 2


# --- validate_allele_extra ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "DRB1_1602",
        "DRB1_1602,DRB1_1301",
        "HLA-DQA10501-DQB10201",
        "HLA-DPA10201-DPB10101",
        "DRB3_0101,HLA-DQA10501-DQB10201,HLA-DPA10201-DPB10101",
    ],
)
def test_validate_allele_extra_formatos_validos(value):
    assert validate_allele_extra(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "DRB1_1602, DRB1_1301",  # espacio
        "DRB1_16O2",  # letra 'O' en vez de digito
        "DQA10501-DQB10201",  # falta prefijo 'HLA-'
        "DRB1_160",  # solo 3 digitos
        "DRB1_1602,",  # token vacio al final
    ],
)
def test_validate_allele_extra_formatos_invalidos(value):
    with pytest.raises(ValueError):
        validate_allele_extra(value)


def test_max_peptide_mode_length_sigue_bajo_el_margen_de_crash_conocido():
    # Ver netmhciipan_engine.py: crash confirmado empiricamente en 56 aa con el
    # panel de 27 alelos de IEDB_REFERENCE_PANEL. Este test documenta el supuesto
    # como regresion: si alguien baja el margen por error, se entera aqui.
    assert _MAX_PEPTIDE_MODE_LENGTH < 56
