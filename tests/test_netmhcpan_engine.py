"""Tests de sanidad del panel de referencia de la Fase 5b (src/engines/netmhcpan_engine.py).

No hay tests del motor completo (``predict_netmhcpan`` invoca el binario
propietario de NetMHCpan-4.2, con licencia academica -- mismo criterio de
"sin mockear el binario real" que el resto del pipeline para engines
DTU Health Tech no cubiertos aun). Este archivo cubre lo que SI se puede
verificar sin el binario instalado: el formato y contenido de
``NETMHCPAN_REFERENCE_PANEL`` en si, en particular la ampliacion con HLA-C
(2026-07-22, ver docstring del modulo y STATUS.md para la investigacion
completa) -- para que un futuro cambio accidental del panel (typo, alelo
duplicado, espacio que rompe el parser de NetMHCpan) falle un test en vez
de solo notarse en una corrida real.
"""

from src.engines.netmhcpan_engine import NETMHCPAN_REFERENCE_PANEL


def _alleles():
    return NETMHCPAN_REFERENCE_PANEL.split(",")


def test_panel_no_tiene_espacios():
    # NetMHCpan pasa el panel tal cual a su parser de '-a': un espacio entre
    # comas rompe el parseo del alelo siguiente (ver docstring del modulo).
    assert " " not in NETMHCPAN_REFERENCE_PANEL


def test_panel_no_tiene_alelos_duplicados():
    alleles = _alleles()
    assert len(alleles) == len(set(alleles))


def test_panel_incluye_hla_a_b_y_c():
    alleles = _alleles()
    assert any(a.startswith("HLA-A") for a in alleles)
    assert any(a.startswith("HLA-B") for a in alleles)
    assert any(a.startswith("HLA-C") for a in alleles)


def test_panel_tiene_12_hla_a_b_y_11_hla_c():
    # 12 (Sidney et al. 2008, supertipos A/B) + 11 HLA-C (Rasmussen et al.
    # 2014 + criterio de frecuencia poblacional >=1% de IEDB, agregados
    # 2026-07-22). Ver docstring del modulo para el detalle completo.
    alleles = _alleles()
    n_a = sum(1 for a in alleles if a.startswith("HLA-A"))
    n_b = sum(1 for a in alleles if a.startswith("HLA-B"))
    n_c = sum(1 for a in alleles if a.startswith("HLA-C"))
    assert (n_a, n_b, n_c) == (5, 7, 11)
    assert len(alleles) == 23


def test_alelos_hla_c_tienen_formato_netmhcpan():
    # Formato NetMHCpan verificado contra el binario local (ver
    # 'netMHCpan-4.2/data/allelenames'): 'HLA-C<2 digitos>:<2 digitos>',
    # sin '*' (a diferencia de la nomenclatura oficial HLA-C*01:02).
    for allele in _alleles():
        if allele.startswith("HLA-C"):
            assert "*" not in allele
            resto = allele[len("HLA-C"):]
            digitos, _, sufijo = resto.partition(":")
            assert len(digitos) == 2 and digitos.isdigit()
            assert len(sufijo) == 2 and sufijo.isdigit()
