"""Tests del helper compartido de formateo de tablas (src/utils/table_format.py)."""

from src.utils.table_format import Column, print_fixed_width_table


def test_print_fixed_width_table_alinea_columnas(capsys):
    columns = [
        Column("Nombre", lambda r: r["nombre"], 10, "<"),
        Column("Edad", lambda r: str(r["edad"]), 5, ">"),
    ]
    rows = [{"nombre": "Ana", "edad": 30}, {"nombre": "Bartolo", "edad": 5}]

    print_fixed_width_table(rows, columns)

    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"{'Nombre':<10}{'Edad':>5}"
    assert out[1] == "-" * len(out[0])
    assert out[2] == f"{'Ana':<10}{'30':>5}"
    assert out[3] == f"{'Bartolo':<10}{'5':>5}"


def test_print_fixed_width_table_width_cero_no_agrega_padding(capsys):
    columns = [Column("Texto", lambda r: r, 0, "<")]
    print_fixed_width_table(["hola"], columns)
    out = capsys.readouterr().out.splitlines()
    assert out[2] == "hola"


def test_print_fixed_width_table_prefix_se_imprime_antes_de_la_columna(capsys):
    columns = [
        Column("A", lambda r: "x", 3, "<"),
        Column("B", lambda r: "y", 3, "<", prefix="  "),
    ]
    print_fixed_width_table(["fila"], columns)
    out = capsys.readouterr().out.splitlines()
    assert out[2] == "x    y  "
