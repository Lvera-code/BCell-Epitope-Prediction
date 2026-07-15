"""Helper compartido para las tablas de ancho fijo que imprimen en consola los distintos
motores del pipeline (``blast_engine.print_blast_report``, ``consensus.print_union_table``,
``netmhciipan_engine.print_th_report``).

Antes de este modulo, cada uno de esos 3 lugares reimplementaba a mano el mismo patron:
calcular un ancho de columna, construir el header con f-strings alineados, imprimir una
linea de guiones del mismo largo como separador, y volver a aplicar los mismos anchos en
un loop de filas. Este helper extrae esa mecanica repetida sin imponer un formato fijo de
columnas: cada llamador sigue decidiendo que columnas mostrar, sus anchos, alineacion y
como se formatea cada valor (incluyendo casos especiales como NaN), via una lista de
``Column``.

No cambia ningun resultado visual: la migracion de los 3 usos existentes se verifico con
un diff exacto contra la salida previa al refactor.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List

_Align = str  # "<" o ">"


@dataclass
class Column:
    """Especificacion de una columna para ``print_fixed_width_table``.

    Attributes:
        header: Texto de la cabecera de esta columna.
        render: Funcion que recibe una fila (namedtuple de ``df.itertuples()``) y
            devuelve el valor YA FORMATEADO como string (sin padding), p. ej.
            ``lambda r: f"{r.min_rank_el:.3f}"``. Puede ignorar la fila y devolver un
            valor constante (util para columnas literales, como un separador "/").
        width: Ancho minimo de la columna. ``0`` desactiva el padding (la columna se
            imprime con su longitud natural, util para una columna final sin limite).
        align: ``"<"`` (izquierda) o ``">"`` (derecha), mismo significado que el
            especificador de formato de Python.
        prefix: Texto literal impreso inmediatamente antes de esta columna (p. ej. un
            separador de doble espacio entre grupos de columnas).
    """

    header: str
    render: Callable[[Any], str]
    width: int
    align: _Align = "<"
    prefix: str = ""

    def _cell(self, text: str) -> str:
        return f"{self.prefix}{text:{self.align}{self.width}}"


def print_fixed_width_table(rows: Iterable[Any], columns: List[Column]) -> None:
    """Imprime ``rows`` como una tabla de ancho fijo: cabecera, separador y filas.

    No maneja el caso de tabla vacia ni el resumen final: cada llamador conserva su
    propio mensaje de "no hay resultados" y su propia linea de resumen, ya que ese
    texto es especifico de cada fase.

    Args:
        rows: Iterable de filas (tipicamente ``df.itertuples(index=False)``), pasado
            tal cual a cada ``Column.render``.
        columns: Especificacion de columnas, en el orden en que se imprimen.
    """
    header_line = "".join(col._cell(col.header) for col in columns)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(col._cell(col.render(row)) for col in columns))
