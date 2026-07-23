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
from typing import Any, Callable, Iterable, List, Optional

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


def print_fixed_width_table(
    rows: Iterable[Any],
    columns: List[Column],
    line_formatter: Optional[Callable[[str, Any], str]] = None,
    group_by: Optional[Callable[[Any], Any]] = None,
    repeat_header_every: Optional[int] = None,
) -> None:
    """Imprime ``rows`` como una tabla de ancho fijo: cabecera, separador y filas.

    No maneja el caso de tabla vacia ni el resumen final: cada llamador conserva su
    propio mensaje de "no hay resultados" y su propia linea de resumen, ya que ese
    texto es especifico de cada fase.

    Args:
        rows: Iterable de filas (tipicamente ``df.itertuples(index=False)``), pasado
            tal cual a cada ``Column.render``.
        columns: Especificacion de columnas, en el orden en que se imprimen.
        line_formatter: Opcional. Si se pasa, se aplica a cada linea de fila YA
            formateada (con todo el padding aplicado) antes de imprimirla --
            pensado para inyectar resaltado ANSI sin romper el alineado (los
            codigos ANSI no ocupan espacio visible en terminal pero SI cuentan
            para ``len()``, asi que insertarlos ANTES del padding desalinea la
            columna; ver ``netmhciipan_engine.print_traceback_table`` para el
            caso donde se confirmo esto empiricamente). Recibe ``(linea, fila)``
            y devuelve la linea final a imprimir; si la fila no aplica
            resaltado, debe devolver ``linea`` sin modificar.
        group_by: Opcional. Si se pasa, se llama con cada fila para obtener una
            clave de agrupamiento (tipicamente ``lambda r: r.accession``) --
            cada vez que la clave cambia respecto a la fila anterior se
            imprime una linea separadora ANTES de esa fila (nunca antes de la
            primera). Pensado para FASTA multi-registro: que cada proteina se
            lea como un bloque propio en vez de una lista continua (mismo
            patron que ya usaba a mano
            ``netmhciipan_engine.print_traceback_table`` antes de este
            parametro). Sin este argumento (default), el comportamiento no
            cambia respecto a antes.
        repeat_header_every: Opcional. Si se pasa, reimprime cabecera +
            separador cada N filas de datos ademas de la inicial -- pensado
            para tablas que pueden crecer a cientos de filas (ej. Fase 6
            cruzando cientos de candidatos contra cientos de epitopos de
            referencia, ver ``lanl_catnap_engine.print_bnab_crossref_report``),
            donde con la cabecera fuera del scrollback de la terminal ya no
            se puede saber que representa cada columna. Sin este argumento
            (default), el comportamiento no cambia.
    """
    header_line = "".join(col._cell(col.header) for col in columns)
    separator = "-" * len(header_line)

    def _print_header() -> None:
        print(header_line)
        print(separator)

    _print_header()
    prev_key = None
    is_first = True
    n_printed = 0
    for row in rows:
        if group_by is not None:
            key = group_by(row)
            if not is_first and key != prev_key:
                print(separator)
            prev_key = key
        if repeat_header_every and n_printed > 0 and n_printed % repeat_header_every == 0:
            print()
            _print_header()
        is_first = False
        line = "".join(col._cell(col.render(row)) for col in columns)
        print(line_formatter(line, row) if line_formatter else line)
        n_printed += 1
