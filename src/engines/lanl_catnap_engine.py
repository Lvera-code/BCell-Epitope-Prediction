"""Cruce de candidatos contra epitopos de bnAb conocidos (LANL Immunology DB + CATNAP), 100% local.

bNAber (fuente original pedida por el usuario) esta muerta -- dominio
parqueado/hijackeado, confirmado via Wayback Machine desde antes de mayo
2025 (ver STATUS.md). Reemplazado por datos ya descargados en
``reference_db/`` (decision del usuario, "busca una alternativa que cumpla
la misma funcion"):

- ``reference_db/lanl_immunology/ab_all.csv``: LANL HIV Molecular Immunology
  Database, 1790+ registros de anticuerpos (mismo origen que alimentaba a
  bNAber). Fuente PRIMARIA de este motor: su columna ``Epitope`` trae la
  secuencia lineal del epitopo cuando se conoce (771 de 3799 registros --
  el resto son epitopos conformacionales sin secuencia lineal reportable,
  se omiten sin intentar adivinar una).
- ``reference_db/catnap/`` (CATNAP): neutralizacion (IC50/IC80),
  secuencias y germlines. Se usa aqui solo ``abs_2026-07-01.txt`` para
  anexar potencia/amplitud de neutralizacion (``Mean panel IC50``, numero
  de virus del panel) cuando el nombre del anticuerpo de un match en
  ``ab_all.csv`` coincide (best-effort por nombre exacto normalizado, sin
  fuzzy-matching: un miss silencioso es preferible a una potencia atribuida
  al anticuerpo equivocado).

Este motor NO hace alineamiento estructural ni mapeo de coordenadas HXB2:
compara directamente las secuencias lineales de los candidatos contra las
de ``ab_all.csv`` por solapamiento de subcadena. Es deliberadamente simple
-- sin red, sin dependencias nuevas, sobre CSVs locales ya descargados
(pandas puro, ver STATUS.md) -- a costa de no capturar epitopos
conformacionales (fuera de alcance: requeririan estructura 3D, no solo
secuencia).
"""

import csv
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

_OUTPUT_COLUMNS = [
    "sequence", "antibody_name", "epitope_sequence", "match_length", "epitope_name",
    "hxb2_location", "neutralizing", "antibody_type", "subtype", "binding_region",
    "catnap_mean_ic50", "catnap_n_viruses",
]

_AA_ONLY = re.compile(r"[A-Zx]+")

# Por debajo de esto un solapamiento de subcadena es ruido estadistico
# (aparece por azar en cualquier proteina): ver distribucion real de
# longitudes de epitopo en ab_all.csv (min 3, mediana 12, max 47) -- para
# epitopos YA MAS CORTOS que este umbral se exige igual el match completo
# (no un umbral mas laxo), nunca uno mas corto que el propio epitopo de
# referencia.
_DEFAULT_MIN_OVERLAP = 6


def _load_bnab_epitopes(lanl_ab_all_path: Path) -> pd.DataFrame:
    """Parsea ``ab_all.csv`` y se queda solo con registros con epitopo lineal (secuencia de una sola cadena AA)."""
    with lanl_ab_all_path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    idx = {name: i for i, name in enumerate(header)}
    required = ["Antibody name (alias)", "Epitope", "Epitope name", "HXB2 protein location",
                "Neutralizing", "Antibody type", "Binding region", "Subtype"]
    missing = [c for c in required if c not in idx]
    if missing:
        raise ValueError(f"'{lanl_ab_all_path}' no tiene las columnas esperadas: faltan {missing}.")

    records = []
    for row in rows:
        epitope = row[idx["Epitope"]].strip()
        if not epitope or not _AA_ONLY.fullmatch(epitope):
            continue  # epitopo conformacional, con notacion compuesta ('A + B'), o vacio -- fuera de alcance
        records.append(
            {
                "antibody_name": row[idx["Antibody name (alias)"]].strip(),
                "epitope_sequence": epitope.upper(),
                "epitope_name": row[idx["Epitope name"]].strip(),
                "hxb2_location": row[idx["HXB2 protein location"]].strip(),
                "neutralizing": row[idx["Neutralizing"]].strip(),
                "antibody_type": row[idx["Antibody type"]].strip(),
                "binding_region": row[idx["Binding region"]].strip(),
                "subtype": row[idx["Subtype"]].strip(),
            }
        )
    return pd.DataFrame.from_records(records)


def _load_catnap_potency(catnap_abs_path: Path) -> pd.DataFrame:
    """Parsea ``abs_2026-07-01.txt`` (CATNAP) para anexar potencia/amplitud de neutralizacion por anticuerpo."""
    raw = pd.read_csv(catnap_abs_path, sep="\t", dtype=str)
    required = ["Name", "Mean panel IC50", "# of viruses tested"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"'{catnap_abs_path}' no tiene las columnas esperadas: faltan {missing}.")

    result = raw[required].copy()
    result.columns = ["antibody_name_norm", "catnap_mean_ic50", "catnap_n_viruses"]
    result["antibody_name_norm"] = result["antibody_name_norm"].str.strip().str.upper()
    result["catnap_mean_ic50"] = pd.to_numeric(result["catnap_mean_ic50"], errors="coerce")
    result["catnap_n_viruses"] = pd.to_numeric(result["catnap_n_viruses"], errors="coerce")
    return result.drop_duplicates(subset="antibody_name_norm", keep="first")


def _longest_common_substring_len(a: str, b: str) -> int:
    """Longitud de la subcadena comun mas larga entre ``a`` y ``b`` (DP O(len(a)*len(b)), ambos strings cortos aqui)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                best = max(best, curr[j])
        prev = curr
    return best


def query_bnab_crossref(
    sequences: List[str],
    lanl_ab_all_path: Path,
    catnap_abs_path: Optional[Path] = None,
    min_overlap: int = _DEFAULT_MIN_OVERLAP,
) -> pd.DataFrame:
    """Cruza ``sequences`` contra epitopos lineales de bnAb conocidos (LANL Immunology DB).

    Args:
        sequences: Peptidos/secuencias candidatos a evaluar. Vacio -> DataFrame vacio.
        lanl_ab_all_path: Ruta a ``reference_db/lanl_immunology/ab_all.csv``.
        catnap_abs_path: Ruta a ``reference_db/catnap/abs_YYYY-MM-DD.txt`` (opcional). Si se
            omite, ``catnap_mean_ic50``/``catnap_n_viruses`` quedan en NA para todos los matches.
        min_overlap: Longitud minima de solapamiento de subcadena para reportar un match. Para
            epitopos de referencia MAS CORTOS que este umbral, se exige el match completo del
            epitopo entero (nunca un umbral mas laxo que el propio epitopo).

    Returns:
        DataFrame con una fila por (candidato, epitopo de referencia) que solapa lo suficiente
        (ver ``min_overlap``), columnas ``_OUTPUT_COLUMNS``. Vacio si no hay ningun match o
        ``sequences`` esta vacio.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    bnab_df = _load_bnab_epitopes(lanl_ab_all_path)
    if bnab_df.empty:
        logger.warning("'%s' no aporto ningun epitopo lineal utilizable.", lanl_ab_all_path)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    potency_df = _load_catnap_potency(catnap_abs_path) if catnap_abs_path is not None else None

    rows = []
    for seq in sequences:
        seq_upper = seq.upper()
        for ref in bnab_df.itertuples(index=False):
            required_overlap = min(min_overlap, len(ref.epitope_sequence))
            match_len = _longest_common_substring_len(seq_upper, ref.epitope_sequence)
            if match_len < required_overlap:
                continue

            record = {
                "sequence": seq,
                "antibody_name": ref.antibody_name,
                "epitope_sequence": ref.epitope_sequence,
                "match_length": match_len,
                "epitope_name": ref.epitope_name,
                "hxb2_location": ref.hxb2_location,
                "neutralizing": ref.neutralizing,
                "antibody_type": ref.antibody_type,
                "subtype": ref.subtype,
                "binding_region": ref.binding_region,
                "catnap_mean_ic50": pd.NA,
                "catnap_n_viruses": pd.NA,
            }
            if potency_df is not None:
                hit = potency_df[potency_df["antibody_name_norm"] == ref.antibody_name.strip().upper()]
                if not hit.empty:
                    record["catnap_mean_ic50"] = hit.iloc[0]["catnap_mean_ic50"]
                    record["catnap_n_viruses"] = hit.iloc[0]["catnap_n_viruses"]
            rows.append(record)

    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS) if rows else pd.DataFrame(columns=_OUTPUT_COLUMNS)


_MAX_CONSOLE_ROWS = 30

# Mismo amarillo/negrita que el nucleo MHC, el sequon y los linkers del
# constructo -- un unico codigo de resaltado consistente en todo el pipeline.
_MATCH_ANSI_START = "\033[1;33m"
_MATCH_ANSI_END = "\033[0m"


def _find_match_span(sequence: str, epitope_sequence: str, match_length: int) -> Optional[Tuple[int, int]]:
    """Ubica dentro de ``sequence`` el primer tramo de ``match_length`` aa que tambien aparece en ``epitope_sequence``.

    ``query_bnab_crossref`` solo persiste la LONGITUD del match (``match_length``,
    via ``_longest_common_substring_len``), no su posicion -- para resaltarlo
    hay que volver a ubicarlo. Con ``match_length`` ya confirmado por esa
    funcion, alcanza con probar cada ventana de ese largo dentro de
    ``sequence`` y ver cual aparece tambien en ``epitope_sequence`` (no hace
    falta repetir la programacion dinamica completa).
    """
    if match_length <= 0 or match_length > len(sequence):
        return None
    for start in range(len(sequence) - match_length + 1):
        if sequence[start:start + match_length] in epitope_sequence:
            return start, start + match_length
    return None


_SEQ_WRAP = 40


def _wrap_sequence(sequence: str, wrap: int) -> List[Tuple[int, str]]:
    """Corta ``sequence`` en tramos de a lo sumo ``wrap`` caracteres: ``[(offset_absoluto, tramo), ...]``.

    Es el mecanismo de "cortar la Secuencia hacia abajo": en vez de que la
    columna se estire tanto como el peptido mas largo (hasta 76+ aa en
    corridas reales de VIH, empujando el resto de columnas fuera de la
    pantalla a la derecha), cada fila ocupa varias LINEAS con un tramo fijo
    de ancho, y el resto de columnas solo se imprime una vez en la primera
    linea de esa fila.
    """
    if not sequence:
        return [(0, "")]
    return [(i, sequence[i:i + wrap]) for i in range(0, len(sequence), wrap)]


def _highlight_chunk(padded_cell: str, chunk_offset: int, chunk_len: int, span: Optional[Tuple[int, int]]) -> str:
    """Inyecta amarillo en ``padded_cell`` (un tramo de Secuencia YA paddeado a ancho fijo).

    Igual que en ``netmhciipan_engine.print_traceback_table``: el color se
    inyecta DESPUES de paddear el texto plano, nunca antes -- los codigos
    ANSI cuentan para ``len()`` aunque no ocupen espacio visible, asi que
    colorear antes de aplicar el ancho fijo desalinea la columna.
    """
    if span is None:
        return padded_cell
    local_start = max(span[0], chunk_offset) - chunk_offset
    local_end = min(span[1], chunk_offset + chunk_len) - chunk_offset
    if local_start >= local_end:
        return padded_cell
    return f"{padded_cell[:local_start]}{_MATCH_ANSI_START}{padded_cell[local_start:local_end]}{_MATCH_ANSI_END}{padded_cell[local_end:]}"


def _truncate(text: str, max_len: int) -> str:
    """Corta ``text`` a ``max_len`` caracteres (con '...' final) -- ultimo recurso de
    seguridad para el puñado de valores que ni ``_short_name``/``_short_domain``
    logran acortar lo suficiente (ver ``_display``), nunca el mecanismo principal."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _short_name(value: str) -> str:
    """Se queda con la parte de ``antibody_name`` ANTES del primer ' (' (alias/variantes
    entre parentesis, ej. ``'1F10 (VHH 1F10, H2, VHH H2)'`` -> ``'1F10'``) -- nunca corta
    una palabra a la mitad, el nombre completo sigue en el CSV."""
    return value.split(" (", 1)[0].strip()


def _short_domain(value: str) -> str:
    """Reduce ``binding_region`` al dominio (``'gp120'``/``'gp41'``) solo, salvo que la
    sub-region sea corta y especifica (``'V2'``/``'V3'``/``'MPER'``, <=4 caracteres, sin
    '/') -- ahi se conserva completo (``'gp120 V3'``). Para cualquier otra cosa (listas
    con '/', 'constant regions', 'other/undefined', composiciones de varias regiones)
    se descarta el resto y solo queda el dominio -- nunca una palabra cortada a la mitad
    ni un '...'."""
    tokens = value.split()
    if not tokens:
        return value
    if len(tokens) == 2 and len(tokens[1]) <= 4 and "/" not in tokens[1]:
        return value
    return tokens[0]


def _display(value, max_len: Optional[int] = None, shorten: Optional[Callable[[str], str]] = None) -> str:
    """Formatea un campo de texto de LANL para consola: '-' si es NaN O cadena vacia.

    ``_load_bnab_epitopes`` deja estos campos como cadena vacia (``''``) para
    'sin dato' (nunca ``NaN``, ver su docstring) -- solo se vuelven ``NaN`` si
    el reporte se guarda a CSV y se vuelve a leer con ``pd.read_csv`` (ahi
    pandas si convierte campos vacios a ``NaN``). Un chequeo que solo mire
    ``pd.notna`` deja pasar la cadena vacia tal cual (celda en blanco en vez
    de '-'), asi que hace falta cubrir ambos casos.

    Args:
        shorten: Reduccion SEMANTICA opcional (``_short_name``/``_short_domain``)
            aplicada ANTES de cualquier truncado por caracteres -- el objetivo
            es que la version corta siga siendo una palabra/codigo completo y
            legible, nunca 'gp41 NHR/CH…' cortado a la mitad. ``max_len`` sigue
            actuando como ultimo recurso de seguridad despues de ``shorten``,
            para el puñado de valores que ni asi entran.
    """
    if pd.isna(value) or value == "":
        return "-"
    text = shorten(value) if shorten else value
    return _truncate(text, max_len) if max_len else text


def _tipo_anticuerpo(antibody_type) -> str:
    """Simplifica ``antibody_type`` a 'mono'/'poli' para la columna compacta de consola (el CSV conserva el valor original)."""
    if pd.isna(antibody_type):
        return "-"
    normalized = str(antibody_type).lower()
    if "monoclonal" in normalized:
        return "mono"
    if normalized == "polyclonal":
        return "poli"
    return "-"


def print_bnab_crossref_report(report_df: pd.DataFrame, csv_path: Optional[Path] = None) -> None:
    """Imprime el detalle del cruce con bnAb: analogo a ``algpred_engine.print_allergenicity_report``.

    Trunca la consola a ``_MAX_CONSOLE_ROWS`` filas: un peptido candidato
    puede matchear contra decenas o cientos de anticuerpos de referencia
    (444/440 matches reales en corridas de clado B/C de VIH-1), y volcar
    todo eso a la terminal empuja el resultado de fases ANTERIORES fuera del
    scrollback -- no alcanza con repetir la cabecera (eso ayuda a leer esta
    tabla puntual, no evita que otras tablas mas arriba se pierdan). El CSV
    persistido (``<input_stem>_bnab_crossref.csv``) siempre tiene el 100% de
    los matches Y todas las columnas (incluidas ``epitope_sequence``/
    ``epitope_name``/``hxb2_location``, que no se imprimen en consola), nunca
    se pierde informacion, solo se deja de inundar la consola.

    Las filas mostradas se priorizan (no es simplemente "las primeras N" en
    el orden en que salieron de ``query_bnab_crossref``): primero los
    anticuerpos neutralizantes confirmados, despues por mayor longitud de
    solape -- son los matches mas relevantes biologicamente, los que mas
    conviene ver si hay que truncar.

    ``Secuencia`` se corta hacia ABAJO en vez de estirar la tabla a lo ancho
    (ver ``_wrap_sequence``): un peptido candidato de HIV Env puede medir
    70+ aa, y una sola columna asi de ancha empuja el resto de columnas
    fuera de la pantalla a la derecha -- el objetivo es poder leer la fila
    de izquierda a derecha sin que se superpongan. El tramo que matchea el
    epitopo bnAb se resalta en amarillo (puede caer en cualquiera de los
    tramos de una misma fila, ver ``_highlight_chunk``). Por el mismo motivo
    de espacio, la columna ``Epitopo`` (``epitope_sequence`` completo) no se
    imprime en consola -- solo se usa internamente para ubicar el tramo a
    resaltar; sigue completa en el CSV.

    ``Ab``/``Dominio`` (``antibody_name``/``binding_region`` del CSV) se
    acortan SEMANTICAMENTE, no por cantidad de caracteres -- nunca queda una
    palabra cortada a la mitad con '...' colgando:

    * ``Ab`` (``_short_name``): solo la parte antes del primer ' (' -- ej.
      ``'1F10 (VHH 1F10, H2, VHH H2)'`` -> ``'1F10'`` (hay ``antibody_name``
      reales de hasta 193 caracteres, alias compuestos entre parentesis,
      contra una mediana real de ~5-8).
    * ``Dominio`` (``_short_domain``): solo el dominio (``'gp120'``/
      ``'gp41'``), salvo que la sub-region sea corta y especifica (``'V2'``/
      ``'V3'``/``'MPER'``) -- ahi se conserva completo (``'gp120 V3'``). Para
      cualquier otra cosa (``'gp41 cluster I/II/III'``, ``'gp120 other/
      undefined'``, composiciones con '/') se descarta el resto.

    ``_AB_MAX``/``_DOMAIN_MAX``/``_SUBTYPE_MAX`` (ver ``_display``) son un
    ultimo recurso de seguridad por caracteres, no el mecanismo principal --
    rara vez se disparan con el acortado semantico ya aplicado. El CSV
    siempre tiene el valor completo, nunca acortado. ``Tipo`` simplifica
    ``antibody_type`` a 'mono'/'poli' (ver ``_tipo_anticuerpo``).

    Args:
        report_df: Salida de ``query_bnab_crossref``.
        csv_path: Ruta del CSV persistido con el reporte completo, para
            referenciarla en el aviso de truncado. Si se omite, el aviso no
            menciona ningun nombre de archivo especifico.
    """
    if report_df.empty:
        print("Ningun peptido coincide con un epitopo lineal de bnAb conocido (esperado si la entrada no es HIV Env).")
        return

    _DOMAIN_MAX = 14
    _SUBTYPE_MAX = 10
    _AB_MAX = 16

    ab_width = max(6, report_df["antibody_name"].apply(lambda v: len(_display(v, _AB_MAX, _short_name))).max() + 2)
    domain_width = max(
        9, report_df["binding_region"].apply(lambda v: len(_display(v, _DOMAIN_MAX, _short_domain))).max() + 2
    )
    subtype_width = max(9, report_df["subtype"].apply(lambda v: len(_display(v, _SUBTYPE_MAX))).max() + 2)

    seq_col = Column("Secuencia", lambda r: r.sequence, _SEQ_WRAP, "<")
    rest_columns = [
        Column("Ab", lambda r: _display(r.antibody_name, _AB_MAX, _short_name), ab_width, "<"),
        Column("Neutr", lambda r: _display(r.neutralizing), 8, ">"),
        Column("Subtipo", lambda r: _display(r.subtype, _SUBTYPE_MAX), subtype_width, ">"),
        # prefix="  ": "Subtipo" es right-aligned, sin este separador
        # explicito queda pegada a "Dominio" (mismo caso que tmbed_engine.py).
        Column("Dominio", lambda r: _display(r.binding_region, _DOMAIN_MAX, _short_domain), domain_width, "<", prefix="  "),
        Column("Tipo", lambda r: _tipo_anticuerpo(r.antibody_type), 5, "<"),
        Column("IC50", lambda r: f"{r.catnap_mean_ic50:.3f}" if pd.notna(r.catnap_mean_ic50) else "-", 7, ">"),
        Column("Nvir", lambda r: str(int(r.catnap_n_viruses)) if pd.notna(r.catnap_n_viruses) else "-", 5, ">"),
    ]

    header_line = "".join(c._cell(c.header) for c in [seq_col] + rest_columns)
    separator = "-" * len(header_line)
    blank_rest = "".join(c._cell("") for c in rest_columns)

    def _print_header() -> None:
        print(header_line)
        print(separator)

    n_total = len(report_df)
    if n_total > _MAX_CONSOLE_ROWS:
        priority = (report_df["neutralizing"] == "yes").astype(int)
        display_df = (
            report_df.assign(_priority=priority)
            .sort_values(["_priority", "match_length"], ascending=[False, False])
            .drop(columns="_priority")
            .head(_MAX_CONSOLE_ROWS)
        )
    else:
        display_df = report_df

    _print_header()
    for i, row in enumerate(display_df.itertuples(index=False)):
        if i > 0 and i % 30 == 0:
            print()
            _print_header()

        span = _find_match_span(row.sequence, row.epitope_sequence, int(row.match_length))
        chunks = _wrap_sequence(row.sequence, _SEQ_WRAP)
        rest_line = "".join(c._cell(c.render(row)) for c in rest_columns)

        offset, chunk = chunks[0]
        first_cell = _highlight_chunk(seq_col._cell(chunk), offset, len(chunk), span)
        print(f"{first_cell}{rest_line}")
        for offset, chunk in chunks[1:]:
            cell = _highlight_chunk(seq_col._cell(chunk), offset, len(chunk), span)
            print(f"{cell}{blank_rest}")

    if n_total > _MAX_CONSOLE_ROWS:
        remaining = n_total - _MAX_CONSOLE_ROWS
        where = f" -- ver '{csv_path}' para el detalle completo" if csv_path is not None else ""
        print(f"\n[+{remaining} match(es) mas, no mostrados en consola{where}]")

    n_candidates = report_df["sequence"].nunique()
    n_neutralizing = report_df[report_df["neutralizing"] == "yes"]["sequence"].nunique()
    print(f"\nResumen Fase 6: {n_candidates} peptido(s) coinciden con >=1 epitopo de bnAb conocido "
          f"({n_neutralizing} de ellos con >=1 anticuerpo neutralizante confirmado).")
