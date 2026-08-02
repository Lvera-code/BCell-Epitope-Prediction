"""Cruce de candidatos B-cell contra epitopos con proteccion/neutralizacion documentada (IEDB), 100% local.

Feedback de Carmen Elena Gomez (group leader Poxvirus and Vaccines, CNB-CSIC,
2026-07-30): identificar que regiones de un patogeno tienen efecto
protector/neutralizante YA DOCUMENTADO (su ejemplo: V1/V2, CD4bs, MPER de
HIV), para poder priorizarlas -- lo mismo que ``lanl_catnap_engine`` (Fase 6)
ya hace, pero especifico de HIV Env. Este motor generaliza esa misma idea a
CUALQUIER patogeno estudiado, usando el bulk export B-cell de IEDB
(iedb.org) como fuente.

``reference_db/iedb/bcell_protective_epitopes.csv`` es un subconjunto YA
FILTRADO y persistido localmente del export completo de IEDB
(``bcell_full_v3_single_file.zip``, 1,688,617 filas / 3.24 GB descomprimido
-- ese archivo crudo NO se distribuye con el repo, solo el filtrado). El
filtro (ver ADR completo en el vault,
``01-Proyectos/BCell-Epitope-Prediction/Decisiones/2026-08-01-investigacion-iedb-regiones-interes.md``)
se aplico UNA SOLA VEZ como paso de SETUP (descarga+filtrado, nunca en
runtime, mismo principio que ``reference_db/lanl_immunology/ab_all.csv``):

    - ``Epitope.Object Type == 'Linear peptide'`` (excluye epitopos
      discontinuos/conformacionales y no-peptidicos, sin secuencia lineal
      reportable -- mismo criterio que ``lanl_catnap_engine`` con
      ``ab_all.csv``).
    - ``Assay.Response measured`` en {'neutralization', 'protection from
      pathogen/tumor/other challenge'(+variantes 'after adoptive transfer'),
      'protection from fertility'} -- distingue un ensayo de EFECTO real
      (neutraliza / protege) de "reconocido" generico (binding/ELISA sin
      consecuencia funcional).
    - ``Assay.Qualitative Measure`` empieza con 'Positive' (excluye
      'Negative': un ensayo NEGATIVO de neutralizacion no es una region de
      interes).

Resultado: 3429 registros (de 1.68M), 561 organismos distintos -- HIV es
apenas uno mas entre muchos (SARS-CoV-2, P. falciparum, HCV, HPV16,
poliovirus, B. anthracis, S. aureus, etc.), no un caso especial.

Regenerar el CSV filtrado (si IEDB publica una version mas reciente del
bulk export): descargar
``https://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3_single_file.zip``,
descomprimir, y aplicar el mismo filtro de 3 columnas de arriba sobre
``('Epitope','Name')``/``('Epitope','Object Type')``/``('Epitope','Source
Organism')``/``('Assay','Response measured')``/``('Assay','Qualitative
Measure')`` (encabezado de 2 niveles, ``pd.read_csv(header=[0,1],
encoding='latin-1')``) -- ver README.md, Seccion de Instalacion, para el
comando completo.

Sin mapeo organismo<->accession: a diferencia de intentar filtrar el CSV por
el patogeno de la corrida actual (requeriria taxonomia, sin forma confiable
de derivarla del header de un FASTA de entrada), este motor NO pre-filtra
por organismo -- el MATCH POR SECUENCIA ya funciona como filtro implicito
(si el organismo de entrada no tiene ningun epitopo documentado que
solape, simplemente no hay filas de resultado), exactamente el mismo
principio que ya usa Fase 6 (``lanl_catnap_engine``, que tampoco filtra por
organismo antes de cruzar). Reusa ``_longest_common_substring_len`` de ese
motor sin duplicar el algoritmo.

Solo se aplica a candidatos B-cell (``safe_df``), NO a HTL/CTL: el export de
IEDB usado aqui es especificamente de ensayos B-cell (union/neutralizacion
por ANTICUERPO), un mecanismo de reconocimiento distinto al de presentacion
MHC-I/II que evaluan HTL/CTL -- mismo razonamiento ya aplicado en
``construct_assembly`` para no aplicarle el filtro de alergenicidad
(AlgPred2) a HTL/CTL (necesita un epitopo libre/circulante, que un nucleo
enterrado en el surco del MHC nunca es).
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.engines.lanl_catnap_engine import _find_match_span, _longest_common_substring_len
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column

logger = setup_logger(__name__)

_OUTPUT_COLUMNS = [
    "sequence", "epitope_sequence", "match_length", "source_organism",
    "response_measured", "qualitative_measure", "method", "host", "pmid",
]

_RAW_CSV_COLUMNS = [
    "epitope_sequence", "source_organism", "response_measured",
    "qualitative_measure", "method", "host", "pmid",
]

# Mismo umbral y mismo rationale que LANL_CATNAP_MIN_OVERLAP (ver
# lanl_catnap_engine.py): por debajo de esto, un solapamiento de subcadena es
# ruido estadistico esperable por azar contra cualquier proteina.
_DEFAULT_MIN_OVERLAP = 6


def _load_iedb_epitopes(iedb_csv_path: Path) -> pd.DataFrame:
    """Carga el subconjunto YA FILTRADO de IEDB (ver docstring del modulo)."""
    df = pd.read_csv(iedb_csv_path, dtype=str)
    missing = [c for c in _RAW_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"'{iedb_csv_path}' no tiene las columnas esperadas: faltan {missing}.")
    df["epitope_sequence"] = df["epitope_sequence"].str.strip().str.upper()
    return df


def query_iedb_crossref(
    sequences: List[str],
    iedb_csv_path: Path,
    min_overlap: int = _DEFAULT_MIN_OVERLAP,
) -> pd.DataFrame:
    """Cruza ``sequences`` contra epitopos IEDB con proteccion/neutralizacion documentada.

    Args:
        sequences: Peptidos candidatos B-cell a evaluar (``safe_df['sequence']``,
            mismo insumo que Fase 6). Vacio -> DataFrame vacio.
        iedb_csv_path: Ruta a ``reference_db/iedb/bcell_protective_epitopes.csv``.
        min_overlap: Longitud minima de solapamiento de subcadena para reportar un
            match. Para epitopos de referencia MAS CORTOS que este umbral, se exige
            el match completo del epitopo entero (nunca un umbral mas laxo).

    Returns:
        DataFrame con una fila por (candidato, epitopo de referencia) que solapa lo
        suficiente, columnas ``_OUTPUT_COLUMNS``. Vacio si no hay ningun match o
        ``sequences`` esta vacio.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    iedb_df = _load_iedb_epitopes(iedb_csv_path)
    if iedb_df.empty:
        logger.warning("'%s' no aporto ningun epitopo utilizable.", iedb_csv_path)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    rows = []
    for seq in sequences:
        seq_upper = seq.upper()
        for ref in iedb_df.itertuples(index=False):
            required_overlap = min(min_overlap, len(ref.epitope_sequence))
            match_len = _longest_common_substring_len(seq_upper, ref.epitope_sequence)
            if match_len < required_overlap:
                continue

            rows.append({
                "sequence": seq,
                "epitope_sequence": ref.epitope_sequence,
                "match_length": match_len,
                "source_organism": ref.source_organism,
                "response_measured": ref.response_measured,
                "qualitative_measure": ref.qualitative_measure,
                "method": ref.method,
                "host": ref.host,
                "pmid": ref.pmid,
            })

    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS) if rows else pd.DataFrame(columns=_OUTPUT_COLUMNS)


_MAX_CONSOLE_ROWS = 30
_MATCH_ANSI_START = "\033[1;33m"
_MATCH_ANSI_END = "\033[0m"
_SEQ_WRAP = 40


def _wrap_sequence(sequence: str, wrap: int):
    if not sequence:
        return [(0, "")]
    return [(i, sequence[i:i + wrap]) for i in range(0, len(sequence), wrap)]


def _highlight_chunk(padded_cell: str, chunk_offset: int, chunk_len: int, span) -> str:
    if span is None:
        return padded_cell
    local_start = max(span[0], chunk_offset) - chunk_offset
    local_end = min(span[1], chunk_offset + chunk_len) - chunk_offset
    if local_start >= local_end:
        return padded_cell
    return f"{padded_cell[:local_start]}{_MATCH_ANSI_START}{padded_cell[local_start:local_end]}{_MATCH_ANSI_END}{padded_cell[local_end:]}"


def _truncate(text: str, max_len: int) -> str:
    if pd.isna(text) or text == "":
        return "-"
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _short_response(value, max_len: int) -> str:
    """Bucketiza 'response_measured' a 'Neutralization'/'Protection' en vez de truncar.

    El dataset local (``reference_db/iedb/bcell_protective_epitopes.csv``) solo
    tiene 10 valores distintos para este campo: ``'neutralization'`` (exacto,
    2964/3429 filas) y 9 variantes que arrancan con ``'protection'`` pero
    difieren en un calificador largo (p. ej. ``'protection from pathogen
    challenge after adoptive transfer'``, 60 caracteres) que no aporta nada a
    un vistazo rapido en consola y forzaba truncado con "…" incluso a un ancho
    generoso. El valor completo sigue integro en el CSV persistido.
    """
    if pd.isna(value) or value == "":
        return "-"
    text = str(value)
    if text == "neutralization":
        return "Neutralization"
    if text.startswith("protection"):
        return "Protection"
    return _truncate(text, max_len)  # valor inesperado fuera del dataset local conocido


def print_iedb_crossref_report(report_df: pd.DataFrame, csv_path: Optional[Path] = None) -> None:
    """Imprime el detalle del cruce con IEDB: analogo a ``lanl_catnap_engine.print_bnab_crossref_report``.

    Trunca la consola a ``_MAX_CONSOLE_ROWS`` filas (priorizadas por
    ``qualitative_measure`` con mas evidencia -- 'Positive-High' primero --
    luego por mayor ``match_length``), igual mecanismo que Fase 6: el CSV
    persistido siempre tiene el 100% de los matches.

    ``Respuesta`` bucketiza a 'Neutralization'/'Protection' (``_short_response``)
    en vez de truncar con "…" -- ver su docstring, cubre el 100% de los 10
    valores distintos que trae el dataset local de IEDB. ``Organismo`` sigue
    truncando (``_ORG_MAX``, ahora 50 en vez de 28): a diferencia de
    ``response_measured``, source_organism tiene 561 valores distintos sin un
    patron limpio para bucketizar (solo 67/561 tienen un parentesis del que
    cortar; el resto es sufijo de cepa/aislado libre, ej. fechas/codigos de
    secuenciacion) -- 50 cubre ~90% de esos 561 sin "…" (percentil 90 = 47
    caracteres), el resto sigue disponible integro en el CSV.
    """
    if report_df.empty:
        print("Ningun peptido coincide con una region IEDB de proteccion/neutralizacion documentada.")
        return

    _ORG_MAX = 50
    _RESP_MAX = 20

    org_width = max(10, report_df["source_organism"].apply(lambda v: len(_truncate(v, _ORG_MAX))).max() + 2)
    resp_width = max(10, report_df["response_measured"].apply(lambda v: len(_short_response(v, _RESP_MAX))).max() + 2)

    seq_col = Column("Secuencia", lambda r: r.sequence, _SEQ_WRAP, "<")
    rest_columns = [
        Column("Organismo", lambda r: _truncate(r.source_organism, _ORG_MAX), org_width, "<", prefix="  "),
        Column("Respuesta", lambda r: _short_response(r.response_measured, _RESP_MAX), resp_width, "<"),
        Column("Medida", lambda r: r.qualitative_measure if pd.notna(r.qualitative_measure) else "-", 20, "<"),
        Column("PMID", lambda r: str(int(float(r.pmid))) if pd.notna(r.pmid) else "-", 10, ">"),
    ]

    header_line = "".join(c._cell(c.header) for c in [seq_col] + rest_columns)
    separator = "-" * len(header_line)
    blank_rest = "".join(c._cell("") for c in rest_columns)

    def _print_header() -> None:
        print(header_line)
        print(separator)

    n_total = len(report_df)
    if n_total > _MAX_CONSOLE_ROWS:
        priority = report_df["qualitative_measure"].apply(lambda v: 1 if v == "Positive-High" else 0)
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
    n_organisms = report_df["source_organism"].nunique()
    print(f"\nResumen Fase 6c: {n_candidates} peptido(s) coinciden con >=1 region documentada de IEDB "
          f"({n_organisms} organismo(s) distinto(s) representado(s) en los matches).")
