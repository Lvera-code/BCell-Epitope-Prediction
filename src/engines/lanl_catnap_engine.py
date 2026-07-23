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
from typing import List, Optional

import pandas as pd

from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

_OUTPUT_COLUMNS = [
    "sequence", "antibody_name", "epitope_sequence", "match_length", "epitope_name",
    "hxb2_location", "neutralizing", "antibody_type", "binding_region",
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
                "Neutralizing", "Antibody type", "Binding region"]
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


def print_bnab_crossref_report(report_df: pd.DataFrame) -> None:
    """Imprime el detalle del cruce con bnAb: analogo a ``algpred_engine.print_allergenicity_report``."""
    if report_df.empty:
        print("Ningun peptido coincide con un epitopo lineal de bnAb conocido (esperado si la entrada no es HIV Env).")
        return

    seq_width = max(30, report_df["sequence"].str.len().max() + 2)
    ab_width = max(14, report_df["antibody_name"].str.len().max() + 2)
    epitope_width = max(20, report_df["epitope_sequence"].str.len().max() + 2)
    columns = [
        Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
        Column("Anticuerpo", lambda r: r.antibody_name, ab_width, "<"),
        Column("Epitopo bnAb", lambda r: r.epitope_sequence, epitope_width, "<"),
        Column("Match(aa)", lambda r: str(r.match_length), 10, ">"),
        Column("Neutralizante", lambda r: r.neutralizing, 14, ">"),
        Column("IC50 medio", lambda r: f"{r.catnap_mean_ic50:.3f}" if pd.notna(r.catnap_mean_ic50) else "-", 11, ">"),
        Column("N virus", lambda r: str(int(r.catnap_n_viruses)) if pd.notna(r.catnap_n_viruses) else "-", 8, ">"),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns)

    n_candidates = report_df["sequence"].nunique()
    n_neutralizing = report_df[report_df["neutralizing"] == "yes"]["sequence"].nunique()
    print(f"\nResumen Fase 6: {n_candidates} peptido(s) coinciden con >=1 epitopo de bnAb conocido "
          f"({n_neutralizing} de ellos con >=1 anticuerpo neutralizante confirmado).")
