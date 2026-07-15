"""Fase 3 (tabla 3 de 3): union logica anotada de regiones de epitopo entre BepiPred-3.0 y EpiDope.

Directiva de "Union Logica Anotada" (reemplaza el criterio de interseccion
anterior, que descartaba toda region sin coincidencia exacta entre motores):

* Preservacion de datos: TODA region detectada por BepiPred y/o por EpiDope
  avanza a la Fase 4, no solo las que coinciden entre ambos.
* Fusion de solapamientos: si una region de BepiPred y una de EpiDope
  SOLAPAN (comparten al menos un residuo), no se recortan a la interseccion:
  se fusionan tomando el ``start`` minimo y el ``end`` maximo de ambas,
  preservando el peptido completo (incluye fusion transitiva: una cadena de
  regiones A-B-C encadenadas por solapamientos sucesivos se fusiona en una
  sola region final, aunque A y C por si solas no se solapen).
* Etiquetado analitico: cada region final queda marcada en la columna
  ``origen`` como ``'Consenso'`` (fusion de ambos motores), ``'BepiPred'`` o
  ``'EpiDope'`` (un solo motor, sin solapamiento con el otro).

Como la fusion transitiva puede extender una region final mas alla del span
detectado por cualquiera de los motores por separado, la subsecuencia de
cada region final se reconstruye desde un lookup de secuencia completa por
accession (``sequence_lookup``, ver
``src.engines.epitope_mapping.build_sequence_lookup``) en vez de recortar
las subsecuencias individuales que ya trae cada motor.

Filtro de longitud INQUEBRANTABLE (``MIN_FINAL_PEPTIDE_LENGTH``): antes de
devolver la tabla final se descarta cualquier region con longitud resultante
menor a 9 aminoacidos -el mismo footprint minimo de union a MHC-II que exige
NetMHCIIpan en la Fase 5 (ver ``_MIN_PEPTIDE_LENGTH`` en
``netmhciipan_engine.py``)-, evitando gastar BLASTp (Fase 4) en peptidos que
de todas formas no podrian evaluarse en la Fase 5. Deliberadamente NO es
configurable via ``Settings``/env var ni flag de CLI: es un piso biologico
fijo, no un parametro ajustable.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.utils.table_format import Column, print_fixed_width_table

MIN_FINAL_PEPTIDE_LENGTH = 9

_UNION_COLUMNS = [
    "accession", "start", "end", "length", "sequence", "origen",
    "bepipred_score", "epidope_score", "bepipred_region", "epidope_region",
]


def accession_id(accession: str) -> str:
    """Normaliza un accession a su primer token separado por espacios.

    BepiPred-3.0 conserva la cabecera FASTA completa como accession, mientras
    que EpiDope (via su propio CLI, flags ``--idpos 0 --delim ' '`` por
    defecto) solo conserva el primer token como ID de gen. Sin esta
    normalizacion el cruce entre motores nunca encuentra coincidencias para
    cualquier cabecera con mas de una palabra.
    """
    return accession.split()[0] if accession else accession


def _merge_accession_intervals(
    bp_group: Optional[pd.DataFrame], ed_group: Optional[pd.DataFrame], full_sequence: str
) -> List[dict]:
    """Fusiona por solapamiento los intervalos de BepiPred y EpiDope de UNA accession.

    Barrido de linea estandar: junta los intervalos de ambos motores, los
    ordena por ``start`` y fusiona los que comparten al menos un residuo
    (``next.start <= current_end``), sin importar de que motor provienen.
    Cada grupo fusionado conserva, por motor contribuyente, el promedio de
    ``mean_score`` y la lista de regiones originales (coordenadas
    ``start-end``) para trazabilidad.
    """
    intervals = []
    if bp_group is not None:
        for row in bp_group.itertuples(index=False):
            intervals.append((row.start, row.end, "BepiPred", row.mean_score))
    if ed_group is not None:
        for row in ed_group.itertuples(index=False):
            intervals.append((row.start, row.end, "EpiDope", row.mean_score))

    if not intervals:
        return []

    intervals.sort(key=lambda iv: (iv[0], iv[1]))

    def _new_bucket_map(start: int, end: int, source: str, score: float) -> dict:
        return {source: {"scores": [score], "regions": [f"{start}-{end}"]}}

    first_start, first_end, first_source, first_score = intervals[0]
    merged_groups = [[first_start, first_end, _new_bucket_map(first_start, first_end, first_source, first_score)]]

    for start, end, source, score in intervals[1:]:
        group = merged_groups[-1]
        if start <= group[1]:  # solapamiento: comparte al menos un residuo con el grupo abierto
            group[1] = max(group[1], end)
            bucket = group[2].setdefault(source, {"scores": [], "regions": []})
            bucket["scores"].append(score)
            bucket["regions"].append(f"{start}-{end}")
        else:
            merged_groups.append([start, end, _new_bucket_map(start, end, source, score)])

    records = []
    for group_start, group_end, members in merged_groups:
        sources = set(members.keys())
        if sources == {"BepiPred", "EpiDope"}:
            origen = "Consenso"
        elif sources == {"BepiPred"}:
            origen = "BepiPred"
        else:
            origen = "EpiDope"

        bp_info = members.get("BepiPred")
        ed_info = members.get("EpiDope")
        length = group_end - group_start + 1
        records.append(
            {
                "start": group_start,
                "end": group_end,
                "length": length,
                "sequence": full_sequence[group_start - 1 : group_end] if full_sequence else "",
                "origen": origen,
                "bepipred_score": (sum(bp_info["scores"]) / len(bp_info["scores"])) if bp_info else float("nan"),
                "epidope_score": (sum(ed_info["scores"]) / len(ed_info["scores"])) if ed_info else float("nan"),
                "bepipred_region": ";".join(bp_info["regions"]) if bp_info else "",
                "epidope_region": ";".join(ed_info["regions"]) if ed_info else "",
            }
        )
    return records


def build_annotated_union_table(
    bepipred_df: pd.DataFrame,
    epidope_df: pd.DataFrame,
    sequence_lookup: Dict[str, str],
    min_length: int = MIN_FINAL_PEPTIDE_LENGTH,
) -> pd.DataFrame:
    """Une (no interseca) las regiones de epitopo de BepiPred y EpiDope, fusionando solapes.

    Args:
        bepipred_df: Salida de ``bepipred_engine.extract_epitopes`` (columnas
            ``accession``, ``start``, ``end``, ``sequence``, ``mean_score``, ...).
        epidope_df: Salida de ``epidope_engine.extract_epitopes``, mismo esquema.
        sequence_lookup: ``accession -> secuencia completa``, ver
            ``src.engines.epitope_mapping.build_sequence_lookup``. Necesario
            porque una region fusionada puede exceder el span original de
            cualquiera de los dos motores por separado.
        min_length: Filtro de longitud inquebrantable aplicado al final
            (``MIN_FINAL_PEPTIDE_LENGTH`` por defecto, 9 aa).

    Returns:
        DataFrame con una fila por region final (fusionada o individual):
        ``accession``, ``start``/``end``/``length``, ``sequence``,
        ``origen`` (``'Consenso'``/``'BepiPred'``/``'EpiDope'``),
        ``bepipred_score``/``epidope_score`` (promedio de ``mean_score`` de
        las regiones de origen que se fusionaron; ``NaN`` si ese motor no
        contribuyo a la region) y ``bepipred_region``/``epidope_region``
        (coordenadas ``start-end`` de cada region de origen, separadas por
        ``;`` si se fusiono mas de una del mismo motor, para trazabilidad).
        Filas con ``length < min_length`` quedan excluidas.
    """
    if bepipred_df.empty and epidope_df.empty:
        return pd.DataFrame(columns=_UNION_COLUMNS)

    bp_by_id: Dict[str, pd.DataFrame] = (
        {aid: g for aid, g in bepipred_df.groupby(bepipred_df["accession"].map(accession_id), sort=False)}
        if not bepipred_df.empty
        else {}
    )
    ed_by_id: Dict[str, pd.DataFrame] = (
        {aid: g for aid, g in epidope_df.groupby(epidope_df["accession"].map(accession_id), sort=False)}
        if not epidope_df.empty
        else {}
    )
    all_ids = list(dict.fromkeys(list(bp_by_id.keys()) + list(ed_by_id.keys())))

    records = []
    for accession in all_ids:
        full_sequence = sequence_lookup.get(accession, "")
        for rec in _merge_accession_intervals(bp_by_id.get(accession), ed_by_id.get(accession), full_sequence):
            rec["accession"] = accession
            records.append(rec)

    union_df = pd.DataFrame.from_records(records, columns=_UNION_COLUMNS)
    if union_df.empty:
        return union_df

    union_df = union_df[union_df["length"] >= min_length].reset_index(drop=True)
    union_df = union_df.sort_values(["accession", "start", "end"]).reset_index(drop=True)
    return union_df


def print_union_table(union_df: pd.DataFrame) -> None:
    """Imprime la tabla de union anotada (BepiPred / EpiDope / Consenso) en consola."""
    if union_df.empty:
        print(
            f"No se encontraron regiones de epitopo (BepiPred y/o EpiDope) de al menos "
            f"{MIN_FINAL_PEPTIDE_LENGTH} aa tras la union."
        )
        return

    def _score(value: float) -> str:
        return f"{value:.4f}" if pd.notna(value) else "-"

    columns = [
        Column("accession", lambda r: r.accession, 28, "<"),
        Column("start", lambda r: str(r.start), 7, ">"),
        Column("end", lambda r: str(r.end), 7, ">"),
        Column("len", lambda r: str(r.length), 6, ">"),
        Column("origen", lambda r: r.origen, 10, "<", prefix="  "),
        Column("bp_score", lambda r: _score(r.bepipred_score), 10, ">"),
        Column("ed_score", lambda r: _score(r.epidope_score), 10, ">"),
        Column("sequence", lambda r: r.sequence, 0, "<", prefix="  "),
    ]
    print_fixed_width_table(union_df.itertuples(index=False), columns)

    n_consenso = int((union_df["origen"] == "Consenso").sum())
    n_bepipred = int((union_df["origen"] == "BepiPred").sum())
    n_epidope = int((union_df["origen"] == "EpiDope").sum())
    print(
        f"\nResumen Fase 3: {len(union_df)} region(es) tras filtro de longitud (>= "
        f"{MIN_FINAL_PEPTIDE_LENGTH} aa) -> {n_consenso} Consenso, {n_bepipred} solo BepiPred, "
        f"{n_epidope} solo EpiDope."
    )
