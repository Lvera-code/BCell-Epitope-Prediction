"""Fase 3 (tabla final): union logica anotada de regiones de epitopo entre N motores.

Generalizacion de la union original (BepiPred U EpiDope, 2 motores fijos) a
un numero arbitrario de motores de antigenicidad (2, 3 o 4 segun el camino de
entrada -- FASTA puro, PDB en modo 'structure_only', o PDB en modo
'structure_and_sequence', ver ``pipeline.py``/``src.engines.engine_registry``).

Directiva de "Union Logica Anotada" (sin cambios respecto al diseño
original):

* Preservacion de datos: TODA region detectada por CUALQUIERA de los motores
  avanza a la Fase 4, no solo las que coinciden entre varios.
* Fusion de solapamientos: si regiones de motores distintos SOLAPAN
  (comparten al menos un residuo), no se recortan a la interseccion: se
  fusionan tomando el ``start`` minimo y el ``end`` maximo de todas,
  preservando el peptido completo (incluye fusion transitiva: una cadena de
  regiones A-B-C encadenadas por solapamientos sucesivos se fusiona en una
  sola region final, aunque A y C por si solas no se solapen).
* Etiquetado analitico: cada region final queda marcada en la columna
  ``origen`` segun sus motores contribuyentes, usando abreviaturas de 2
  letras (ver ``_origen_label``): ``Bp`` (BepiPred), ``Ed`` (EpiDope), ``Dt``
  (DiscoTope-3.0), ``Sn`` (ScanNet). Un unico motor se reporta solo (p. ej.
  ``'Bp'``); dos o tres motores se unen con ``'+'`` en el mismo orden en que
  aparecen las claves de ``engine_dfs`` (p. ej. ``'Bp+Ed'``, ``'Dt+Sn'``,
  ``'Bp+Dt'``, ``'Bp+Ed+Dt'``) -- esto permite distinguir CUALQUIER
  combinacion parcial sin ambiguedad, independientemente de cuales sean.
  Unica excepcion: cuando los 4 motores contribuyen a la vez, la etiqueta es
  ``'Consenso total'`` en vez de ``'Bp+Ed+Dt+Sn'``.

Como la fusion transitiva puede extender una region final mas alla del span
detectado por cualquiera de los motores por separado, la subsecuencia de
cada region final se reconstruye desde un lookup de secuencia completa por
accession (``sequence_lookup``, ver
``src.engines.epitope_mapping.build_sequence_lookup`` para motores de
secuencia y ``src.utils.structure_parser.StructureRecord.sequence`` para
motores estructurales) en vez de recortar las subsecuencias individuales que
ya trae cada motor.

ADR -- coordenadas de motores estructurales y ``position_mapping``
----------------------------------------------------------------------
DiscoTope-3.0 y ScanNet corren sobre el PDB de una sola cadena que escribe
Fase 1.5 (``StructureRecord.chain_pdb_path``) y ambos re-numeran sus
resultados desde 1 sobre esa misma cadena (documentado en el propio README de
DiscoTope-3.0: "Relative residue index, re-numerado desde 1"). Esa numeracion
es, por construccion, la MISMA convencion posicional que
``StructureRecord.position_mapping.fasta_position`` (tambien secuencial desde
1 sobre la cadena elegida): en el caso esperado, ``start``/``end`` de un
motor estructural YA coinciden con la posicion en ``sequence_lookup`` sin
necesitar ninguna transformacion aritmetica. ``position_mapping`` se usa aqui
como una VERIFICACION de esa suposicion, no como una tabla de traduccion
aritmetica: si algun motor reporta coordenadas mas alla de la longitud de la
secuencia derivada (senal de que ese motor parseo un subconjunto de residuos
distinto al que uso ``structure_parser``, p. ej. por una regla de backbone
incompleto distinta), se loggea un WARNING explicito en vez de fallar en
silencio o fabricar una traduccion no verificada. CONFIRMADO empiricamente
(2026-07-20, PDB real 7c4s vs. instalacion real de DiscoTope-3.0/ScanNet): la
numeracion de ambos motores coincidio exactamente con ``fasta_position`` en
las 282 posiciones de la cadena de prueba, sin disparar ningun warning de
este chequeo. Sigue siendo una unica estructura de ejemplo -- si una
estructura distinta dispara este warning, este es el punto exacto del codigo
a revisar.
"""

from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

MIN_FINAL_PEPTIDE_LENGTH = 9

_BASE_COLUMNS = ["accession", "start", "end", "length", "sequence", "origen"]

# Nombres de presentacion para la columna 'origen'. Motores no listados aqui
# (cualquier clave nueva agregada a ENGINE_REGISTRY en el futuro) caen a
# `key.capitalize()` como fallback razonable, ver `_display_name`.
_ENGINE_DISPLAY_NAMES = {
    "bepipred": "BepiPred",
    "epidope": "EpiDope",
    "discotope": "DiscoTope-3.0",
    "scannet": "ScanNet",
}

# Abreviaturas de 2 letras para la columna 'origen' (ver `_origen_label`).
# Motores no listados aqui (cualquier clave nueva agregada a ENGINE_REGISTRY
# en el futuro) caen a `key[:2].capitalize()` como fallback razonable.
_ENGINE_ABBREVIATIONS = {
    "bepipred": "Bp",
    "epidope": "Ed",
    "discotope": "Dt",
    "scannet": "Sn",
}

# Conjunto de motores para el que 'origen' usa la etiqueta especial
# 'Consenso total' en vez de las abreviaturas unidas por '+' (ver
# `_origen_label`). Fijo a los 4 motores actuales de ENGINE_REGISTRY.
_ALL_ENGINES = frozenset({"bepipred", "epidope", "discotope", "scannet"})


def _engine_abbreviation(engine_key: str) -> str:
    return _ENGINE_ABBREVIATIONS.get(engine_key, engine_key[:2].capitalize())


def _origen_label(contributing_keys: Sequence[str]) -> str:
    """Etiqueta de 'origen' para un conjunto de motores contribuyentes.

    ``'Consenso total'`` si contribuyen EXACTAMENTE los 4 motores de
    ``_ALL_ENGINES``; si no, las abreviaturas de 2 letras de los motores
    contribuyentes unidas por ``'+'``, preservando el orden de
    ``contributing_keys`` (un unico motor se reporta solo, sin '+').
    """
    if frozenset(contributing_keys) == _ALL_ENGINES:
        return "Consenso total"
    return "+".join(_engine_abbreviation(key) for key in contributing_keys)


def _display_name(engine_key: str) -> str:
    return _ENGINE_DISPLAY_NAMES.get(engine_key, engine_key.capitalize())


def accession_id(accession: str) -> str:
    """Normaliza un accession a su primer token separado por espacios.

    BepiPred-3.0 conserva la cabecera FASTA completa como accession, mientras
    que EpiDope (via su propio CLI, flags ``--idpos 0 --delim ' '`` por
    defecto) solo conserva el primer token como ID de gen. Los motores
    estructurales (DiscoTope-3.0/ScanNet) ya reportan un accession de un solo
    token (``Path(pdb_path).stem``, ver ``discotope_engine.py``/
    ``scannet_engine.py``), asi que esta normalizacion es un no-op para
    ellos. Sin esta normalizacion el cruce entre motores nunca encuentra
    coincidencias para cualquier cabecera con mas de una palabra.
    """
    return accession.split()[0] if accession else accession


def _build_columns(engine_keys: Sequence[str]) -> List[str]:
    per_engine: List[str] = []
    for key in engine_keys:
        per_engine += [f"{key}_score", f"{key}_region"]
    return _BASE_COLUMNS + per_engine


def _warn_if_out_of_bounds(
    accession: str,
    engine_groups: Dict[str, Optional[pd.DataFrame]],
    full_sequence: str,
) -> None:
    """Verifica que ningun motor reporte coordenadas mas alla de ``full_sequence``.

    Ver ADR del modulo: para motores estructurales esto es una verificacion
    de la suposicion "numeracion del motor == fasta_position", no una
    traduccion aritmetica.
    """
    if not full_sequence:
        return
    expected_len = len(full_sequence)
    for engine_key, group in engine_groups.items():
        if group is None or group.empty:
            continue
        max_end = int(group["end"].max())
        if max_end > expected_len:
            logger.warning(
                "Accession '%s': el motor '%s' reporta coordenadas hasta la posicion %d, mas alla "
                "de la longitud de la secuencia derivada (%d aa). Se asume que la numeracion de "
                "cada motor coincide 1:1 con 'fasta_position' (ver ADR en consensus.py); esta "
                "discrepancia sugiere que el motor parseo un subconjunto de residuos distinto al "
                "que uso structure_parser para esta accession -- revisar manualmente antes de "
                "confiar en las coordenadas de esta region.",
                accession, _display_name(engine_key), max_end, expected_len,
            )


def _merge_accession_intervals(
    engine_groups: Dict[str, Optional[pd.DataFrame]], full_sequence: str
) -> List[dict]:
    """Fusiona por solapamiento los intervalos de TODOS los motores de UNA accession.

    Barrido de linea estandar: junta los intervalos de todos los motores
    presentes en ``engine_groups``, los ordena por ``start`` y fusiona los
    que comparten al menos un residuo (``next.start <= current_end``), sin
    importar de que motor provienen. Cada grupo fusionado conserva, por motor
    contribuyente, el promedio de ``mean_score`` y la lista de regiones
    originales (coordenadas ``start-end``) para trazabilidad.
    """
    intervals = []
    for engine_key, group in engine_groups.items():
        if group is None:
            continue
        for row in group.itertuples(index=False):
            intervals.append((row.start, row.end, engine_key, row.mean_score))

    if not intervals:
        return []

    intervals.sort(key=lambda iv: (iv[0], iv[1]))

    def _new_bucket_map(start: int, end: int, engine_key: str, score: float) -> dict:
        return {engine_key: {"scores": [score], "regions": [f"{start}-{end}"]}}

    first_start, first_end, first_engine, first_score = intervals[0]
    merged_groups = [[first_start, first_end, _new_bucket_map(first_start, first_end, first_engine, first_score)]]

    for start, end, engine_key, score in intervals[1:]:
        group = merged_groups[-1]
        if start <= group[1]:  # solapamiento: comparte al menos un residuo con el grupo abierto
            group[1] = max(group[1], end)
            bucket = group[2].setdefault(engine_key, {"scores": [], "regions": []})
            bucket["scores"].append(score)
            bucket["regions"].append(f"{start}-{end}")
        else:
            merged_groups.append([start, end, _new_bucket_map(start, end, engine_key, score)])

    engine_order = list(engine_groups.keys())
    records = []
    for group_start, group_end, members in merged_groups:
        contributing = [key for key in engine_order if key in members]
        origen = _origen_label(contributing)
        length = group_end - group_start + 1

        record = {
            "start": group_start,
            "end": group_end,
            "length": length,
            "sequence": full_sequence[group_start - 1 : group_end] if full_sequence else "",
            "origen": origen,
        }
        for engine_key in engine_order:
            info = members.get(engine_key)
            record[f"{engine_key}_score"] = (sum(info["scores"]) / len(info["scores"])) if info else float("nan")
            record[f"{engine_key}_region"] = ";".join(info["regions"]) if info else ""
        records.append(record)
    return records


def build_annotated_union_table(
    engine_dfs: Dict[str, pd.DataFrame],
    sequence_lookup: Dict[str, str],
    position_mapping: Optional[pd.DataFrame] = None,
    min_length: int = MIN_FINAL_PEPTIDE_LENGTH,
) -> pd.DataFrame:
    """Une (no interseca) las regiones de epitopo de N motores, fusionando solapes.

    Funciona correctamente con cualquier subconjunto no vacio de motores: los
    3 escenarios soportados por el pipeline son ``{bepipred, epidope}``
    (Camino 1), ``{discotope, scannet}`` (Camino 2, solo estructurales -- NO
    asume que siempre hay al menos un motor de secuencia contribuyendo) y los
    4 motores juntos (Camino 3).

    Args:
        engine_dfs: Diccionario ``nombre_motor -> DataFrame`` de epitopos ya
            extraidos (salida de la ``extract_epitopes`` de cada motor:
            columnas ``accession``, ``start``, ``end``, ``mean_score``, ...).
            Las claves deben coincidir con las de ``ENGINE_REGISTRY``
            (``'bepipred'``, ``'epidope'``, ``'discotope'``, ``'scannet'``);
            el orden de las claves determina el orden en que se listan los
            motores contribuyentes en la columna ``origen``. Un DataFrame
            ``None`` o vacio para una clave se trata como "ese motor no
            corrio para ninguna accession" (no es un error).
        sequence_lookup: ``accession -> secuencia completa``. Para motores de
            secuencia, ver ``src.engines.epitope_mapping.build_sequence_lookup``;
            para input de estructura, ``StructureRecord.sequence`` (ATMSEQ).
            Necesario porque una region fusionada puede exceder el span
            detectado por cualquiera de los motores por separado.
        position_mapping: Opcional. ``DataFrame`` de
            ``StructureRecord.position_mapping`` (una fila por accession de
            tipo estructura), usado UNICAMENTE para verificar que las
            coordenadas de motores estructurales no excedan la longitud de la
            secuencia derivada (ver ADR del modulo) -- no se usa para
            traducir coordenadas aritmeticamente. Si es ``None``, se omite
            esa verificacion.
        min_length: Filtro de longitud inquebrantable aplicado al final
            (``MIN_FINAL_PEPTIDE_LENGTH`` por defecto, 9 aa).

    Returns:
        DataFrame con una fila por region final (fusionada o individual):
        ``accession``, ``start``/``end``/``length``, ``sequence``,
        ``origen`` (ver :func:`_origen_label`: abreviaturas de 2 letras
        unidas por ``'+'`` -``'Bp'``, ``'Ed'``, ``'Dt'``, ``'Sn'``, o
        combinaciones como ``'Bp+Ed'``/``'Dt+Sn'``/``'Bp+Dt'``/etc.-, excepto
        ``'Consenso total'`` cuando contribuyen los 4 motores a la vez) y,
        por cada motor de ``engine_dfs``,
        ``{motor}_score``/``{motor}_region`` (``NaN``/vacio si ese motor no
        contribuyo a la region). Filas con ``length < min_length`` quedan
        excluidas.

    Raises:
        ValueError: Si ``engine_dfs`` esta vacio (se necesita al menos un
            motor para construir la union).
    """
    if not engine_dfs:
        raise ValueError("engine_dfs no puede estar vacio: se necesita al menos un motor de antigenicidad.")

    columns = _build_columns(engine_dfs.keys())

    by_engine_by_accession: Dict[str, Dict[str, pd.DataFrame]] = {}
    for engine_key, df in engine_dfs.items():
        if df is None or df.empty:
            by_engine_by_accession[engine_key] = {}
        else:
            by_engine_by_accession[engine_key] = {
                aid: group for aid, group in df.groupby(df["accession"].map(accession_id), sort=False)
            }

    all_ids = list(dict.fromkeys(aid for groups in by_engine_by_accession.values() for aid in groups.keys()))
    if not all_ids:
        return pd.DataFrame(columns=columns)

    records = []
    for accession in all_ids:
        full_sequence = sequence_lookup.get(accession, "")
        engine_groups = {key: by_engine_by_accession[key].get(accession) for key in engine_dfs.keys()}

        if position_mapping is not None:
            _warn_if_out_of_bounds(accession, engine_groups, full_sequence)

        for rec in _merge_accession_intervals(engine_groups, full_sequence):
            rec["accession"] = accession
            records.append(rec)

    union_df = pd.DataFrame.from_records(records, columns=columns)
    if union_df.empty:
        return union_df

    union_df = union_df[union_df["length"] >= min_length].reset_index(drop=True)
    union_df = union_df.sort_values(["accession", "start", "end"]).reset_index(drop=True)
    return union_df


def print_union_table(union_df: pd.DataFrame, engine_keys: Optional[Sequence[str]] = None) -> None:
    """Imprime la tabla de union anotada (motores contribuyentes por region) en consola.

    Args:
        union_df: Salida de :func:`build_annotated_union_table`.
        engine_keys: Orden explicito de las columnas ``{motor}_score`` a
            mostrar. Si es ``None`` (default), se derivan de las columnas
            presentes en ``union_df`` que terminan en ``'_score'``,
            preservando su orden de aparicion.
    """
    if union_df.empty:
        print(
            f"No se encontraron regiones de epitopo (de ningun motor) de al menos "
            f"{MIN_FINAL_PEPTIDE_LENGTH} aa tras la union."
        )
        return

    if engine_keys is None:
        engine_keys = [c[: -len("_score")] for c in union_df.columns if c.endswith("_score")]

    def _score(value: float) -> str:
        return f"{value:.4f}" if pd.notna(value) else "-"

    columns = [
        Column("accession", lambda r: r.accession, 28, "<"),
        Column("start", lambda r: str(r.start), 7, ">"),
        Column("end", lambda r: str(r.end), 7, ">"),
        Column("len", lambda r: str(r.length), 6, ">"),
        Column("origen", lambda r: r.origen, 18, "<", prefix="  "),
    ]
    for key in engine_keys:
        header = f"{key[:8]}_sc"
        columns.append(Column(header, lambda r, k=key: _score(getattr(r, f"{k}_score")), 10, ">", prefix="  "))
    columns.append(Column("sequence", lambda r: r.sequence, 0, "<", prefix="  "))

    print_fixed_width_table(union_df.itertuples(index=False), columns)

    origen_counts: Dict[str, int] = {}
    for value in union_df["origen"]:
        origen_counts[value] = origen_counts.get(value, 0) + 1
    summary = ", ".join(f"{count} {origen}" for origen, count in origen_counts.items())
    print(
        f"\nResumen Fase 3: {len(union_df)} region(es) tras filtro de longitud (>= "
        f"{MIN_FINAL_PEPTIDE_LENGTH} aa) -> {summary}."
    )
