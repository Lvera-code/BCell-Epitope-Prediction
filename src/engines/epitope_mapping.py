"""Logica generica de Fase 3: mapeo de regiones de epitopo por ventana deslizante.

Extraido de ``bepipred_engine.py`` para reutilizarlo tal cual entre motores de
antigenicidad distintos (BepiPred-3.0, EpiDope): ambos producen un DataFrame
de scores crudos por residuo (una fila por residuo, agrupable por accession),
y esta misma ventana deslizante tolerante a gaps se aplica sobre cualquiera de
los dos, solo cambiando el nombre de columna de score/accession/residuo.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def resolve_residue_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    logger.warning(
        "No se encontro ninguna columna de residuo entre %s. Columnas disponibles: %s. "
        "Las regiones de epitopo se reportaran sin secuencia de aminoacidos.",
        candidates,
        list(df.columns),
    )
    return None


def find_valid_windows(
    scores: List[float], threshold: float, window_size: int, max_gap_residues: int
) -> List[Tuple[int, int]]:
    """Desliza una ventana de ``window_size`` (paso=1) y devuelve los rangos validos.

    Una ventana ``[i, i + window_size - 1]`` (0-indexada, inclusive) es valida
    si, a la vez: (a) a lo sumo ``max_gap_residues`` de sus residuos tienen un
    score individual por debajo de ``threshold``, y (b) el score medio de la
    ventana completa es ``>= threshold``.

    Returns:
        Lista de rangos ``(start, end)`` validos, en orden de aparicion (sin
        fusionar todavia).
    """
    n = len(scores)
    valid_windows = []
    for i in range(0, n - window_size + 1):
        window = scores[i : i + window_size]
        below_count = sum(1 for score in window if score < threshold)
        if below_count <= max_gap_residues and (sum(window) / window_size) >= threshold:
            valid_windows.append((i, i + window_size - 1))
    return valid_windows


def merge_overlapping_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Fusiona ventanas validas solapadas o adyacentes en regiones continuas.

    Asume ``windows`` ordenado por posicion de inicio (garantizado por el
    barrido secuencial de :func:`find_valid_windows`).
    """
    if not windows:
        return []

    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:  # solapada o adyacente
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def extract_epitope_regions(
    raw_scores_df: pd.DataFrame,
    accession_col: str,
    score_col: str,
    residue_col_candidates: Sequence[str],
    threshold: float,
    min_length: int,
    window_size: int,
    max_gap_residues: int,
) -> pd.DataFrame:
    """Mapea regiones de epitopo con una ventana deslizante tolerante a gaps.

    Logica 100% local, generica sobre cualquier DataFrame de scores crudos por
    residuo (una fila por residuo, agrupable por ``accession_col``). La
    posicion de cada residuo se deriva del orden de las filas dentro de cada
    accession (1-indexado), no de una columna de posicion.

    Algoritmo (por accession, paso de la ventana = 1 residuo):

    1. Para cada posicion, evalua la ventana de ``window_size`` residuos que
       comienza ahi: es "valida" si a lo sumo ``max_gap_residues`` residuos
       individuales caen por debajo de ``threshold`` Y el score medio de la
       ventana completa es ``>= threshold``.
    2. Las ventanas validas que se solapan o son adyacentes se fusionan en
       una unica region antigenica continua.
    3. Cada region fusionada se reporta como epitopo si su longitud final es
       ``>= min_length``.

    Returns:
        DataFrame con columnas ``accession``, ``start``, ``end``, ``length``,
        ``mean_score``, ``max_score`` y ``sequence``.
    """
    missing = {accession_col, score_col} - set(raw_scores_df.columns)
    if missing:
        raise ValueError(
            f"El DataFrame de entrada no contiene las columnas requeridas {sorted(missing)}. "
            f"Columnas encontradas: {list(raw_scores_df.columns)}."
        )

    residue_col = resolve_residue_column(raw_scores_df, residue_col_candidates)
    records = []

    for accession, group in raw_scores_df.groupby(accession_col, sort=False):
        group = group.reset_index(drop=True)
        scores = group[score_col].tolist()

        valid_windows = find_valid_windows(scores, threshold, window_size, max_gap_residues)
        merged_regions = merge_overlapping_windows(valid_windows)

        for start, end in merged_regions:
            length = end - start + 1
            if length < min_length:
                continue

            block = group.iloc[start : end + 1]
            sequence = "".join(block[residue_col].astype(str)) if residue_col else ""
            records.append(
                {
                    "accession": accession,
                    "start": start + 1,
                    "end": end + 1,
                    "length": length,
                    "mean_score": float(block[score_col].mean()),
                    "max_score": float(block[score_col].max()),
                    "sequence": sequence,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=["accession", "start", "end", "length", "mean_score", "max_score", "sequence"],
    )


def build_sequence_lookup(
    raw_scores_df: pd.DataFrame, accession_col: str, residue_col_candidates: Sequence[str]
) -> Dict[str, str]:
    """Reconstruye la secuencia completa por accession a partir de scores crudos por residuo.

    Concatena la columna de residuo en el orden de las filas de cada grupo
    (el mismo orden posicional 1-indexado que usa :func:`extract_epitope_regions`
    para calcular ``start``/``end``). Usado por ``src.engines.consensus`` para
    reconstruir la subsecuencia de regiones FUSIONADAS cuyo span final puede
    exceder el intervalo detectado por un unico motor por separado.

    Returns:
        Diccionario ``accession -> secuencia completa``. Vacio si no se pudo
        resolver la columna de residuo (ver :func:`resolve_residue_column`).
    """
    residue_col = resolve_residue_column(raw_scores_df, residue_col_candidates)
    if residue_col is None:
        return {}
    return {
        accession: "".join(group[residue_col].astype(str))
        for accession, group in raw_scores_df.groupby(accession_col, sort=False)
    }


def print_epitope_table(epitopes_df: pd.DataFrame, empty_message: str) -> None:
    """Imprime una tabla de regiones de epitopo en consola."""
    if epitopes_df.empty:
        print(empty_message)
        return

    header = f"{'accession':<28}{'start':>7}{'end':>7}{'len':>6}{'mean':>8}{'max':>8}  sequence"
    print(header)
    print("-" * len(header))
    for row in epitopes_df.itertuples(index=False):
        print(
            f"{row.accession:<28}{row.start:>7}{row.end:>7}{row.length:>6}"
            f"{row.mean_score:>8.4f}{row.max_score:>8.4f}  {row.sequence}"
        )
