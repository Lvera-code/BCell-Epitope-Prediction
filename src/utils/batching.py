"""Construccion de mini-lotes dinamicos acotados por presupuesto de residuos.

En un entorno de 16 GB de RAM sin GPU, un tamano de lote fijo es peligroso: un
lote de N secuencias cortas consume una fraccion minima de memoria comparado
con un lote de N secuencias largas, ya que el costo de un forward pass (tanto
en la 1D-CNN como en ESM-2) escala con ``N * L^2`` o ``N * L`` segun la capa.
Esta utilidad agrupa una lista de items (con una funcion de longitud asociada)
en lotes cuyo costo total en residuos no supera un presupuesto configurado,
garantizando un techo de memoria predecible independientemente de la
distribucion de longitudes del FASTA de entrada.
"""

from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")


def dynamic_batches(
    items: Sequence[T],
    length_fn: Callable[[T], int],
    max_residues_per_batch: int,
    max_items_per_batch: int,
) -> List[List[T]]:
    """Agrupa ``items`` en mini-lotes acotados por longitud total y por conteo.

    Los items se ordenan por longitud ascendente antes de agrupar, de forma que
    cada lote contenga secuencias de longitud similar. Esto minimiza el
    padding desperdiciado (y por tanto el computo y memoria desperdiciados)
    cuando los tensores se acolchan (pad) a la longitud maxima del lote.

    Args:
        items: Secuencia de elementos a agrupar (p. ej. ``SequenceRecord``).
        length_fn: Funcion que extrae la longitud relevante (en residuos) de
            cada item.
        max_residues_per_batch: Presupuesto maximo de ``longitud_max * n_items``
            (area acolchada) tolerado por lote.
        max_items_per_batch: Limite duro adicional de items por lote,
            independiente del presupuesto de residuos (evita lotes de miles de
            secuencias muy cortas que saturarian el overhead de Python/tokenizer).

    Returns:
        Lista de lotes (listas de items), preservando el orden ascendente de
        longitud entre lotes pero no el orden original de ``items``.

    Raises:
        ValueError: Si ``max_residues_per_batch`` o ``max_items_per_batch`` no
            son positivos.
    """
    if max_residues_per_batch <= 0:
        raise ValueError("max_residues_per_batch debe ser un entero positivo.")
    if max_items_per_batch <= 0:
        raise ValueError("max_items_per_batch debe ser un entero positivo.")

    ordered = sorted(items, key=length_fn)

    batches: List[List[T]] = []
    current_batch: List[T] = []
    current_max_len = 0

    for item in ordered:
        item_len = length_fn(item)
        prospective_max_len = max(current_max_len, item_len)
        prospective_count = len(current_batch) + 1
        prospective_padded_area = prospective_max_len * prospective_count

        would_exceed_residues = prospective_padded_area > max_residues_per_batch
        would_exceed_items = prospective_count > max_items_per_batch

        if current_batch and (would_exceed_residues or would_exceed_items):
            batches.append(current_batch)
            current_batch = [item]
            current_max_len = item_len
        else:
            current_batch.append(item)
            current_max_len = prospective_max_len

    if current_batch:
        batches.append(current_batch)

    return batches
