"""Registry of Phase 2 engines and resolution of the active subset per input path.

The single source of truth about which engine consumes which input type (``'fasta'``
or ``'pdb'``) and about which engines run for each of the 3 paths that
the pipeline supports (see ``pipeline.py``):

    Path 1 (FASTA input)                                  -> bepipred + epidope
    Path 2 (PDB input, PDB_PROCESSING_MODE='structure_only')       -> discotope + scannet
    Path 3 (PDB input, PDB_PROCESSING_MODE='structure_and_sequence') -> all 4

``pipeline.py`` is the only consumer: it uses ``active_engines_for`` to
decide what runs in Phase 2, and ``ENGINE_REGISTRY`` to know whether each active
engine needs a FASTA (``clean_fasta``) or a single-chain PDB
(``StructureRecord.chain_pdb_path``) as input.
"""

from typing import Dict, List, Optional, Tuple, Type

from src.engines.base_engine import BaseEngine
from src.engines.bepipred_engine import BepiPredEngine
from src.engines.discotope_engine import DiscoTopeEngine
from src.engines.epidope_engine import EpidopeEngine
from src.engines.scannet_engine import ScanNetEngine
from src.utils.exceptions import InputRoutingError

# nombre_motor -> (clase del motor, tipo de input que consume: 'fasta' | 'pdb')
ENGINE_REGISTRY: Dict[str, Tuple[Type[BaseEngine], str]] = {
    "bepipred": (BepiPredEngine, "fasta"),
    "epidope": (EpidopeEngine, "fasta"),
    "discotope": (DiscoTopeEngine, "pdb"),
    "scannet": (ScanNetEngine, "pdb"),
}


def active_engines_for(input_type: str, pdb_mode: Optional[str] = None) -> List[str]:
    """Resuelve el subconjunto de motores de ``ENGINE_REGISTRY`` activo para este camino.

    Args:
        input_type: ``'fasta'`` o ``'structure'`` (ver
            ``src.utils.input_router.RoutedInput.input_type``).
        pdb_mode: Requerido (y solo relevante) cuando ``input_type ==
            'structure'``: ``'structure_only'`` o ``'structure_and_sequence'``
            (ver ``Settings.PDB_PROCESSING_MODE``). Se ignora si
            ``input_type == 'fasta'``.

    Returns:
        Lista de claves de ``ENGINE_REGISTRY`` a ejecutar, en un orden fijo y
        deterministico (mismo orden en el que luego se etiqueta la columna
        ``origen`` de ``src.engines.consensus.build_annotated_union_table``).

    Raises:
        InputRoutingError: Si ``input_type`` no es reconocido, o si
            ``input_type == 'structure'`` con un ``pdb_mode`` no reconocido.
    """
    if input_type == "fasta":
        return ["bepipred", "epidope"]

    if input_type == "structure":
        if pdb_mode == "structure_only":
            return ["discotope", "scannet"]
        if pdb_mode == "structure_and_sequence":
            return ["bepipred", "epidope", "discotope", "scannet"]
        raise InputRoutingError(
            f"PDB_PROCESSING_MODE='{pdb_mode}' no reconocido para input_type='structure' "
            "(valores validos: 'structure_only', 'structure_and_sequence')."
        )

    raise InputRoutingError(
        f"input_type='{input_type}' no reconocido (valores validos: 'fasta', 'structure')."
    )
