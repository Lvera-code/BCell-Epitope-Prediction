"""Enrutador de input: clasifica cada archivo de entrada como FASTA o estructura.

Corre ANTES de la Fase 1. Decide si un archivo del batch alimenta el flujo de
secuencia existente (Fase 1: ``src.utils.fasta_parser``, sin cambios) o el
nuevo flujo de estructura (Fase 1.5: ``src.utils.structure_parser``).

Estrategia de deteccion en dos capas, nunca confiando ciegamente en la
extension:

1. Heuristica primaria por extension (``.fasta``/``.fa``/``.faa`` -> FASTA;
   ``.pdb``/``.ent`` -> PDB; ``.cif``/``.mmcif`` -> mmCIF).
2. Validacion por contenido como respaldo/confirmacion: primera linea no
   vacia empieza con ``'>'`` -> FASTA; primeras lineas contienen registros
   tipo ``ATOM``/``HEADER``/``CRYST1`` (PDB legado) o un bloque ``data_``
   (mmCIF) -> estructura.

Si ninguna de las dos capas permite determinar el tipo con confianza, se
lanza ``InputRoutingError`` en vez de adivinar en silencio y enrutar mal (un
FASTA enrutado como estructura, o viceversa, fallaria mas adelante con un
error opaco y dificil de diagnosticar).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from src.utils.exceptions import InputRoutingError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

FASTA_EXTENSIONS = {".fasta", ".fa", ".faa"}
PDB_EXTENSIONS = {".pdb", ".ent"}
MMCIF_EXTENSIONS = {".cif", ".mmcif"}
STRUCTURE_EXTENSIONS = PDB_EXTENSIONS | MMCIF_EXTENSIONS

# Registros PDB legado que confirman contenido de estructura por texto plano
# (ver seccion 6 del formato PDB: https://www.wwpdb.org/documentation/file-format).
_PDB_CONTENT_MARKERS = ("ATOM", "HEADER", "CRYST1", "HETATM", "MODEL")
_N_LINES_TO_SNIFF = 20


@dataclass(frozen=True)
class RoutedInput:
    """Un archivo de entrada ya clasificado, listo para Fase 1 o Fase 1.5."""

    path: Path
    input_type: str  # "fasta" | "structure"


def _sniff_content_type(path: Path) -> str:
    """Inspecciona las primeras lineas no vacias del archivo para inferir su tipo.

    Returns:
        ``"fasta"``, ``"structure"`` o ``"unknown"``.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = []
            for line in fh:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
                if len(lines) >= _N_LINES_TO_SNIFF:
                    break
    except OSError:
        return "unknown"

    if not lines:
        return "unknown"

    if lines[0].startswith(">"):
        return "fasta"

    if lines[0].startswith("data_"):
        return "structure"

    for line in lines:
        if line.split(maxsplit=1)[0] in _PDB_CONTENT_MARKERS:
            return "structure"

    return "unknown"


def _extension_type(path: Path) -> str:
    """Clasificacion por extension. Returns ``"fasta"``, ``"structure"`` o ``"unknown"``."""
    suffix = path.suffix.lower()
    if suffix in FASTA_EXTENSIONS:
        return "fasta"
    if suffix in STRUCTURE_EXTENSIONS:
        return "structure"
    return "unknown"


def route_input(path: Path) -> RoutedInput:
    """Clasifica un unico archivo de entrada como ``"fasta"`` o ``"structure"``.

    La extension y el contenido se consultan siempre ambos: si coinciden, se
    usa ese tipo directamente. Si la extension no es reconocible pero el
    contenido si lo es, se usa el tipo de contenido (respaldo documentado en
    el brief: "extension incorrecta pero contenido reconocible"). Si la
    extension SI es reconocible pero el contenido reconocible NO coincide con
    ella, se prioriza el contenido (mas confiable que el nombre de archivo) y
    se deja constancia en el log de la discrepancia.

    Args:
        path: Ruta al archivo de entrada.

    Returns:
        :class:`RoutedInput` con el tipo resuelto.

    Raises:
        FileNotFoundError: Si ``path`` no existe.
        InputRoutingError: Si ni la extension ni el contenido permiten
            determinar el tipo con confianza.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el archivo de entrada: {path}")

    ext_type = _extension_type(path)
    content_type = _sniff_content_type(path)

    if content_type != "unknown":
        if ext_type != "unknown" and ext_type != content_type:
            logger.warning(
                "'%s': la extension sugiere '%s' pero el contenido es '%s'. "
                "Se prioriza el contenido (mas confiable que la extension).",
                path.name, ext_type, content_type,
            )
        elif ext_type == "unknown":
            logger.info(
                "'%s': extension '%s' no reconocida, pero el contenido se identifico como '%s'.",
                path.name, path.suffix, content_type,
            )
        return RoutedInput(path=path, input_type=content_type)

    if ext_type != "unknown":
        logger.warning(
            "'%s': el contenido no pudo confirmarse por sniffing, se usa la extension "
            "('%s') como ultimo recurso.",
            path.name, ext_type,
        )
        return RoutedInput(path=path, input_type=ext_type)

    raise InputRoutingError(
        f"No se pudo determinar el tipo de '{path.name}' (FASTA vs estructura): ni la extension "
        f"('{path.suffix}') ni el contenido de las primeras {_N_LINES_TO_SNIFF} lineas no vacias "
        "son reconocibles. Extensiones esperadas: FASTA "
        f"({sorted(FASTA_EXTENSIONS)}), estructura ({sorted(STRUCTURE_EXTENSIONS)}); o contenido "
        "que empiece con '>' (FASTA), 'data_' (mmCIF), o incluya registros ATOM/HEADER/CRYST1 "
        "(PDB legado). Verifica el archivo o renombralo con una extension reconocida."
    )


def route_inputs(paths: Sequence[Path]) -> List[RoutedInput]:
    """Clasifica un batch de archivos de entrada, en el mismo orden que ``paths``."""
    return [route_input(path) for path in paths]
