"""Fase 1: Saneamiento de FASTA crudo (aduana de entrada del pipeline).

Responsabilidad exclusiva: leer un FASTA tal como llega del usuario, separarlo
en registros (cabecera, secuencia) y producir una version saneada -mayusculas,
sin saltos de linea internos, sin caracteres no canonicos- que sea segura para
enviar a BioLib (Fase 2), a BLASTp (Fase 4) y a los motores de inmunogenicidad
(Fase 5). El descarte de un registro individual por quedar vacio tras el
saneamiento es recuperable (ver ``InvalidSequenceError``); un FASTA sin ningun
'>' es un error fatal (ver ``FastaFormatError``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.utils.exceptions import FastaFormatError, InvalidSequenceError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Alfabeto canonico de los 20 aminoacidos estandar. Cualquier otro caracter
# (ambiguedades IUPAC como X/B/Z/J/U/O, gaps '-', stops '*', digitos, etc.)
# se elimina en el saneamiento.
CANONICAL_AMINOACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class FastaRecord:
    """Un registro FASTA ya saneado, listo para las fases siguientes."""

    header: str
    accession: str
    sequence: str
    removed_chars: int


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    """Separa un archivo FASTA crudo en pares ``(cabecera, secuencia_cruda)``.

    Args:
        path: Ruta al archivo FASTA de entrada.

    Returns:
        Lista de tuplas ``(header_sin_'>', secuencia_sin_saltos_de_linea)``
        en el mismo orden en que aparecen en el archivo.

    Raises:
        FileNotFoundError: Si ``path`` no existe.
        FastaFormatError: Si el archivo no contiene ningun registro valido
            (no empieza con '>' o esta vacio). Es un error fatal.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el archivo FASTA de entrada: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not lines or not lines[0].startswith(">"):
        raise FastaFormatError(
            f"'{path.name}' no cumple la sintaxis FASTA minima (debe iniciar con '>')."
        )

    records: List[Tuple[str, str]] = []
    header: str = ""
    seq_chunks: List[str] = []

    for line in lines:
        if line.startswith(">"):
            if header:
                records.append((header, "".join(seq_chunks)))
            header = line[1:].strip()
            seq_chunks = []
        else:
            seq_chunks.append(line)

    if header:
        records.append((header, "".join(seq_chunks)))

    if not records:
        raise FastaFormatError(f"'{path.name}' no contiene ningun registro FASTA valido.")

    return records


def sanitize_sequence(raw_sequence: str) -> Tuple[str, int]:
    """Normaliza una secuencia cruda: mayusculas + solo aminoacidos canonicos.

    Args:
        raw_sequence: Secuencia de aminoacidos sin procesar.

    Returns:
        Tupla ``(secuencia_limpia, n_caracteres_eliminados)``.
    """
    upper = raw_sequence.upper()
    clean = "".join(c for c in upper if c in CANONICAL_AMINOACIDS)
    return clean, len(upper) - len(clean)


def load_and_sanitize(path: Path) -> List[FastaRecord]:
    """Lee y sanea un FASTA completo, descartando registros que queden vacios.

    Args:
        path: Ruta al archivo FASTA de entrada (dentro de ``fasta_inputs/``).

    Returns:
        Lista de :class:`FastaRecord` saneados.

    Raises:
        FastaFormatError: Si el archivo no tiene sintaxis FASTA valida (fatal).
        InvalidSequenceError: Si NINGUN registro sobrevive el saneamiento
            (fatal a nivel de archivo). El descarte de registros individuales
            solo se loggea como warning y no detiene el resto del lote.
    """
    raw_records = parse_fasta(path)
    sane_records: List[FastaRecord] = []

    for header, raw_seq in raw_records:
        clean_seq, removed = sanitize_sequence(raw_seq)
        if not clean_seq:
            logger.warning(
                "Registro '%s' descartado: no quedaron residuos canonicos tras el saneamiento.",
                header,
            )
            continue

        accession = header.split()[0] if header else "UNKNOWN"
        sane_records.append(
            FastaRecord(header=header, accession=accession, sequence=clean_seq, removed_chars=removed)
        )

    if not sane_records:
        raise InvalidSequenceError(
            f"Ninguna secuencia valida en '{path.name}' tras eliminar caracteres no canonicos."
        )

    return sane_records


def write_fasta(records: List[FastaRecord], out_path: Path, line_width: int = 60) -> None:
    """Escribe una lista de :class:`FastaRecord` saneados como FASTA valido.

    Args:
        records: Registros a escribir, en orden.
        out_path: Ruta de salida (se sobreescribe si ya existe).
        line_width: Ancho de linea para el envoltorio de la secuencia.
    """
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(f">{record.header}\n")
            seq = record.sequence
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")
