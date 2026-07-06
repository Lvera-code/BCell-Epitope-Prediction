"""Modulo de aduana y saneamiento de secuencias FASTA.

Actua como frontera estricta entre datos crudos externos y el resto del
pipeline: normaliza el formato, excinde residuos ambiguos IUPAC (``X``, ``B``,
``Z``, ``J``) y descarta cualquier secuencia que, tras el saneamiento, no
alcance la longitud minima requerida para el calculo de covarianza posterior
(Auto Cross Covariance / campo receptivo de la 1D-CNN en Fase 1).
"""

from pathlib import Path
from typing import List, Set

from src.models import SequenceRecord
from src.utils.exceptions import FastaFormatError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

CANONICAL_AA: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_RESIDUES: Set[str] = {"X", "B", "Z", "J"}
DEFAULT_MIN_LENGTH: int = 9


class FastaParser:
    """Parser FASTA estricto con saneamiento biologico integrado."""

    @staticmethod
    def parse(file_path: Path, min_length: int = DEFAULT_MIN_LENGTH) -> List[SequenceRecord]:
        """Parsea, sanea y filtra un archivo FASTA.

        El proceso por secuencia es:

        1. Normalizacion: mayusculas, eliminacion de espacios/tabulaciones internas.
        2. Excision de residuos ambiguos IUPAC (``X``, ``B``, ``Z``, ``J``): se
           remueven del cuerpo de la secuencia (no se descarta el registro).
        3. Rechazo si, tras la excision, quedan caracteres fuera del alfabeto
           canonico de 20 aminoacidos (anomalia estructural: digitos, ``*``,
           gaps, aminoacidos no estandar como ``U``/``O``, etc.).
        4. Rechazo (WARNING) si la longitud resultante es menor que
           ``min_length``.

        Args:
            file_path: Ruta al archivo FASTA de entrada.
            min_length: Longitud minima de aminoacidos exigida tras el
                saneamiento. Por defecto 9, el minimo estructural para que el
                calculo de Auto Cross Covariance (lag hasta 8) sea valido.

        Returns:
            Lista de :class:`~src.models.SequenceRecord` que superaron el
            saneamiento completo.

        Raises:
            FileNotFoundError: Si ``file_path`` no existe.
            FastaFormatError: Si el archivo no contiene ninguna cabecera FASTA
                valida (``>``), lo que indica un formato de entrada corrupto.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo FASTA no existe: {file_path}")

        raw_entries = FastaParser._split_entries(file_path)
        if not raw_entries:
            raise FastaFormatError(
                f"El archivo '{file_path}' no contiene ninguna cabecera FASTA ('>') valida."
            )

        records: List[SequenceRecord] = []
        for seq_id, description, raw_seq in raw_entries:
            record = FastaParser._sanitize_entry(seq_id, description, raw_seq, min_length)
            if record is not None:
                records.append(record)

        if not records:
            logger.warning(
                "No se extrajo ninguna secuencia valida de '%s' tras el saneamiento completo.",
                file_path,
            )
        else:
            logger.info(
                "Saneamiento completado: %d/%d secuencias validas extraidas de '%s'.",
                len(records),
                len(raw_entries),
                file_path,
            )

        return records

    @staticmethod
    def _split_entries(file_path: Path) -> List[tuple]:
        """Divide el archivo en tuplas crudas ``(id, descripcion, secuencia)``."""
        entries: List[tuple] = []
        current_id = ""
        current_desc = ""
        current_chunks: List[str] = []

        with open(file_path, mode="r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id:
                        entries.append((current_id, current_desc, "".join(current_chunks)))
                    header = line[1:].strip()
                    tokens = header.split(maxsplit=1)
                    current_id = tokens[0] if tokens else ""
                    current_desc = tokens[1] if len(tokens) > 1 else ""
                    current_chunks = []
                else:
                    current_chunks.append(line)

            if current_id:
                entries.append((current_id, current_desc, "".join(current_chunks)))

        return entries

    @staticmethod
    def _sanitize_entry(
        seq_id: str, description: str, raw_seq: str, min_length: int
    ) -> "SequenceRecord | None":
        """Aplica la cadena de saneamiento a una unica entrada FASTA."""
        normalized = "".join(raw_seq.upper().split())

        ambiguous_found = {c for c in normalized if c in AMBIGUOUS_RESIDUES}
        if ambiguous_found:
            removed_count = sum(1 for c in normalized if c in AMBIGUOUS_RESIDUES)
            logger.info(
                "Secuencia '%s': excindidos %d residuos ambiguos %s.",
                seq_id,
                removed_count,
                sorted(ambiguous_found),
            )
            cleaned = "".join(c for c in normalized if c not in AMBIGUOUS_RESIDUES)
        else:
            cleaned = normalized

        illegal_chars = set(cleaned) - CANONICAL_AA
        if illegal_chars:
            logger.warning(
                "Secuencia descartada '%s': contiene residuos no canonicos irrecuperables %s.",
                seq_id,
                sorted(illegal_chars),
            )
            return None

        if len(cleaned) < min_length:
            logger.warning(
                "Secuencia descartada '%s': longitud insuficiente tras saneamiento "
                "(%d aa < minimo %d aa).",
                seq_id,
                len(cleaned),
                min_length,
            )
            return None

        return SequenceRecord(id=seq_id, sequence=cleaned, description=description)
