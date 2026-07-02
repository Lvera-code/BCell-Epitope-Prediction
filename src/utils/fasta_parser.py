from pathlib import Path
from typing import List
from src.models import SequenceRecord
from src.utils.exceptions import InvalidSequenceError
from src.utils.logger_config import setup_logger

logger = setup_logger()
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


class FastaParser:
    @staticmethod
    def parse(fasta_path: Path, min_length: int = 10) -> List[SequenceRecord]:
        if not fasta_path.exists():
            raise FileNotFoundError(f"Archivo FASTA no encontrado: {fasta_path}")

        records: List[SequenceRecord] = []
        current_id = ""
        current_seq: List[str] = []

        with open(fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id:
                        FastaParser._process_record(current_id, current_seq, min_length, records)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id:
                FastaParser._process_record(current_id, current_seq, min_length, records)

        return records

    @staticmethod
    def _process_record(seq_id: str, seq_fragments: List[str], min_length: int, records: List[SequenceRecord]) -> None:
        seq_str = "".join(seq_fragments).upper()
        try:
            FastaParser._validate_sequence(seq_id, seq_str, min_length)
            records.append(SequenceRecord(id=seq_id, sequence=seq_str))
        except InvalidSequenceError as e:
            # Capturamos el error por secuencia sin abortar el lote general
            logger.warning(f"Secuencia descartada en FASTA: {e}")

    @staticmethod
    def _validate_sequence(seq_id: str, seq: str, min_length: int) -> None:
        if len(seq) < min_length:
            raise InvalidSequenceError(
                f"'{seq_id}' es demasiado corta ({len(seq)} aa < mínimo {min_length})."
            )
        invalid_chars = set(seq) - STANDARD_AA
        if invalid_chars:
            raise InvalidSequenceError(
                f"'{seq_id}' contiene residuos ilegales o no estándar: {invalid_chars}"
            )