from pathlib import Path
from typing import List, Set
from dataclasses import dataclass
from src.utils.logger_config import setup_logger

logger = setup_logger()

# Los 20 aminoácidos canónicos estándar
CANONICAL_AA: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class FastaRecord:
    id: str
    sequence: str

class FastaParser:
    @staticmethod
    def parse(file_path: Path, min_length: int = 8) -> List[FastaRecord]:
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo FASTA no existe: {file_path}")

        records: List[FastaRecord] = []
        current_id = ""
        current_seq_chunks = []

        with open(file_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id and current_seq_chunks:
                        FastaParser._process_and_append(current_id, "".join(current_seq_chunks), min_length, records)
                    current_id = line[1:].strip().split()[0]  # Tomar solo el primer ID limpio
                    current_seq_chunks = []
                else:
                    current_seq_chunks.append(line)

        # Procesar el último registro
        if current_id and current_seq_chunks:
            FastaParser._process_and_append(current_id, "".join(current_seq_chunks), min_length, records)

        if not records:
            logger.warning("No se extrajo ninguna secuencia válida que cumpla los criterios de longitud o pureza.")
            
        return records

    @staticmethod
    def _process_and_append(seq_id: str, raw_seq: str, min_len: int, records: List[FastaRecord]) -> None:
        # 1. Normalizar: Mayúsculas y eliminar cualquier espacio, tabulación o número incrustado
        clean_seq = "".join(raw_seq.upper().split())
        
        # 2. Filtrar por longitud mínima estructural
        if len(clean_seq) < min_len:
            logger.warning(f"Secuencia descartada '{seq_id}': Longitud insuficiente ({len(clean_seq)} aa < mínimo {min_len} aa).")
            return

        # 3. Detectar residuos anómalos (X, U, B, Z, O, *, etc.)
        illegal_chars = set(clean_seq) - CANONICAL_AA
        if illegal_chars:
            logger.warning(f"Secuencia descartada '{seq_id}': Contiene residuos ambiguos: {illegal_chars}")
            return

        records.append(FastaRecord(id=seq_id, sequence=clean_seq))