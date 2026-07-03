from pathlib import Path
import torch
from src.models import OrganismClass


class Settings:
    # Rutas
    RAW_FASTA_PATH: Path = Path("data/raw/candidatos.fasta")
    PROCESSED_DIR: Path = Path("data/processed")
    
    # Fase 1: VaxiJen (Criba de Antigenicidad)
    VAXIJEN_THRESHOLD: float = 0.51
    VAXIJEN_ORGANISM: OrganismClass = OrganismClass.VIRAL
    VAXIJEN_LAG: int = 8
    
    # Fase 2: BepiPred (ESM-2 Embeddings)
    ESM_MODEL_NAME: str = "facebook/esm2_t30_150M_UR50D"
    ESM_BATCH_SIZE: int = 4
    EPITOPE_THRESHOLD: float = 0.35
    
    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls) -> None:
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_FASTA_PATH.parent.mkdir(parents=True, exist_ok=True)