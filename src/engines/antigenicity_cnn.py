"""Fase 1: Cribado de Antigenicidad mediante una 1D-CNN sobre Escalas Z de Hellberg.

Sustituye el modelo lineal ACC (Auto Cross Covariance) clasico por una
proyeccion biofisica espacial de cada secuencia a un tensor ``(3, N)`` -- las
tres primeras componentes principales de las propiedades fisicoquimicas de los
20 aminoacidos canonicos (Hellberg et al., 1987) -- seguida de una Red
Convolucional Unidimensional nativa en PyTorch capaz de detectar motifs
antigenicos lineales sin necesidad de alineamiento de secuencias.

Toda la inferencia se ejecuta bajo ``torch.no_grad()`` estricto y con
liberacion explicita de memoria (``del`` + ``gc.collect()``) tras cada lote,
para garantizar cero desbordamientos de RAM en ejecuciones HTS prolongadas
sobre CPU.
"""

import gc
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config.settings import Settings
from src.engines.base_engine import BaseEngine
from src.models import AntigenicityResult, SequenceRecord
from src.utils.batching import dynamic_batches
from src.utils.exceptions import ModelLoadError
from src.utils.logger_config import setup_logger
from src.utils.memory_profiler import log_memory_checkpoint, warn_if_over_budget

logger = setup_logger(__name__)

# Escalas Z de Hellberg (1987): z1 (hidrofobicidad/electronegatividad),
# z2 (volumen estereo-molecular), z3 (polaridad/carga electronica).
# Derivadas por PCA sobre 29 descriptores fisicoquimicos medidos empiricamente
# para los 20 aminoacidos codificados geneticamente.
HELLBERG_Z_SCALES: Dict[str, Tuple[float, float, float]] = {
    "A": (0.07, -1.73, 0.09),
    "V": (-2.69, -2.53, -1.29),
    "L": (-4.19, -1.03, -0.98),
    "I": (-4.44, -1.68, -1.03),
    "P": (-1.22, 0.88, 2.23),
    "F": (-4.92, 1.30, 0.45),
    "W": (-4.75, 3.65, 0.85),
    "M": (-2.49, -0.27, -0.41),
    "K": (2.84, 1.41, -3.14),
    "R": (2.88, 2.52, -3.44),
    "H": (2.41, 1.74, 1.11),
    "G": (2.23, -5.36, 0.30),
    "S": (1.96, -1.63, 0.57),
    "T": (0.92, -2.09, -1.40),
    "C": (0.71, -0.97, 4.13),
    "Y": (-1.39, 2.32, 0.01),
    "N": (3.22, 1.45, 0.84),
    "Q": (2.18, 0.53, -1.14),
    "D": (3.64, 1.13, 2.36),
    "E": (3.08, 0.39, -0.07),
}


class ZScaleEncoder:
    """Convierte secuencias de aminoacidos en tensores biofisicos ``(3, N)``."""

    @staticmethod
    def encode_sequence(sequence: str) -> np.ndarray:
        """Proyecta una unica secuencia a una matriz de Escalas Z.

        Args:
            sequence: Secuencia de aminoacidos canonicos (ya saneada).

        Returns:
            Array de ``numpy`` de forma ``(3, N)`` con ``N = len(sequence)``.
        """
        matrix = np.array([HELLBERG_Z_SCALES[aa] for aa in sequence], dtype=np.float32)
        return matrix.T  # (3, N)

    @staticmethod
    def encode_batch(sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Codifica y acolcha (pad) un lote de secuencias de longitud variable.

        Args:
            sequences: Lote de secuencias a codificar.

        Returns:
            Tupla ``(tensor, mask)`` donde ``tensor`` tiene forma
            ``(B, 3, L_max)`` y ``mask`` es un tensor booleano ``(B, L_max)``
            que marca con ``True`` las posiciones reales (no acolchadas).
        """
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)

        padded = np.zeros((batch_size, 3, max_len), dtype=np.float32)
        mask = np.zeros((batch_size, max_len), dtype=bool)

        for i, seq in enumerate(sequences):
            encoded = ZScaleEncoder.encode_sequence(seq)
            seq_len = encoded.shape[1]
            padded[i, :, :seq_len] = encoded
            mask[i, :seq_len] = True

        return torch.from_numpy(padded), torch.from_numpy(mask)


class AntigenicityCNN(nn.Module):
    """Red Convolucional 1D con pooling enmascarado para deteccion de motifs antigenicos.

    Arquitectura: ``Conv1d -> BatchNorm1d -> ReLU`` (x2) seguido de un Global
    Max Pooling enmascarado (ignora posiciones de padding) y una cabeza densa
    con activacion sigmoide que produce una unica probabilidad de
    antigenicidad por secuencia.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, kernel_size: int = 5):
        """Inicializa las capas de la red.

        Args:
            in_channels: Numero de canales de entrada (3: z1, z2, z3).
            hidden_channels: Numero de filtros de la primera capa convolucional.
                La segunda capa usa el doble.
            kernel_size: Tamano del kernel convolucional (impar, para padding
                simetrico "same").
        """
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.classifier_head = nn.Linear(hidden_channels * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Ejecuta el forward pass completo de la red.

        Args:
            x: Tensor de entrada de forma ``(B, 3, L)``.
            mask: Tensor booleano de forma ``(B, L)`` marcando posiciones reales.

        Returns:
            Tensor de forma ``(B,)`` con la probabilidad de antigenicidad de
            cada secuencia del lote, en el rango ``[0, 1]``.
        """
        features = self.relu(self.bn1(self.conv1(x)))
        features = self.relu(self.bn2(self.conv2(features)))  # (B, C, L)

        pooling_mask = mask.unsqueeze(1)  # (B, 1, L)
        masked_features = features.masked_fill(~pooling_mask, float("-inf"))
        pooled, _ = torch.max(masked_features, dim=2)  # Global Max Pooling -> (B, C)

        logits = self.classifier_head(pooled).squeeze(-1)  # (B,)
        return self.sigmoid(logits)


class AntigenicityCNNEngine(BaseEngine[SequenceRecord, AntigenicityResult]):
    """Motor de Fase 1: orquesta codificacion, inferencia y gestion de memoria."""

    METHOD_NAME: str = "1D-CNN-HellbergZScale"

    def __init__(self, threshold: Optional[float] = None):
        """Instancia la red y carga pesos entrenados si estan disponibles.

        Si ``Settings.ANTIGENICITY_CNN_WEIGHTS_PATH`` no apunta a un archivo
        existente, la red se inicializa de forma deterministica (semilla fija)
        y se emite una advertencia: sin entrenamiento supervisado sobre un
        corpus IEDB, la salida de la red no esta calibrada para uso en
        produccion, solo demuestra la arquitectura y el flujo de inferencia.

        Args:
            threshold: Umbral de decision de antigenicidad. Si es ``None``, se
                usa ``Settings.ANTIGENICITY_THRESHOLD``.

        Raises:
            ModelLoadError: Si existe un archivo de pesos pero esta corrupto o
                es incompatible con la arquitectura actual.
        """
        self.device = torch.device(Settings.DEVICE)
        self.threshold = threshold if threshold is not None else Settings.ANTIGENICITY_THRESHOLD

        torch.manual_seed(Settings.ANTIGENICITY_RANDOM_SEED)
        self.model = AntigenicityCNN(
            in_channels=3,
            hidden_channels=Settings.ANTIGENICITY_CNN_CHANNELS,
            kernel_size=Settings.ANTIGENICITY_CNN_KERNEL_SIZE,
        ).to(self.device)

        weights_path = Settings.ANTIGENICITY_CNN_WEIGHTS_PATH
        if weights_path is not None and weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as exc:
                raise ModelLoadError(
                    f"Fallo al cargar pesos de la 1D-CNN desde '{weights_path}': {exc}"
                ) from exc
        else:
            logger.warning(
                "No se encontraron pesos entrenados para la 1D-CNN de antigenicidad "
                "(ANTIGENICITY_CNN_WEIGHTS_PATH no configurado o inexistente). "
                "La red opera con inicializacion aleatoria deterministica: los scores "
                "reflejan la arquitectura, no un modelo calibrado sobre IEDB."
            )

        self.model.eval()

    def run(self, items: Sequence[SequenceRecord]) -> List[AntigenicityResult]:
        """Clasifica un lote de secuencias saneadas por antigenicidad.

        Args:
            items: Secuencias ya saneadas por ``FastaParser``.

        Returns:
            Lista de :class:`~src.models.AntigenicityResult`, en el mismo
            orden que ``items``.
        """
        if not items:
            return []

        id_to_result: Dict[str, AntigenicityResult] = {}
        batches = dynamic_batches(
            items=items,
            length_fn=lambda record: len(record.sequence),
            max_residues_per_batch=Settings.ANTIGENICITY_MAX_RESIDUES_PER_BATCH,
            max_items_per_batch=Settings.ANTIGENICITY_MAX_ITEMS_PER_BATCH,
        )

        logger.info(
            "Fase 1 (Antigenicidad): %d secuencias agrupadas en %d mini-lotes dinamicos.",
            len(items),
            len(batches),
        )

        for batch_idx, batch in enumerate(batches):
            sequences = [record.sequence for record in batch]
            tensor, mask = ZScaleEncoder.encode_batch(sequences)
            tensor = tensor.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                probs = self.model(tensor, mask)

            probs_np = probs.cpu().numpy()
            for record, score in zip(batch, probs_np):
                score_val = float(score)
                id_to_result[record.id] = AntigenicityResult(
                    record=record,
                    score=score_val,
                    is_antigenic=score_val >= self.threshold,
                    method=self.METHOD_NAME,
                )

            del tensor, mask, probs, probs_np
            rss_mb = log_memory_checkpoint(logger, f"antigenicity_batch_{batch_idx + 1}/{len(batches)}")
            warn_if_over_budget(
                logger,
                f"antigenicity_batch_{batch_idx + 1}",
                rss_mb,
                Settings.ANTIGENICITY_MEMORY_BUDGET_MB,
            )

        gc.collect()
        return [id_to_result[record.id] for record in items]
