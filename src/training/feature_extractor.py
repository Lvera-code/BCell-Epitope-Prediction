"""Pre-extraccion fuera de memoria de features para el entrenamiento (Fase 1 y Fase 2).

Calcula, una unica vez y bajo ``torch.no_grad()`` estricto, las dos
representaciones de entrada consumidas por los bucles de entrenamiento de
``src/training/trainer.py``:

* Matrices biofisicas de Escalas Z de Hellberg ``(3, L)`` (entrada de la
  1D-CNN de Fase 1).
* Embeddings residuo a residuo de ESM-2 ``(L, 640)`` (entrada del
  ``ResidueClassifier`` de Fase 2).

Esta prohibido recalcular embeddings de ESM-2 en cada epoca de entrenamiento:
esta pre-extraccion se ejecuta una sola vez por split (train/val/test) y
escribe los resultados progresivamente en *shards* (``.pt``) a disco,
liberando la RAM (``del`` + ``gc.collect()``) tras cada lote procesado. Los
bucles de entrenamiento nunca vuelven a invocar el modelo ESM-2: solo leen
estos shards de disco de forma perezosa (lazy).
"""

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import List, NamedTuple

import torch
from transformers import AutoTokenizer, EsmModel
from transformers import logging as hf_logging

from src.config.settings import Settings
from src.engines.antigenicity_cnn import ZScaleEncoder
from src.utils.batching import dynamic_batches
from src.utils.exceptions import ModelLoadError
from src.utils.logger_config import setup_logger
from src.utils.memory_profiler import log_memory_checkpoint

hf_logging.set_verbosity_error()

logger = setup_logger(__name__)


class ManifestRecord(NamedTuple):
    """Una fila del manifiesto CSV producido por ``dataset_prep.py``."""

    id: str
    label: int
    sequence: str


def read_manifest(manifest_path: Path) -> List[ManifestRecord]:
    """Lee un manifiesto CSV de dataset curado.

    Args:
        manifest_path: Ruta al archivo ``{split}_manifest.csv``.

    Returns:
        Lista de :class:`ManifestRecord`.

    Raises:
        FileNotFoundError: Si el manifiesto no existe.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifiesto no encontrado: {manifest_path}")

    records: List[ManifestRecord] = []
    with open(manifest_path, mode="r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                ManifestRecord(id=row["id"], label=int(row["label"]), sequence=row["sequence"])
            )
    return records


def _enforce_max_sequence_length(records: List[ManifestRecord]) -> List[ManifestRecord]:
    """Trunca defensivamente secuencias que excedan el limite fisico de ESM-2 (1022 aa).

    Args:
        records: Registros del manifiesto a validar.

    Returns:
        Registros con secuencias acotadas a ``Settings.ESM_MAX_SEQUENCE_LENGTH``.
    """
    max_len = Settings.ESM_MAX_SEQUENCE_LENGTH
    result: List[ManifestRecord] = []

    for record in records:
        if len(record.sequence) > max_len:
            logger.warning(
                "Secuencia '%s' (%d aa) excede el limite fisico de ESM-2 (%d aa); se trunca.",
                record.id,
                len(record.sequence),
                max_len,
            )
            record = record._replace(sequence=record.sequence[:max_len])
        result.append(record)

    return result


def _load_esm2():
    """Carga tokenizer y modelo ESM-2 para extraccion de embeddings (sin cabeza de clasificacion).

    Returns:
        Tupla ``(tokenizer, model, hidden_dim)``.

    Raises:
        ModelLoadError: Si el modelo o el tokenizer no pueden cargarse.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(Settings.ESM_MODEL_NAME)
        model = EsmModel.from_pretrained(Settings.ESM_MODEL_NAME).to(Settings.DEVICE)
        model.eval()
    except Exception as exc:
        raise ModelLoadError(
            f"Fallo al cargar ESM-2 '{Settings.ESM_MODEL_NAME}' para extraccion de features: {exc}"
        ) from exc
    return tokenizer, model, model.config.hidden_size


def extract_features_for_split(
    split_name: str,
    manifest_path: Path,
    output_root: Path = Settings.FEATURES_DIR,
) -> int:
    """Extrae y guarda en shards las features de Fase 1 y Fase 2 para un split.

    Args:
        split_name: Nombre del split (``"train"``, ``"val"`` o ``"test"``).
        manifest_path: Ruta al manifiesto CSV del split.
        output_root: Directorio raiz donde se escriben los shards.

    Returns:
        Numero de shards escritos.
    """
    records = read_manifest(manifest_path)
    if not records:
        logger.warning("Manifiesto '%s' esta vacio; no se extraen features.", manifest_path)
        return 0

    records = _enforce_max_sequence_length(records)

    hellberg_dir = output_root / split_name / "hellberg"
    esm2_dir = output_root / split_name / "esm2"
    hellberg_dir.mkdir(parents=True, exist_ok=True)
    esm2_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando ESM-2 para extraccion de embeddings ('%s')...", split_name)
    tokenizer, model, hidden_dim = _load_esm2()

    shard_item_cap = min(Settings.ESM_MAX_ITEMS_PER_BATCH, Settings.FEATURE_SHARD_SIZE)
    batches = dynamic_batches(
        items=records,
        length_fn=lambda rec: len(rec.sequence),
        max_residues_per_batch=Settings.ESM_MAX_RESIDUES_PER_BATCH,
        max_items_per_batch=shard_item_cap,
    )

    logger.info(
        "Split '%s': %d secuencias -> %d shards (hidden_dim=%d).",
        split_name,
        len(records),
        len(batches),
        hidden_dim,
    )

    device = torch.device(Settings.DEVICE)

    for shard_idx, batch in enumerate(batches):
        ids = [rec.id for rec in batch]
        labels = [rec.label for rec in batch]
        sequences = [rec.sequence for rec in batch]

        hellberg_tensors = [
            torch.from_numpy(ZScaleEncoder.encode_sequence(seq)).clone() for seq in sequences
        ]

        inputs = tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (B, L+2, hidden_dim)

        esm2_tensors = [
            hidden_states[i, 1 : len(sequences[i]) + 1, :].clone().cpu()
            for i in range(len(sequences))
        ]

        torch.save(
            {"ids": ids, "z_scale": hellberg_tensors, "label": labels},
            hellberg_dir / f"shard_{shard_idx:04d}.pt",
        )
        torch.save(
            {"ids": ids, "embeddings": esm2_tensors, "label": labels},
            esm2_dir / f"shard_{shard_idx:04d}.pt",
        )

        del inputs, outputs, hidden_states, hellberg_tensors, esm2_tensors
        log_memory_checkpoint(logger, f"feature_extraction[{split_name}]_shard_{shard_idx + 1}/{len(batches)}")

    del model, tokenizer
    gc.collect()

    logger.info("Split '%s': extraccion completada (%d shards).", split_name, len(batches))
    return len(batches)


def _parse_arguments() -> argparse.Namespace:
    """Define la interfaz de linea de comandos para la extraccion standalone."""
    parser = argparse.ArgumentParser(
        description="Pre-extraccion de features (Hellberg + ESM-2) para entrenamiento offline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(("train", "val", "test")), choices=["train", "val", "test"]
    )
    parser.add_argument("--data-dir", type=Path, default=Settings.TRAINING_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Settings.FEATURES_DIR)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Punto de entrada standalone: ``python -m src.training.feature_extractor``."""
    args = _parse_arguments()

    if args.offline:
        Settings.apply_offline_mode()

    Settings.apply_thread_limits()

    try:
        for split_name in args.splits:
            manifest_path = args.data_dir / f"{split_name}_manifest.csv"
            extract_features_for_split(split_name, manifest_path, args.output_dir)
        return 0
    except (FileNotFoundError, ModelLoadError) as exc:
        logger.critical("Fallo fatal durante la extraccion de features: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
