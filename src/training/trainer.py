"""Entrenamiento eficiente en CPU de la 1D-CNN (Fase 1) y el ResidueClassifier (Fase 2).

Consume exclusivamente los shards precomputados por
``src/training/feature_extractor.py``: en ningun momento se invoca ESM-2
durante el entrenamiento. Los datasets son ``IterableDataset`` que cargan un
unico shard en memoria a la vez (orden de shards barajado por epoca),
manteniendo un techo de RAM acotado independientemente del tamano total del
dataset.

Ambos bucles de entrenamiento usan ``AdamW``, ``Early Stopping`` sobre la
perdida de validacion y guardan el mejor checkpoint en las rutas exactas
declaradas en ``Settings.ANTIGENICITY_CNN_WEIGHTS_PATH`` y
``Settings.RESIDUE_CLASSIFIER_WEIGHTS_PATH``.

Al finalizar, ``main()`` ejecuta automaticamente
``src/validation/benchmark_suite.py`` sobre el split de test, una vez con
pesos aleatorios (linea base, capturada antes de sobreescribir un checkpoint
preexistente) y una vez con los pesos recien calibrados, mostrando la mejora
real en ROC-AUC y Especificidad.
"""

import argparse
import gc
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from src.config.settings import Settings
from src.engines.antigenicity_cnn import AntigenicityCNN, AntigenicityCNNEngine
from src.engines.epitope_engine import NativeESM2Engine, ResidueClassifier
from src.models import BenchmarkReport
from src.utils.fasta_parser import FastaParser
from src.utils.logger_config import setup_logger
from src.utils.memory_profiler import log_memory_checkpoint
from src.validation.benchmark_suite import BenchmarkSuite, print_benchmark_report

logger = setup_logger(__name__)

MIN_DELTA: float = 1e-4


class ShardIterableDataset(IterableDataset):
    """Dataset perezoso que itera shards ``.pt`` uno a la vez, liberando RAM entre ellos."""

    def __init__(
        self, shard_dir: Path, tensor_key: str, shuffle: bool = True, seed: int = 0
    ):
        """Indexa los archivos de shard disponibles sin cargarlos en memoria.

        Args:
            shard_dir: Directorio con archivos ``shard_XXXX.pt``.
            tensor_key: Clave del tensor de interes dentro de cada shard
                (``"z_scale"`` o ``"embeddings"``).
            shuffle: Si ``True``, baraja el orden de shards y las muestras
                dentro de cada shard en cada epoca.
            seed: Semilla base para el barajado reproducible.

        Raises:
            FileNotFoundError: Si ``shard_dir`` no contiene ningun shard.
        """
        super().__init__()
        self.shard_dir = shard_dir
        self.tensor_key = tensor_key
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        self.shard_paths: List[Path] = sorted(shard_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No se encontraron shards en '{shard_dir}'. "
                "Ejecute 'python -m src.training.feature_extractor' primero."
            )
        self._length = self._count_items()

    def _count_items(self) -> int:
        """Cuenta el total de muestras sumando la longitud de cada shard."""
        total = 0
        for path in self.shard_paths:
            shard = torch.load(path, weights_only=False)
            total += len(shard["label"])
            del shard
        return total

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, float]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        shard_paths = list(self.shard_paths)
        if self.shuffle:
            rng.shuffle(shard_paths)

        for path in shard_paths:
            shard = torch.load(path, weights_only=False)
            items = list(zip(shard["ids"], shard[self.tensor_key], shard["label"]))
            if self.shuffle:
                rng.shuffle(items)

            for _seq_id, tensor, label in items:
                yield tensor, float(label)

            del shard, items
            gc.collect()


def collate_hellberg(
    batch: List[Tuple[torch.Tensor, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Acolcha un lote de matrices de Hellberg ``(3, L_i)`` de longitud variable.

    Args:
        batch: Lista de tuplas ``(tensor(3, L_i), label)``.

    Returns:
        Tupla ``(x, mask, y)``: ``x`` de forma ``(B, 3, L_max)``, ``mask``
        booleana ``(B, L_max)``, ``y`` de forma ``(B,)``.
    """
    tensors, labels = zip(*batch)
    max_len = max(t.shape[1] for t in tensors)
    batch_size = len(tensors)

    x = torch.zeros(batch_size, 3, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, tensor in enumerate(tensors):
        seq_len = tensor.shape[1]
        x[i, :, :seq_len] = tensor
        mask[i, :seq_len] = True

    y = torch.tensor(labels, dtype=torch.float32)
    return x, mask, y


def collate_esm2(
    batch: List[Tuple[torch.Tensor, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Acolcha un lote de embeddings ESM-2 ``(L_i, hidden_dim)`` de longitud variable.

    Las etiquetas por residuo se generan por supervision debil: cada residuo
    de un peptido positivo se etiqueta como epitopo (1) y cada residuo de un
    peptido negativo como no-epitopo (0), heredando la etiqueta a nivel de
    peptido asignada durante la curacion del dataset.

    Args:
        batch: Lista de tuplas ``(tensor(L_i, hidden_dim), label)``.

    Returns:
        Tupla ``(x, mask, residue_labels)``: ``x`` de forma
        ``(B, L_max, hidden_dim)``, ``mask`` booleana ``(B, L_max)``,
        ``residue_labels`` de forma ``(B, L_max)``.
    """
    tensors, labels = zip(*batch)
    max_len = max(t.shape[0] for t in tensors)
    batch_size = len(tensors)
    hidden_dim = tensors[0].shape[1]

    x = torch.zeros(batch_size, max_len, hidden_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    residue_labels = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, tensor in enumerate(tensors):
        seq_len = tensor.shape[0]
        x[i, :seq_len, :] = tensor
        mask[i, :seq_len] = True
        residue_labels[i, :seq_len] = float(labels[i])

    return x, mask, residue_labels


def _peek_embedding_dim(shard_dir: Path) -> int:
    """Obtiene la dimension de embedding leyendo un unico shard de referencia."""
    first_shard_path = sorted(shard_dir.glob("shard_*.pt"))[0]
    shard = torch.load(first_shard_path, weights_only=False)
    hidden_dim = int(shard["embeddings"][0].shape[1])
    del shard
    return hidden_dim


def train_antigenicity_cnn(train_shard_dir: Path, val_shard_dir: Path, save_path: Path) -> float:
    """Entrena la 1D-CNN de antigenicidad sobre matrices de Hellberg precomputadas.

    Args:
        train_shard_dir: Directorio de shards ``z_scale`` de entrenamiento.
        val_shard_dir: Directorio de shards ``z_scale`` de validacion.
        save_path: Ruta donde persistir el mejor ``state_dict``.

    Returns:
        La mejor perdida de validacion alcanzada.
    """
    torch.manual_seed(Settings.ANTIGENICITY_RANDOM_SEED)
    device = torch.device(Settings.DEVICE)

    model = AntigenicityCNN(
        in_channels=3,
        hidden_channels=Settings.ANTIGENICITY_CNN_CHANNELS,
        kernel_size=Settings.ANTIGENICITY_CNN_KERNEL_SIZE,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Settings.TRAINING_LEARNING_RATE,
        weight_decay=Settings.TRAINING_WEIGHT_DECAY,
    )
    loss_fn = nn.BCELoss()

    train_dataset = ShardIterableDataset(
        train_shard_dir, tensor_key="z_scale", shuffle=True, seed=Settings.TRAINING_SEED
    )
    val_dataset = ShardIterableDataset(
        val_shard_dir, tensor_key="z_scale", shuffle=False, seed=Settings.TRAINING_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Settings.TRAINING_BATCH_SIZE,
        collate_fn=collate_hellberg,
        drop_last=len(train_dataset) > Settings.TRAINING_BATCH_SIZE,
    )
    val_loader = DataLoader(val_dataset, batch_size=Settings.TRAINING_BATCH_SIZE, collate_fn=collate_hellberg)

    logger.info(
        "Entrenando 1D-CNN de antigenicidad: %d train / %d val secuencias.",
        len(train_dataset),
        len(val_dataset),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, Settings.TRAINING_MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x, mask)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            del x, mask, y, preds, loss

        gc.collect()
        train_loss = running_loss / max(n_batches, 1)
        val_loss = _evaluate_antigenicity(model, val_loader, loss_fn, device)
        log_memory_checkpoint(logger, f"antigenicity_train_epoch_{epoch}")

        logger.info(
            "Fase 1 | Epoca %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch,
            Settings.TRAINING_MAX_EPOCHS,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info("Nuevo mejor checkpoint de Fase 1 guardado en '%s' (val_loss=%.4f).", save_path, best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= Settings.TRAINING_EARLY_STOP_PATIENCE:
                logger.info(
                    "Early stopping en epoca %d (sin mejora en %d epocas consecutivas).",
                    epoch,
                    patience_counter,
                )
                break

    return best_val_loss


def _evaluate_antigenicity(
    model: AntigenicityCNN, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    """Calcula la perdida promedio de validacion para la 1D-CNN, sin gradientes."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            preds = model(x, mask)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
            n_batches += 1
            del x, mask, y, preds, loss

    gc.collect()
    return total_loss / max(n_batches, 1)


def train_residue_classifier(
    train_shard_dir: Path,
    val_shard_dir: Path,
    save_path: Path,
    fine_tune_from: Optional[Path] = None,
) -> float:
    """Entrena (o afina) el ``ResidueClassifier`` sobre embeddings ESM-2 precomputados.

    Args:
        train_shard_dir: Directorio de shards ``embeddings`` de entrenamiento.
        val_shard_dir: Directorio de shards ``embeddings`` de validacion.
        save_path: Ruta donde persistir el mejor ``state_dict``.
        fine_tune_from: Si se provee y el archivo existe, se cargan esos
            pesos como punto de partida (fine-tuning) en lugar de
            inicializar aleatoriamente. Util para corregir un sesgo
            especifico (p. ej. distribution shift en secuencias largas) sin
            perder la calibracion ya aprendida sobre el corpus original.

    Returns:
        La mejor perdida de validacion alcanzada.
    """
    device = torch.device(Settings.DEVICE)
    hidden_dim = _peek_embedding_dim(train_shard_dir)

    torch.manual_seed(Settings.ANTIGENICITY_RANDOM_SEED)
    model = ResidueClassifier(
        input_dim=hidden_dim,
        hidden_dim=Settings.RESIDUE_CLASSIFIER_HIDDEN_DIM,
        dropout=Settings.RESIDUE_CLASSIFIER_DROPOUT,
    ).to(device)

    if fine_tune_from is not None and fine_tune_from.exists():
        model.load_state_dict(torch.load(fine_tune_from, map_location=device))
        logger.info("ResidueClassifier: fine-tuning a partir de '%s'.", fine_tune_from)
    else:
        logger.info("ResidueClassifier: entrenamiento desde inicializacion aleatoria.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Settings.TRAINING_LEARNING_RATE,
        weight_decay=Settings.TRAINING_WEIGHT_DECAY,
    )

    train_dataset = ShardIterableDataset(
        train_shard_dir, tensor_key="embeddings", shuffle=True, seed=Settings.TRAINING_SEED
    )
    val_dataset = ShardIterableDataset(
        val_shard_dir, tensor_key="embeddings", shuffle=False, seed=Settings.TRAINING_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Settings.TRAINING_BATCH_SIZE,
        collate_fn=collate_esm2,
        drop_last=len(train_dataset) > Settings.TRAINING_BATCH_SIZE,
    )
    val_loader = DataLoader(val_dataset, batch_size=Settings.TRAINING_BATCH_SIZE, collate_fn=collate_esm2)

    logger.info(
        "Entrenando ResidueClassifier (hidden_dim=%d): %d train / %d val secuencias.",
        hidden_dim,
        len(train_dataset),
        len(val_dataset),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, Settings.TRAINING_MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, mask, residue_labels in train_loader:
            x, mask, residue_labels = x.to(device), mask.to(device), residue_labels.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = _masked_bce(preds, residue_labels, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            del x, mask, residue_labels, preds, loss

        gc.collect()
        train_loss = running_loss / max(n_batches, 1)
        val_loss = _evaluate_residue_classifier(model, val_loader, device)
        log_memory_checkpoint(logger, f"residue_classifier_train_epoch_{epoch}")

        logger.info(
            "Fase 2 | Epoca %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch,
            Settings.TRAINING_MAX_EPOCHS,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(
                "Nuevo mejor checkpoint de Fase 2 guardado en '%s' (val_loss=%.4f).", save_path, best_val_loss
            )
        else:
            patience_counter += 1
            if patience_counter >= Settings.TRAINING_EARLY_STOP_PATIENCE:
                logger.info(
                    "Early stopping en epoca %d (sin mejora en %d epocas consecutivas).",
                    epoch,
                    patience_counter,
                )
                break

    return best_val_loss


def _masked_bce(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropia cruzada binaria promediada primero por secuencia, luego por lote.

    CRITICO: promediar sobre TODAS las posiciones validas del lote de forma
    plana (``sum(loss) / sum(mask)``) hace que las secuencias largas (hasta
    1022 aa, p. ej. los hard negatives macromoleculares) dominen el gradiente
    en proporcion a su longitud, no a su conteo. Con hard negatives de ~560
    aa promedio conviviendo con peptidos de ~17 aa, ese esquema plano concentra
    >90% de la masa de perdida en las secuencias largas incluso siendo solo el
    30% de las secuencias -- colapsando la red a predecir ~0 en todas partes
    (sobre-correccion observada empiricamente: densidad 98% -> 0.00%).

    Este promedio en dos pasos (media intra-secuencia, luego media entre
    secuencias) pondera cada SECUENCIA por igual sin importar su longitud,
    preservando la proporcion 30/70 realmente pretendida a nivel de secuencia.

    Args:
        preds: Probabilidades predichas, forma ``(B, L_max)``.
        targets: Etiquetas objetivo, forma ``(B, L_max)``.
        mask: Mascara booleana de posiciones reales, forma ``(B, L_max)``.

    Returns:
        Escalar con la perdida promediada por secuencia.
    """
    per_element = F.binary_cross_entropy(preds, targets, reduction="none")
    mask_float = mask.float()
    per_sequence_loss = (per_element * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
    return per_sequence_loss.mean()


def _evaluate_residue_classifier(model: ResidueClassifier, loader: DataLoader, device: torch.device) -> float:
    """Calcula la perdida promedio de validacion para el ResidueClassifier, sin gradientes."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, mask, residue_labels in loader:
            x, mask, residue_labels = x.to(device), mask.to(device), residue_labels.to(device)
            preds = model(x)
            loss = _masked_bce(preds, residue_labels, mask)
            total_loss += loss.item()
            n_batches += 1
            del x, mask, residue_labels, preds, loss

    gc.collect()
    return total_loss / max(n_batches, 1)


def _prepare_baseline_engine(
    threshold: float, weights_path: Path
) -> Tuple[AntigenicityCNNEngine, Optional[Path]]:
    """Instancia un motor de Fase 1 con inicializacion aleatoria como linea base.

    Si ya existe un checkpoint entrenado de una ejecucion anterior, se mueve
    temporalmente a un archivo ``.pretrain_backup`` para forzar una linea base
    honesta (pesos aleatorios), en lugar de comparar contra un modelo ya
    calibrado.

    Args:
        threshold: Umbral de antigenicidad a usar en el motor de linea base.
        weights_path: Ruta del checkpoint de la 1D-CNN.

    Returns:
        Tupla ``(motor_linea_base, ruta_de_backup_o_None)``.
    """
    backup: Optional[Path] = None
    if weights_path.exists():
        backup = weights_path.with_name(weights_path.name + ".pretrain_backup")
        weights_path.rename(backup)
        logger.info("Checkpoint preexistente respaldado temporalmente en '%s'.", backup)

    engine = AntigenicityCNNEngine(threshold=threshold)
    return engine, backup


def _finalize_checkpoint(weights_path: Path, backup: Optional[Path]) -> None:
    """Resuelve el checkpoint de respaldo tras el entrenamiento.

    Args:
        weights_path: Ruta del checkpoint recien entrenado.
        backup: Ruta de respaldo devuelta por :func:`_prepare_baseline_engine`,
            o ``None`` si no existia un checkpoint previo.
    """
    if backup is None:
        return

    if weights_path.exists():
        backup.unlink(missing_ok=True)
        logger.info("Checkpoint previo descartado: el entrenamiento produjo uno mejor.")
    elif backup.exists():
        backup.rename(weights_path)
        logger.warning(
            "El entrenamiento no mejoro ningun checkpoint; se restauro el previo en '%s'.", weights_path
        )


def _run_benchmark(
    weights_path: Path, threshold: float, positive_fasta: Path, negative_fasta: Path
) -> BenchmarkReport:
    """Ejecuta la suite de auditoria cientifica con los pesos actualmente en disco."""
    engine = AntigenicityCNNEngine(threshold=threshold)
    suite = BenchmarkSuite(scorer=engine, threshold=threshold)
    return suite.run(positive_fasta, negative_fasta)


def measure_long_sequence_density(
    fasta_path: Path, antigenicity_threshold: float
) -> Optional[float]:
    """Mide la densidad de residuos predichos como epitopo sobre proteinas largas.

    Diagnostico directo del artefacto de "distribution shift": ejecuta la
    Fase 1 y Fase 2 completas (con los pesos actualmente en disco) sobre
    ``fasta_path`` e informa el porcentaje de residuos marcados como epitopo
    en el total de la(s) proteina(s), sin filtrar por el veredicto de
    antigenicidad de Fase 1 (interesa el comportamiento crudo del
    ``ResidueClassifier`` ante secuencias largas, independientemente de si la
    Fase 1 las habria descartado).

    Args:
        fasta_path: Ruta a un FASTA con una o mas proteinas nativas largas.
        antigenicity_threshold: Umbral de Fase 1 (solo afecta el score
            reportado, no filtra las secuencias que llegan a Fase 2 aqui).

    Returns:
        Densidad porcentual ``[0, 100]``, o ``None`` si el FASTA no aporto
        secuencias validas.
    """
    if not fasta_path.exists():
        return None

    records = FastaParser.parse(fasta_path, min_length=Settings.MIN_SEQUENCE_LENGTH)
    if not records:
        return None

    antigenicity_engine = AntigenicityCNNEngine(threshold=antigenicity_threshold)
    phase1_results = antigenicity_engine.run(records)
    del antigenicity_engine
    gc.collect()

    predictor = NativeESM2Engine()
    try:
        results = predictor.predict(phase1_results)
    finally:
        predictor.close()
        gc.collect()

    if not results:
        return None

    total_residues = sum(len(result.residues) for result in results)
    total_epitope = sum(sum(1 for res in result.residues if res.is_epitope) for result in results)
    return 100.0 * total_epitope / total_residues if total_residues else 0.0


def _parse_arguments() -> argparse.Namespace:
    """Define la interfaz de linea de comandos del entrenamiento standalone."""
    parser = argparse.ArgumentParser(
        description="Calibracion de la 1D-CNN de antigenicidad y el ResidueClassifier de epitopos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Settings.TRAINING_DATA_DIR)
    parser.add_argument("--features-dir", type=Path, default=Settings.FEATURES_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--skip-antigenicity", action="store_true", help="Omite el entrenamiento de la 1D-CNN de Fase 1."
    )
    parser.add_argument(
        "--skip-epitope", action="store_true", help="Omite el entrenamiento del ResidueClassifier de Fase 2."
    )
    parser.add_argument(
        "--fine-tune-epitope",
        action="store_true",
        help="Continua el entrenamiento del ResidueClassifier desde el checkpoint existente en vez de reinicializar.",
    )
    parser.add_argument(
        "--long-test-fasta",
        type=Path,
        default=None,
        help="FASTA con proteina(s) largas nativas para medir la densidad de epitopo antes/despues (diagnostico de distribution shift).",
    )
    return parser.parse_args()


def main() -> int:
    """Punto de entrada standalone: ``python -m src.training.trainer``."""
    args = _parse_arguments()

    if args.offline:
        Settings.apply_offline_mode()
    Settings.apply_thread_limits()

    test_positive = args.data_dir / "test_positive.fasta"
    test_negative = args.data_dir / "test_negative.fasta"
    has_test_set = test_positive.exists() and test_negative.exists()

    density_before: Optional[float] = None
    if args.long_test_fasta is not None:
        logger.info("Midiendo densidad de epitopo ANTES del (re)entrenamiento en '%s'...", args.long_test_fasta)
        density_before = measure_long_sequence_density(args.long_test_fasta, Settings.ANTIGENICITY_THRESHOLD)
        if density_before is not None:
            logger.info("Densidad ANTES: %.2f%%", density_before)

    baseline_report: Optional[BenchmarkReport] = None
    weights_path = Settings.ANTIGENICITY_CNN_WEIGHTS_PATH

    if not args.skip_antigenicity:
        logger.info("=== Fase 1: calibrando 1D-CNN de antigenicidad ===")
        backup: Optional[Path] = None
        if has_test_set:
            baseline_engine, backup = _prepare_baseline_engine(Settings.ANTIGENICITY_THRESHOLD, weights_path)
            baseline_suite = BenchmarkSuite(scorer=baseline_engine, threshold=Settings.ANTIGENICITY_THRESHOLD)
            baseline_report = baseline_suite.run(test_positive, test_negative)
            del baseline_engine
            gc.collect()
        else:
            logger.warning(
                "No se encontraron '%s'/'%s'; se omite la auditoria 'antes' del entrenamiento.",
                test_positive,
                test_negative,
            )

        best_val_loss = train_antigenicity_cnn(
            args.features_dir / "train" / "hellberg",
            args.features_dir / "val" / "hellberg",
            weights_path,
        )
        _finalize_checkpoint(weights_path, backup)
        logger.info("1D-CNN de antigenicidad calibrada. Mejor val_loss=%.4f", best_val_loss)

    if not args.skip_epitope:
        logger.info("=== Fase 2: calibrando ResidueClassifier (embeddings ESM-2) ===")
        fine_tune_source = (
            Settings.RESIDUE_CLASSIFIER_WEIGHTS_PATH if args.fine_tune_epitope else None
        )
        best_val_loss_res = train_residue_classifier(
            args.features_dir / "train" / "esm2",
            args.features_dir / "val" / "esm2",
            Settings.RESIDUE_CLASSIFIER_WEIGHTS_PATH,
            fine_tune_from=fine_tune_source,
        )
        logger.info("ResidueClassifier calibrado. Mejor val_loss=%.4f", best_val_loss_res)

    if has_test_set:
        logger.info("=== Verificacion post-entrenamiento (test set, pesos calibrados) ===")
        trained_report = _run_benchmark(
            weights_path, Settings.ANTIGENICITY_THRESHOLD, test_positive, test_negative
        )

        if baseline_report is not None:
            print("\n>>> ANTES DEL ENTRENAMIENTO (1D-CNN con pesos aleatorios) <<<")
            print_benchmark_report(baseline_report)

        print(">>> DESPUES DEL ENTRENAMIENTO (1D-CNN calibrada) <<<")
        print_benchmark_report(trained_report)

        if baseline_report is not None:
            logger.info(
                "Mejora medida en test: ROC-AUC %.4f -> %.4f (delta=%.4f) | "
                "Especificidad %.4f -> %.4f (delta=%.4f)",
                baseline_report.roc_auc,
                trained_report.roc_auc,
                trained_report.roc_auc - baseline_report.roc_auc,
                baseline_report.specificity,
                trained_report.specificity,
                trained_report.specificity - baseline_report.specificity,
            )

        if args.long_test_fasta is not None:
            logger.info(
                "Midiendo densidad de epitopo DESPUES del (re)entrenamiento en '%s'...",
                args.long_test_fasta,
            )
            density_after = measure_long_sequence_density(
                args.long_test_fasta, Settings.ANTIGENICITY_THRESHOLD
            )

            print("\n┌" + "─" * 71 + "┐")
            print(f"│{'REPORTE EJECUTIVO: CORRECCION DE DISTRIBUTION SHIFT':^71}│")
            print("├" + "─" * 71 + "┤")
            print(f"│ Proteina de prueba: {str(args.long_test_fasta):<49} │")
            if density_before is not None:
                print(f"│ Densidad de epitopo ANTES  : {density_before:>6.2f}%{'':<32} │")
            else:
                print(f"│ Densidad de epitopo ANTES  : {'N/D (sin medicion previa)':<40} │")
            if density_after is not None:
                print(f"│ Densidad de epitopo DESPUES: {density_after:>6.2f}%{'':<32} │")
                fisiologico = 5.0 <= density_after <= 20.0
                veredicto = "DENTRO DEL RANGO FISIOLOGICO (5-20%)" if fisiologico else "FUERA DE RANGO — revisar"
                print(f"│ Veredicto de densidad      : {veredicto:<40} │")
            print(f"│ ROC-AUC en test independiente: {trained_report.roc_auc:.4f}{'':<28} │")
            roc_ok = trained_report.roc_auc >= 0.85
            print(f"│ Robustez ROC-AUC >= 0.85   : {'SI' if roc_ok else 'NO':<40} │")
            print("└" + "─" * 71 + "┘\n")

            if density_before is not None and density_after is not None:
                logger.info(
                    "Correccion de distribution shift: densidad %.2f%% -> %.2f%% "
                    "(delta=%.2f puntos porcentuales).",
                    density_before,
                    density_after,
                    density_after - density_before,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
