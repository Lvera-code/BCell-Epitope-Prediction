"""Configuracion centralizada del pipeline: rutas, umbrales, motores y hardware.

Todos los parametros ajustables se resuelven desde variables de entorno con
valores por defecto conservadores para el entorno objetivo (WSL Ubuntu, Intel
i7 12 nucleos, 16 GB RAM, sin GPU). Esto permite reconfigurar el pipeline en
despliegues HPC/CI sin tocar codigo fuente.
"""

import os
from pathlib import Path

import torch


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_path(key: str) -> "Path | None":
    raw = os.environ.get(key)
    return Path(raw) if raw else None


class Settings:
    """Punto unico de verdad para toda configuracion del pipeline."""

    # --- Rutas de datos ---
    RAW_FASTA_PATH: Path = Path(_env_str("PIPELINE_INPUT_FASTA", "data/raw/candidatos.fasta"))
    PROCESSED_DIR: Path = Path(_env_str("PIPELINE_PROCESSED_DIR", "data/processed"))
    MODELS_DIR: Path = Path(_env_str("PIPELINE_MODELS_DIR", "models"))

    # --- Saneamiento (Modulo de Aduana) ---
    MIN_SEQUENCE_LENGTH: int = _env_int("PIPELINE_MIN_SEQ_LENGTH", 9)

    # --- Fase 1: Cribado de Antigenicidad (1D-CNN sobre Escalas Z de Hellberg) ---
    ANTIGENICITY_THRESHOLD: float = _env_float("ANTIGENICITY_THRESHOLD", 0.60)
    ANTIGENICITY_CNN_WEIGHTS_PATH: Path = Path(
        _env_str("ANTIGENICITY_CNN_WEIGHTS_PATH", "models/antigenicity_cnn.pt")
    )
    ANTIGENICITY_MAX_RESIDUES_PER_BATCH: int = _env_int(
        "ANTIGENICITY_MAX_RESIDUES_PER_BATCH", 20000
    )
    ANTIGENICITY_MAX_ITEMS_PER_BATCH: int = _env_int("ANTIGENICITY_MAX_ITEMS_PER_BATCH", 64)
    ANTIGENICITY_MEMORY_BUDGET_MB: float = _env_float("ANTIGENICITY_MEMORY_BUDGET_MB", 3000.0)
    ANTIGENICITY_CNN_CHANNELS: int = _env_int("ANTIGENICITY_CNN_CHANNELS", 32)
    ANTIGENICITY_CNN_KERNEL_SIZE: int = _env_int("ANTIGENICITY_CNN_KERNEL_SIZE", 5)
    ANTIGENICITY_RANDOM_SEED: int = _env_int("ANTIGENICITY_RANDOM_SEED", 42)

    # El Global Max Pooling de la 1D-CNN es estadisticamente no-decreciente en
    # longitud (mas posiciones = mas intentos de activacion extrema), pero el
    # clasificador lineal solo vio, en entrenamiento, vectores "pooled" de
    # peptidos IEDB de 9-25aa (ver TRAINING_MIN/MAX_PEPTIDE_LEN). Puntuar una
    # proteina completa (cientos/miles de aa) de una sola pasada extrapola muy
    # fuera de ese rango y hunde el logit. Se trocea todo candidato mas largo
    # que ANTIGENICITY_WINDOW_SIZE en ventanas peptidicas del mismo tamano que
    # el entrenamiento y se agrega por maximo (misma estrategia de sliding
    # window que ya usa la Fase 2 via ``compute_sliding_windows``).
    ANTIGENICITY_WINDOW_SIZE: int = _env_int("ANTIGENICITY_WINDOW_SIZE", 25)
    ANTIGENICITY_WINDOW_OVERLAP: int = _env_int("ANTIGENICITY_WINDOW_OVERLAP", 15)

    # Agregar por max() puro entre ventanas es estadisticamente fragil: con un
    # ~28% de falsos positivos por ventana (precision empirica del clasificador
    # de peptidos), una proteina de cientos de ventanas acumula casi con
    # certeza al menos una ventana espuria (problema de comparaciones
    # multiples). Un epitopo real es un parche contiguo (>=5aa) que activa
    # VARIAS ventanas solapadas consecutivas a la vez; el ruido de una sola
    # ventana aislada, no. Se exige corroboracion: la puntuacion final es el
    # maximo, sobre todas las corridas de ``ANTIGENICITY_CORROBORATION_WINDOWS``
    # ventanas consecutivas, del MINIMO dentro de cada corrida (auditado
    # empiricamente sobre el hold-out: recupera ROC-AUC vs. hard-negatives
    # macromoleculares de 0.43 a 0.81 sin perder el control positivo real).
    ANTIGENICITY_CORROBORATION_WINDOWS: int = _env_int("ANTIGENICITY_CORROBORATION_WINDOWS", 3)

    # Regularizacion de la 1D-CNN (mitiga sobreajuste sobre el dataset ampliado).
    ANTIGENICITY_CNN_DROPOUT: float = _env_float("ANTIGENICITY_CNN_DROPOUT", 0.35)

    # --- Entrenamiento de Fase 1: Focal Loss + LR scheduling + early stopping ---
    # Focal Loss (Lin et al., 2017) sustituye a BCEWithLogitsLoss+pos_weight:
    # concentra el gradiente en los ejemplos dificiles (hard negatives
    # macromoleculares) en vez de tratarlos igual que negativos triviales.
    ANTIGENICITY_FOCAL_LOSS_GAMMA: float = _env_float("ANTIGENICITY_FOCAL_LOSS_GAMMA", 2.0)
    ANTIGENICITY_MAX_EPOCHS: int = _env_int("ANTIGENICITY_MAX_EPOCHS", 200)
    ANTIGENICITY_EARLY_STOP_PATIENCE: int = _env_int("ANTIGENICITY_EARLY_STOP_PATIENCE", 20)
    ANTIGENICITY_LR_SCHEDULER_FACTOR: float = _env_float("ANTIGENICITY_LR_SCHEDULER_FACTOR", 0.5)
    ANTIGENICITY_LR_SCHEDULER_PATIENCE: int = _env_int("ANTIGENICITY_LR_SCHEDULER_PATIENCE", 8)

    # Directorio local vendorizado de ESM-2 (pesos + tokenizer ya guardados en
    # disco via save_pretrained()); si existe, se prioriza sobre la cache
    # generica de HuggingFace para arranque 100% autonomo. Ver
    # ``Settings.resolve_esm2_source()``.
    ESM_LOCAL_MODEL_DIR: Path = Path(_env_str("ESM_LOCAL_MODEL_DIR", "models/esm2_local"))

    # --- Calibracion de Platt (Fase 1) ---
    # La 1D-CNN emite logits crudos (sin sigmoide); este artefacto (A, B de una
    # regresion logistica 1D ajustada sobre un hold-out estratificado que NUNCA
    # participa del backprop) mapea logit -> probabilidad calibrada mediante
    # sigmoid(A * logit + B), corrigiendo la compresion de rango causada por el
    # desbalance de clases del dataset de entrenamiento.
    ANTIGENICITY_CALIBRATION_PATH: Path = Path(
        _env_str("ANTIGENICITY_CALIBRATION_PATH", "models/antigenicity_calibration.pkl")
    )
    CALIBRATION_HOLDOUT_RATIO: float = _env_float("CALIBRATION_HOLDOUT_RATIO", 0.10)

    # --- Fase 2: Motor de Prediccion de Epitopos (Patron Adaptador) ---
    PREDICTOR_ENGINE: str = _env_str("PREDICTOR_ENGINE", "esm2").strip().lower()

    ESM_MODEL_NAME: str = _env_str("ESM_MODEL_NAME", "facebook/esm2_t30_150M_UR50D")
    ESM_HIDDEN_DIM: int = _env_int("ESM_HIDDEN_DIM", 640)
    ESM_MAX_RESIDUES_PER_BATCH: int = _env_int("ESM_MAX_RESIDUES_PER_BATCH", 4000)
    ESM_MAX_ITEMS_PER_BATCH: int = _env_int("ESM_MAX_ITEMS_PER_BATCH", 8)
    ESM_MEMORY_BUDGET_MB: float = _env_float("ESM_MEMORY_BUDGET_MB", 6000.0)
    RESIDUE_CLASSIFIER_WEIGHTS_PATH: Path = Path(
        _env_str("RESIDUE_CLASSIFIER_WEIGHTS_PATH", "models/residue_classifier.pt")
    )
    RESIDUE_CLASSIFIER_HIDDEN_DIM: int = _env_int("RESIDUE_CLASSIFIER_HIDDEN_DIM", 128)
    RESIDUE_CLASSIFIER_DROPOUT: float = _env_float("RESIDUE_CLASSIFIER_DROPOUT", 0.10)

    EPITOPE_THRESHOLD: float = _env_float("EPITOPE_THRESHOLD", 0.35)
    # Ventana del filtro de suavizado espacial Savitzky-Golay: los epitopos B
    # (lineales y conformacionales) son parches fisicos continuos de 6-15
    # residuos, no picos aislados; w=9 es el tamano tipico de dicho parche.
    EPITOPE_SMOOTHING_WINDOW: int = _env_int("EPITOPE_SMOOTHING_WINDOW", 9)
    EPITOPE_SMOOTHING_POLYORDER: int = _env_int("EPITOPE_SMOOTHING_POLYORDER", 2)
    EPITOPE_MIN_REGION_LENGTH: int = _env_int("EPITOPE_MIN_REGION_LENGTH", 5)
    # Limite fisico de contexto de ESM-2 (1024 tokens = 1022 aa + BOS/EOS).
    ESM_MAX_SEQUENCE_LENGTH: int = _env_int("ESM_MAX_SEQUENCE_LENGTH", 1022)
    # Sliding Window Stitcher: permite procesar macromoleculas de longitud
    # arbitraria (Spike SARS-CoV-2, ortologos gigantes de Malaria, etc.) sin
    # truncar ni un residuo y sin exceder nunca el limite fisico de ESM-2 en
    # un unico forward pass. window < limite fisico deja margen de seguridad;
    # overlap acota la zona de fusion ponderada (tapering) entre ventanas.
    ESM_SLIDING_WINDOW_SIZE: int = _env_int("ESM_SLIDING_WINDOW_SIZE", 1000)
    ESM_SLIDING_WINDOW_OVERLAP: int = _env_int("ESM_SLIDING_WINDOW_OVERLAP", 200)

    # --- CLIWrapperEngine (interoperabilidad HPC / binarios externos) ---
    BEPIPRED_CLI_PATH: str = _env_str("BEPIPRED_CLI_PATH", "bepipred-cli")
    CLI_TIMEOUT_SECONDS: int = _env_int("CLI_TIMEOUT_SECONDS", 300)
    CLI_TEMP_DIR: "Path | None" = _env_optional_path("CLI_TEMP_DIR")

    # --- Hardware ---
    # El entorno objetivo es CPU puro (Intel i7, sin GPU); se detecta CUDA solo
    # como cortesia defensiva, nunca se asume disponible.
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_NUM_THREADS: int = _env_int("TORCH_NUM_THREADS", os.cpu_count() or 12)

    # --- Modo offline (sin llamadas de red a HuggingFace Hub) ---
    OFFLINE_MODE: bool = _env_bool("PIPELINE_OFFLINE", False)

    # --- Suite de Auditoria Cientifica ---
    BENCHMARK_POSITIVE_FASTA: "Path | None" = _env_optional_path("BENCHMARK_POSITIVE_FASTA")
    BENCHMARK_NEGATIVE_FASTA: "Path | None" = _env_optional_path("BENCHMARK_NEGATIVE_FASTA")

    # --- Curacion de dataset de entrenamiento (src/training/dataset_prep.py) ---
    TRAINING_DATA_DIR: Path = Path(_env_str("TRAINING_DATA_DIR", "data/training"))
    FEATURES_DIR: Path = Path(_env_str("FEATURES_DIR", "data/features"))
    TRAINING_N_POSITIVE: int = _env_int("TRAINING_N_POSITIVE", 1000)
    TRAINING_N_NEGATIVE: int = _env_int("TRAINING_N_NEGATIVE", 1000)
    TRAINING_MIN_PEPTIDE_LEN: int = _env_int("TRAINING_MIN_PEPTIDE_LEN", 9)
    TRAINING_MAX_PEPTIDE_LEN: int = _env_int("TRAINING_MAX_PEPTIDE_LEN", 25)

    # --- Mineria de negativos dificiles (hard negative mining, Fase 2) ---
    # Fragmentos macromoleculares inertes largos (housekeeping) etiquetados
    # 100% negativos a nivel de residuo, para corregir el sesgo de "distribution
    # shift" donde el ResidueClassifier, entrenado solo con peptidos cortos,
    # hiper-activaba (~98% densidad) sobre proteinas nativas largas.
    HARD_NEGATIVE_RATIO: float = _env_float("HARD_NEGATIVE_RATIO", 0.30)
    HARD_NEGATIVE_MIN_LEN: int = _env_int("HARD_NEGATIVE_MIN_LEN", 100)
    HARD_NEGATIVE_MAX_LEN: int = _env_int("HARD_NEGATIVE_MAX_LEN", 1022)
    TRAIN_SPLIT_RATIO: float = _env_float("TRAIN_SPLIT_RATIO", 0.8)
    VAL_SPLIT_RATIO: float = _env_float("VAL_SPLIT_RATIO", 0.1)
    TEST_SPLIT_RATIO: float = _env_float("TEST_SPLIT_RATIO", 0.1)
    TRAINING_SEED: int = _env_int("TRAINING_SEED", 42)

    IEDB_API_BASE: str = _env_str("IEDB_API_BASE", "https://query-api.iedb.org")
    UNIPROT_API_BASE: str = _env_str("UNIPROT_API_BASE", "https://rest.uniprot.org")
    HTTP_TIMEOUT_SECONDS: int = _env_int("HTTP_TIMEOUT_SECONDS", 30)
    HTTP_MAX_RETRIES: int = _env_int("HTTP_MAX_RETRIES", 3)

    # --- Pre-extraccion de features (src/training/feature_extractor.py) ---
    FEATURE_SHARD_SIZE: int = _env_int("FEATURE_SHARD_SIZE", 64)

    # --- Entrenamiento (src/training/trainer.py) ---
    TRAINING_BATCH_SIZE: int = _env_int("TRAINING_BATCH_SIZE", 32)
    TRAINING_MAX_EPOCHS: int = _env_int("TRAINING_MAX_EPOCHS", 100)
    TRAINING_EARLY_STOP_PATIENCE: int = _env_int("TRAINING_EARLY_STOP_PATIENCE", 8)
    TRAINING_LEARNING_RATE: float = _env_float("TRAINING_LEARNING_RATE", 1e-3)
    TRAINING_WEIGHT_DECAY: float = _env_float("TRAINING_WEIGHT_DECAY", 1e-2)

    @classmethod
    def setup_directories(cls) -> None:
        """Crea los directorios de datos requeridos si aun no existen."""
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_FASTA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def apply_offline_mode(cls) -> None:
        """Fija las variables de entorno de HuggingFace para forzar modo offline.

        Debe invocarse antes de instanciar cualquier tokenizer/modelo de
        ``transformers`` para evitar intentos de resolucion de red en clusters
        HPC sin salida a internet.
        """
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cls.OFFLINE_MODE = True

    @classmethod
    def apply_thread_limits(cls) -> None:
        """Fija el numero de hilos de PyTorch al conteo de nucleos disponibles.

        Evita sobre-suscripcion de hilos (thrashing) cuando el pipeline corre
        junto a otros procesos en la misma maquina de 12 nucleos.
        """
        torch.set_num_threads(cls.TORCH_NUM_THREADS)

    @classmethod
    def resolve_esm2_source(cls) -> str:
        """Resuelve el origen de carga de ESM-2: vendorizado local o Hub.

        Prioriza ``ESM_LOCAL_MODEL_DIR`` (pesos + tokenizer ya volcados a
        disco dentro del repo via ``save_pretrained()``) para un arranque
        100% autonomo que no dependa de la cache generica de HuggingFace. Si
        el directorio no existe o esta incompleto, cae al nombre del Hub
        (resuelto contra la cache local si ``OFFLINE_MODE`` esta activo).

        Returns:
            Ruta local (str) o nombre de modelo del Hub, listo para pasar a
            ``from_pretrained()``.
        """
        local_dir = cls.ESM_LOCAL_MODEL_DIR
        if (local_dir / "config.json").exists():
            return str(local_dir)
        return cls.ESM_MODEL_NAME
