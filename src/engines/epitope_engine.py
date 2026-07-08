"""Fase 2: Inferencia de Epitopos de Celulas B mediante Patron Adaptador (Strategy).

Define una interfaz comun (:class:`BaseEpitopePredictor`) con dos motores
intercambiables en tiempo de ejecucion:

* :class:`NativeESM2Engine`: carga ``facebook/esm2_t30_150M_UR50D`` directo a
  RAM, extrae embeddings de la ultima capa (640 dimensiones) y los clasifica
  por residuo con un MLP (:class:`ResidueClassifier`). Optimizado para
  desarrollo agil local en CPU.
* :class:`CLIWrapperEngine`: ejecuta una herramienta externa por subprocess
  (p. ej. ``bepipred-cli``), desacoplando el pipeline de la version instalada
  del binario. Optimizado para clusteres HPC y blindaje ante actualizaciones
  (p. ej. migracion a BepiPred-4.0).

:class:`EpitopePredictorFactory` orquesta la seleccion del motor segun
``Settings.PREDICTOR_ENGINE``.
"""

import csv
import gc
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from huggingface_hub.utils import logging as hf_hub_logging
from transformers import AutoTokenizer, EsmModel
from transformers import logging as hf_logging

from src.config.settings import Settings
from src.models import AntigenicityResult, EpitopeResidue, EpitopeResult
from src.utils.exceptions import CLIWrapperError, ModelLoadError
from src.utils.batching import dynamic_batches
from src.utils.logger_config import setup_logger
from src.utils.memory_profiler import log_memory_checkpoint, warn_if_over_budget

# Silencia las tablas de "UNEXPECTED / MISSING keys", el aviso de HF_TOKEN/Hub
# no autenticado y la barra de progreso de descarga de pesos de HuggingFace.
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
hf_hub_logging.set_verbosity_error()

logger = setup_logger(__name__)


def extract_contiguous_regions(
    residues: Sequence[EpitopeResidue], min_region_length: int
) -> List[Tuple[int, int]]:
    """Colapsa residuos positivos contiguos en regiones de epitopo lineal.

    Args:
        residues: Predicciones por residuo, ordenadas por posicion ascendente.
        min_region_length: Longitud minima (en residuos) para que una racha
            contigua de positivos se reporte como region de epitopo.

    Returns:
        Lista de tuplas ``(inicio, fin)`` 1-indexadas e inclusivas.
    """
    regions: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for residue in residues:
        if residue.is_epitope:
            if start is None:
                start = residue.position
        else:
            if start is not None and (residue.position - start) >= min_region_length:
                regions.append((start, residue.position - 1))
            start = None

    if start is not None and (len(residues) - start + 1) >= min_region_length:
        regions.append((start, len(residues)))

    return regions


def apply_spatial_smoothing(
    raw_probabilities: Sequence[float],
    window: int = Settings.EPITOPE_SMOOTHING_WINDOW,
    polyorder: int = Settings.EPITOPE_SMOOTHING_POLYORDER,
) -> List[float]:
    """Suaviza el vector de probabilidades por residuo con un filtro Savitzky-Golay.

    Los epitopos B (lineales y conformacionales) son parches fisicos
    continuos de 6-15 residuos en la superficie proteica, no picos aislados.
    Un filtro Savitzky-Golay (ajuste polinomico local de grado ``polyorder``
    sobre una ventana deslizante de ``window`` residuos) suprime el ruido
    estadistico de 1-2 residuos preservando la forma de las crestas
    continuas de alta probabilidad -- a diferencia de una media movil
    rectangular, que aplanaria tambien las crestas reales.

    Se aplica identicamente en ambos adaptadores de Fase 2
    (:class:`NativeESM2Engine` y :class:`CLIWrapperEngine`), garantizando que
    el post-procesamiento espacial sea independiente del motor de inferencia
    subyacente.

    Args:
        raw_probabilities: Probabilidades crudas por residuo, en orden de
            secuencia.
        window: Longitud de la ventana de suavizado. Se fuerza a impar
            (requisito de ``savgol_filter``) incrementando en 1 si es par.
            Por defecto, ``Settings.EPITOPE_SMOOTHING_WINDOW`` (9 residuos,
            el tamano tipico de un parche epitopico continuo).
        polyorder: Grado del polinomio local ajustado dentro de cada ventana.

    Returns:
        Probabilidades suavizadas, recortadas al rango ``[0, 1]``, de la
        misma longitud que ``raw_probabilities``. Si la secuencia es mas
        corta que la ventana efectiva, se devuelve sin modificar.
    """
    values = np.asarray(raw_probabilities, dtype=np.float64)
    effective_window = window if window % 2 == 1 else window + 1

    if len(values) < effective_window:
        return values.tolist()

    smoothed = savgol_filter(values, window_length=effective_window, polyorder=polyorder, mode="nearest")
    return np.clip(smoothed, 0.0, 1.0).tolist()


def compute_sliding_windows(seq_len: int, window_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Calcula los limites de las ventanas deslizantes que cubren una secuencia completa.

    ESM-2 trunca fatalmente cualquier secuencia que exceda su limite fisico de
    contexto (1024 tokens). Para procesar macromoleculas de longitud
    arbitraria (glucoproteina Spike, ortologos gigantes de Malaria, etc.) sin
    perder un solo residuo, la secuencia se divide en ventanas de a lo sumo
    ``window_size`` residuos, solapadas en ``overlap`` residuos con sus
    vecinas inmediatas. La union de todas las ventanas cubre exactamente
    ``[0, seq_len)``.

    Args:
        seq_len: Longitud total de la secuencia.
        window_size: Longitud maxima de cada ventana (aa). Debe ser menor que
            ``Settings.ESM_MAX_SEQUENCE_LENGTH`` para dejar margen de
            seguridad frente al limite fisico de ESM-2.
        overlap: Solapamiento deseado entre ventanas consecutivas (aa).

    Returns:
        Lista ordenada de tuplas ``(start, end)`` 0-indexadas y semi-abiertas
        (``end`` exclusivo). Si ``seq_len <= window_size``, se devuelve una
        unica ventana que cubre la secuencia completa (caso normal, identico
        al comportamiento sin sliding window).

    Raises:
        ValueError: Si ``overlap >= window_size`` (el stride resultante no
            seria positivo y el algoritmo no terminaria).
    """
    if seq_len <= 0:
        return []
    if seq_len <= window_size:
        return [(0, seq_len)]

    stride = window_size - overlap
    if stride <= 0:
        raise ValueError(
            f"El solapamiento ({overlap}) debe ser estrictamente menor que el "
            f"tamano de ventana ({window_size})."
        )

    windows: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + window_size, seq_len)
        windows.append((start, end))
        if end >= seq_len:
            break
        start += stride

    return windows


def stitch_window_probabilities(
    window_results: Sequence[Tuple[int, int, Sequence[float]]],
    seq_len: int,
    overlap: int,
) -> List[float]:
    """Funde probabilidades de ventanas solapadas mediante tapering lineal (cross-fade).

    Cada ventana aporta peso maximo (1.0) en su region central y una rampa
    lineal en sus ``overlap`` residuos de solape con la ventana vecina (0 -> 1
    al entrar desde la izquierda si no es la primera ventana; 1 -> 0 al salir
    por la derecha si no es la ultima). Esto evita cortes abruptos (hard
    boundaries) exactamente en las fronteras de ventana, que de otro modo
    introducirian discontinuidades artificiales que el filtro Savitzky-Golay
    posterior interpretaria como bordes reales.

    El valor final en cada posicion es el promedio ponderado de todas las
    ventanas que la cubren (equivalente a un overlap-add normalizado).

    Args:
        window_results: Tuplas ``(start, end, probabilidades_crudas)`` de cada
            ventana, en cualquier orden (se reordenan internamente por
            ``start``).
        seq_len: Longitud total de la secuencia original.
        overlap: Solapamiento nominal configurado entre ventanas consecutivas.

    Returns:
        Vector de probabilidades fusionado, de longitud ``seq_len``.
    """
    ordered = sorted(window_results, key=lambda entry: entry[0])
    accumulated = np.zeros(seq_len, dtype=np.float64)
    weight_sum = np.zeros(seq_len, dtype=np.float64)
    n_windows = len(ordered)

    for idx, (start, end, probs) in enumerate(ordered):
        length = end - start
        if length <= 0:
            continue

        probs_arr = np.asarray(probs, dtype=np.float64)
        weights = np.ones(length, dtype=np.float64)
        effective_overlap = min(overlap, length // 2) if length > 1 else 0

        if idx > 0 and effective_overlap > 0:
            ramp_in = np.linspace(1.0 / effective_overlap, 1.0, effective_overlap)
            weights[:effective_overlap] = np.minimum(weights[:effective_overlap], ramp_in)

        if idx < n_windows - 1 and effective_overlap > 0:
            ramp_out = np.linspace(1.0, 1.0 / effective_overlap, effective_overlap)
            weights[-effective_overlap:] = np.minimum(weights[-effective_overlap:], ramp_out)

        accumulated[start:end] += probs_arr * weights
        weight_sum[start:end] += weights

    safe_weight_sum = np.clip(weight_sum, 1e-9, None)
    return (accumulated / safe_weight_sum).tolist()


class _WindowJob(NamedTuple):
    """Unidad de trabajo para el Sliding Window Stitcher: una ventana de una secuencia."""

    item_id: str
    start: int
    end: int
    sequence: str


class BaseEpitopePredictor(ABC):
    """Interfaz Strategy/Adaptador para motores de prediccion de epitopos."""

    @abstractmethod
    def predict(self, items: Sequence[AntigenicityResult]) -> List[EpitopeResult]:
        """Predice epitopos de celulas B para secuencias que superaron la Fase 1.

        Args:
            items: Resultados de Fase 1 marcados como antigenicos.

        Returns:
            Lista de :class:`~src.models.EpitopeResult`, uno por secuencia de
            entrada.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Libera recursos del motor (modelos en RAM, procesos, temporales).

        Implementacion por defecto sin efecto; los motores que mantengan
        estado pesado (modelos cargados) deben sobrescribirla.
        """
        return None


class ResidueClassifier(nn.Module):
    """MLP en PyTorch que clasifica embeddings ESM-2 por residuo.

    Se elige un MLP nativo de PyTorch (en lugar de un Random Forest de
    scikit-learn) para mantener un unico framework de tensores en todo el
    pipeline de inferencia: evita conversiones numpy<->torch adicionales y
    problemas de compatibilidad de serializacion entre versiones de sklearn.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        """Inicializa el clasificador.

        Args:
            input_dim: Dimension del embedding de entrada (640 para
                ``esm2_t30_150M_UR50D``).
            hidden_dim: Dimension de la capa oculta.
            dropout: Probabilidad de dropout aplicada tras la capa oculta.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Clasifica un lote de embeddings por residuo.

        Args:
            embeddings: Tensor de forma ``(B, L, input_dim)``.

        Returns:
            Tensor de probabilidades de forma ``(B, L)``.
        """
        logits = self.net(embeddings).squeeze(-1)
        return self.sigmoid(logits)


class NativeESM2Engine(BaseEpitopePredictor):
    """Motor local: ESM-2 + MLP de clasificacion residuo a residuo, en RAM."""

    def __init__(self):
        """Carga el modelo ESM-2 y el clasificador de residuos a memoria.

        Raises:
            ModelLoadError: Si el tokenizer o el modelo no pueden cargarse
                (red no disponible en modo offline, nombre de modelo invalido,
                pesos de clasificador corruptos).
        """
        self.device = torch.device(Settings.DEVICE)
        esm2_source = Settings.resolve_esm2_source()
        logger.info(
            f"Cargando modelo ESM-2 desde '{esm2_source}' en dispositivo '{self.device}'..."
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(esm2_source)
            self.model = EsmModel.from_pretrained(esm2_source).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise ModelLoadError(
                f"Fallo critico al cargar el modelo ESM-2 desde '{esm2_source}': {exc}"
            ) from exc

        hidden_dim = self.model.config.hidden_size
        if hidden_dim != Settings.ESM_HIDDEN_DIM:
            logger.warning(
                "La dimension oculta real del modelo (%d) difiere de "
                "Settings.ESM_HIDDEN_DIM (%d). Se usa el valor real del modelo.",
                hidden_dim,
                Settings.ESM_HIDDEN_DIM,
            )

        torch.manual_seed(Settings.ANTIGENICITY_RANDOM_SEED)
        self.residue_classifier = ResidueClassifier(
            input_dim=hidden_dim,
            hidden_dim=Settings.RESIDUE_CLASSIFIER_HIDDEN_DIM,
            dropout=Settings.RESIDUE_CLASSIFIER_DROPOUT,
        ).to(self.device)

        weights_path = Settings.RESIDUE_CLASSIFIER_WEIGHTS_PATH
        if weights_path is not None and weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.residue_classifier.load_state_dict(state_dict)
            except Exception as exc:
                raise ModelLoadError(
                    f"Fallo al cargar pesos del ResidueClassifier desde '{weights_path}': {exc}"
                ) from exc
        else:
            logger.warning(
                "No se encontraron pesos entrenados para el ResidueClassifier "
                "(RESIDUE_CLASSIFIER_WEIGHTS_PATH no configurado o inexistente). "
                "La cabeza de clasificacion opera con inicializacion aleatoria "
                "deterministica: entrenar sobre un corpus IEDB antes de produccion."
            )

        self.residue_classifier.eval()

    def predict(self, items: Sequence[AntigenicityResult]) -> List[EpitopeResult]:
        """Predice epitopos residuo a residuo usando embeddings ESM-2.

        Secuencias mas largas que ``Settings.ESM_SLIDING_WINDOW_SIZE`` se
        procesan mediante un Sliding Window Stitcher: se dividen en ventanas
        solapadas que nunca exceden el limite fisico de contexto de ESM-2,
        cada ventana se infiere de forma independiente bajo
        ``torch.no_grad()`` (memoria acotada, sin importar la longitud total
        de la proteina), y las probabilidades resultantes se funden con
        tapering lineal (:func:`stitch_window_probabilities`) antes de un
        unico paso de suavizado espacial sobre la secuencia completa. Ningun
        residuo se trunca, sin importar cuan larga sea la macromolecula de
        entrada.

        Args:
            items: Resultados de Fase 1 que superaron el umbral de antigenicidad.

        Returns:
            Lista de :class:`~src.models.EpitopeResult`, en el mismo orden que
            ``items``.
        """
        if not items:
            return []

        window_size = Settings.ESM_SLIDING_WINDOW_SIZE
        overlap = Settings.ESM_SLIDING_WINDOW_OVERLAP

        jobs: List[_WindowJob] = []
        multi_window_ids: List[str] = []

        for item in items:
            sequence = item.record.sequence
            windows = compute_sliding_windows(len(sequence), window_size, overlap)

            if len(windows) > 1:
                multi_window_ids.append(item.record.id)
                logger.info(
                    "Secuencia '%s' (%d aa) supera la ventana segura de ESM-2 (%d aa): "
                    "dividida en %d ventanas solapadas (overlap=%d aa) via Sliding Window Stitcher.",
                    item.record.id,
                    len(sequence),
                    window_size,
                    len(windows),
                    overlap,
                )

            for start, end in windows:
                jobs.append(_WindowJob(item_id=item.record.id, start=start, end=end, sequence=sequence[start:end]))

        if multi_window_ids:
            logger.info(
                "Fase 2 (ESM-2): %d de %d secuencias requirieron Sliding Window Stitcher "
                "(%d ventanas totales a procesar).",
                len(multi_window_ids),
                len(items),
                len(jobs),
            )

        batches = dynamic_batches(
            items=jobs,
            length_fn=lambda job: len(job.sequence),
            max_residues_per_batch=Settings.ESM_MAX_RESIDUES_PER_BATCH,
            max_items_per_batch=Settings.ESM_MAX_ITEMS_PER_BATCH,
        )

        logger.info(
            "Fase 2 (ESM-2): %d ventanas agrupadas en %d mini-lotes dinamicos.",
            len(jobs),
            len(batches),
        )

        windows_by_item: Dict[str, List[Tuple[int, int, List[float]]]] = {}

        for batch_idx, batch in enumerate(batches):
            sequences = [job.sequence for job in batch]
            try:
                raw_probs_per_window = self._predict_window_batch(sequences)
            except Exception as exc:
                logger.error(
                    "Error procesando lote de ventanas ESM-2 (ventana inicial de '%s'): %s",
                    batch[0].item_id,
                    exc,
                )
                raise

            for job, raw_probs in zip(batch, raw_probs_per_window):
                windows_by_item.setdefault(job.item_id, []).append((job.start, job.end, raw_probs))

            rss_mb = log_memory_checkpoint(logger, f"esm2_window_batch_{batch_idx + 1}/{len(batches)}")
            warn_if_over_budget(
                logger, f"esm2_window_batch_{batch_idx + 1}", rss_mb, Settings.ESM_MEMORY_BUDGET_MB
            )

        id_to_result: Dict[str, EpitopeResult] = {}
        for item in items:
            sequence = item.record.sequence
            stitched_raw = stitch_window_probabilities(
                windows_by_item[item.record.id], len(sequence), overlap
            )
            smoothed = apply_spatial_smoothing(stitched_raw)
            residues = self._build_residues(sequence, smoothed)
            regions = extract_contiguous_regions(residues, Settings.EPITOPE_MIN_REGION_LENGTH)
            id_to_result[item.record.id] = EpitopeResult(
                antigenicity=item, residues=residues, epitope_regions=regions
            )

        gc.collect()
        return [id_to_result[item.record.id] for item in items]

    def _predict_window_batch(self, sequences: List[str]) -> List[List[float]]:
        """Ejecuta el forward pass ESM-2 + clasificador para un lote homogeneo de ventanas.

        A diferencia de una implementacion sin sliding window, esta funcion
        NO aplica suavizado: cada elemento de ``sequences`` es una VENTANA
        (posiblemente un fragmento de una secuencia mayor), y el filtro
        Savitzky-Golay debe aplicarse una unica vez sobre la secuencia
        completa ya fusionada (:func:`stitch_window_probabilities`), para no
        generar artefactos de borde en cada frontera de ventana.

        Args:
            sequences: Ventanas del lote (longitudes ya agrupadas por
                ``dynamic_batches`` para minimizar padding y acotar memoria).

        Returns:
            Lista de listas de probabilidades CRUDAS (sin suavizar), una por
            ventana, recortadas a su longitud (sin tokens especiales).
        """
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (B, L+2, hidden_dim)
            residue_probs = self.residue_classifier(hidden_states)  # (B, L+2)

        probs_cpu = residue_probs.cpu().numpy()

        batch_probs: List[List[float]] = []
        for idx, sequence in enumerate(sequences):
            seq_len = len(sequence)
            # El tokenizer de ESM-2 antepone <cls> y agrega <eos>: los residuos
            # reales ocupan las posiciones [1, seq_len].
            raw_probs = probs_cpu[idx, 1 : seq_len + 1]
            batch_probs.append(raw_probs.tolist())

        del inputs, outputs, hidden_states, residue_probs, probs_cpu
        return batch_probs

    @staticmethod
    def _build_residues(sequence: str, probs: List[float]) -> List[EpitopeResidue]:
        """Construye la lista de :class:`EpitopeResidue` a partir de probabilidades."""
        return [
            EpitopeResidue(
                position=idx + 1,
                residue=aa,
                epitope_probability=float(prob),
                is_epitope=float(prob) >= Settings.EPITOPE_THRESHOLD,
            )
            for idx, (aa, prob) in enumerate(zip(sequence, probs))
        ]

    def close(self) -> None:
        """Libera el modelo ESM-2 y el clasificador de la memoria del proceso."""
        del self.model
        del self.residue_classifier
        del self.tokenizer
        gc.collect()


class CLIWrapperEngine(BaseEpitopePredictor):
    """Motor desacoplado: invoca un binario externo por subprocess.

    Contrato de interoperabilidad esperado del binario externo (p. ej.
    ``bepipred-cli`` o un wrapper de BepiPred-4.0):

    * Invocacion: ``<binario> --input <fasta_entrada> --output <csv_salida>
      --threshold <umbral>``.
    * Codigo de salida ``0`` en exito; cualquier otro valor se trata como
      fallo irrecuperable del lote.
    * El CSV de salida debe contener, con cabecera, las columnas
      ``sequence_id,position,residue,epitope_probability`` (una fila por
      residuo, ``position`` 1-indexada).

    Este contrato desacopla el pipeline de la implementacion concreta del
    binario, permitiendo blindarse ante actualizaciones del motor externo
    (p. ej. una migracion futura a BepiPred-4.0) sin tocar el orquestador.
    """

    OUTPUT_COLUMNS: Tuple[str, str, str, str] = (
        "sequence_id",
        "position",
        "residue",
        "epitope_probability",
    )

    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        temp_dir: Optional[Path] = None,
    ):
        """Configura la ruta del binario externo y los parametros de ejecucion.

        Args:
            cli_path: Ruta o nombre del ejecutable externo. Si es ``None``, se
                usa ``Settings.BEPIPRED_CLI_PATH``.
            timeout_seconds: Tiempo maximo de espera por lote. Si es ``None``,
                se usa ``Settings.CLI_TIMEOUT_SECONDS``.
            temp_dir: Directorio base para archivos temporales. Si es ``None``,
                se usa ``Settings.CLI_TEMP_DIR`` (o el temporal del sistema si
                tampoco esta configurado).
        """
        self.cli_path = cli_path or Settings.BEPIPRED_CLI_PATH
        self.timeout_seconds = timeout_seconds or Settings.CLI_TIMEOUT_SECONDS
        self.temp_dir = temp_dir or Settings.CLI_TEMP_DIR

        if shutil.which(self.cli_path) is None:
            logger.warning(
                "El binario externo '%s' no se encontro en PATH durante la "
                "inicializacion. Se validara de nuevo al momento de ejecutar "
                "cada lote (util si se instala/monta despues del arranque).",
                self.cli_path,
            )

    def predict(self, items: Sequence[AntigenicityResult]) -> List[EpitopeResult]:
        """Delega la prediccion de epitopos al binario externo via subprocess.

        Args:
            items: Resultados de Fase 1 que superaron el umbral de antigenicidad.

        Returns:
            Lista de :class:`~src.models.EpitopeResult`, en el mismo orden que
            ``items``.

        Raises:
            CLIWrapperError: Si el binario no existe, agota el timeout, sale
                con codigo distinto de cero, o produce un CSV no parseable.
        """
        if not items:
            return []

        if shutil.which(self.cli_path) is None:
            raise CLIWrapperError(
                f"El binario externo '{self.cli_path}' no esta disponible en PATH."
            )

        work_dir = Path(tempfile.mkdtemp(prefix="hts_epitope_cli_", dir=self.temp_dir))
        input_fasta = work_dir / "input.fasta"
        output_csv = work_dir / "output.csv"

        try:
            self._write_fasta(input_fasta, items)

            command = [
                self.cli_path,
                "--input",
                str(input_fasta),
                "--output",
                str(output_csv),
                "--threshold",
                str(Settings.EPITOPE_THRESHOLD),
            ]
            logger.info("Invocando motor externo: %s", " ".join(command))

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise CLIWrapperError(
                    f"El binario '{self.cli_path}' excedio el timeout de "
                    f"{self.timeout_seconds}s: {exc}"
                ) from exc
            except OSError as exc:
                raise CLIWrapperError(
                    f"Fallo al ejecutar el binario '{self.cli_path}': {exc}"
                ) from exc

            if completed.returncode != 0:
                raise CLIWrapperError(
                    f"El binario '{self.cli_path}' finalizo con codigo "
                    f"{completed.returncode}. stderr: {completed.stderr.strip()}"
                )

            if not output_csv.exists():
                raise CLIWrapperError(
                    f"El binario '{self.cli_path}' no genero el archivo de salida "
                    f"esperado '{output_csv}'."
                )

            probs_by_id = self._parse_output_csv(output_csv)
            return self._assemble_results(items, probs_by_id)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _write_fasta(path: Path, items: Sequence[AntigenicityResult]) -> None:
        """Serializa el lote de secuencias aceptadas a un FASTA temporal seguro."""
        with open(path, mode="w", encoding="utf-8") as handle:
            for item in items:
                handle.write(f">{item.record.id}\n{item.record.sequence}\n")

    def _parse_output_csv(self, path: Path) -> Dict[str, List[Tuple[int, str, float]]]:
        """Parsea el CSV de salida del binario externo hacia un mapa por secuencia.

        Args:
            path: Ruta al CSV generado por el binario externo.

        Returns:
            Diccionario ``sequence_id -> [(position, residue, probability), ...]``.

        Raises:
            CLIWrapperError: Si faltan columnas requeridas o los valores no son
                parseables.
        """
        probs_by_id: Dict[str, List[Tuple[int, str, float]]] = {}
        try:
            with open(path, mode="r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not set(self.OUTPUT_COLUMNS).issubset(
                    set(reader.fieldnames)
                ):
                    raise CLIWrapperError(
                        f"El CSV de salida '{path}' no contiene las columnas requeridas "
                        f"{self.OUTPUT_COLUMNS}. Columnas encontradas: {reader.fieldnames}."
                    )
                for row in reader:
                    seq_id = row["sequence_id"]
                    probs_by_id.setdefault(seq_id, []).append(
                        (int(row["position"]), row["residue"], float(row["epitope_probability"]))
                    )
        except (ValueError, KeyError) as exc:
            raise CLIWrapperError(f"El CSV de salida '{path}' es no parseable: {exc}") from exc

        return probs_by_id

    @staticmethod
    def _assemble_results(
        items: Sequence[AntigenicityResult],
        probs_by_id: Dict[str, List[Tuple[int, str, float]]],
    ) -> List[EpitopeResult]:
        """Combina las probabilidades parseadas con los metadatos de Fase 1."""
        results: List[EpitopeResult] = []
        for item in items:
            rows = probs_by_id.get(item.record.id)
            if rows is None:
                raise CLIWrapperError(
                    f"El binario externo no devolvio predicciones para la secuencia "
                    f"'{item.record.id}'."
                )
            rows.sort(key=lambda row: row[0])
            smoothed_probs = apply_spatial_smoothing([probability for _, _, probability in rows])
            residues = [
                EpitopeResidue(
                    position=position,
                    residue=residue,
                    epitope_probability=smoothed_probability,
                    is_epitope=smoothed_probability >= Settings.EPITOPE_THRESHOLD,
                )
                for (position, residue, _), smoothed_probability in zip(rows, smoothed_probs)
            ]
            regions = extract_contiguous_regions(residues, Settings.EPITOPE_MIN_REGION_LENGTH)
            results.append(EpitopeResult(antigenicity=item, residues=residues, epitope_regions=regions))

        return results


class EpitopePredictorFactory:
    """Factoria que instancia el motor de Fase 2 configurado (patron Strategy)."""

    _ESM2_ALIASES = {"esm2", "native", "local"}
    _CLI_ALIASES = {"cli", "hpc", "subprocess"}

    @classmethod
    def create(cls, engine_name: Optional[str] = None) -> BaseEpitopePredictor:
        """Instancia el motor de prediccion de epitopos solicitado.

        Args:
            engine_name: Nombre del motor (``"esm2"`` o ``"cli"``). Si es
                ``None``, se usa ``Settings.PREDICTOR_ENGINE``.

        Returns:
            Instancia concreta de :class:`BaseEpitopePredictor`.

        Raises:
            ValueError: Si ``engine_name`` no corresponde a ningun motor
                soportado.
        """
        name = (engine_name or Settings.PREDICTOR_ENGINE).strip().lower()

        if name in cls._ESM2_ALIASES:
            return NativeESM2Engine()
        if name in cls._CLI_ALIASES:
            return CLIWrapperEngine()

        raise ValueError(
            f"Motor de prediccion de epitopos desconocido: '{name}'. "
            f"Opciones validas: {sorted(cls._ESM2_ALIASES | cls._CLI_ALIASES)}."
        )
