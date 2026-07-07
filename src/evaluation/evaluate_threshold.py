"""Determinacion empirica del umbral de corte de Fase 1 via curva Precision-Recall.

Sustituye el umbral arbitrario (``Settings.ANTIGENICITY_THRESHOLD``) por uno
matematicamente justificado: carga un dataset de validacion ciego (por
defecto, el split ``test`` reservado por ``src/training/dataset_prep.py``, que
la 1D-CNN nunca vio ni en backprop ni en la calibracion de Platt), calcula los
scores CALIBRADOS de :class:`~src.engines.antigenicity_cnn.AntigenicityCNNEngine`
y determina el punto de la curva Precision-Recall que maximiza el F1-Score.

Exporta la curva a un archivo de imagen para auditoria visual.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config.settings import Settings
from src.engines.antigenicity_cnn import AntigenicityCNNEngine
from src.utils.fasta_parser import FastaParser
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def compute_calibrated_scores(
    engine: AntigenicityCNNEngine,
    positive_fasta: Path,
    negative_fasta: Path,
    min_length: int,
) -> Tuple[List[int], List[float]]:
    """Ejecuta el motor de Fase 1 sobre un par de FASTA etiquetados.

    Args:
        engine: Motor de antigenicidad ya instanciado (con o sin calibracion
            de Platt cargada).
        positive_fasta: FASTA de secuencias positivas conocidas.
        negative_fasta: FASTA de secuencias negativas conocidas.
        min_length: Longitud minima de saneamiento aplicada a ambos FASTA.

    Returns:
        Tupla ``(y_true, y_scores)`` en el mismo orden (positivos primero).

    Raises:
        ValueError: Si alguno de los dos FASTA, tras el saneamiento, queda vacio.
    """
    positive_records = FastaParser.parse(positive_fasta, min_length=min_length)
    negative_records = FastaParser.parse(negative_fasta, min_length=min_length)

    if not positive_records:
        raise ValueError(f"El FASTA positivo '{positive_fasta}' no aporto secuencias validas.")
    if not negative_records:
        raise ValueError(f"El FASTA negativo '{negative_fasta}' no aporto secuencias validas.")

    positive_results = engine.run(positive_records)
    negative_results = engine.run(negative_records)

    y_true = [1] * len(positive_results) + [0] * len(negative_results)
    y_scores = [r.score for r in positive_results] + [r.score for r in negative_results]
    return y_true, y_scores


def find_best_f1_threshold(
    y_true: List[int], y_scores: List[float]
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Calcula la curva Precision-Recall y el umbral que maximiza el F1-Score.

    Args:
        y_true: Etiquetas binarias verdaderas.
        y_scores: Scores continuos (calibrados) del motor evaluado.

    Returns:
        Tupla ``(mejor_umbral, mejor_f1, precision_en_el_optimo,
        recall_en_el_optimo, precision_curva, recall_curva)``.

    Raises:
        ValueError: Si la curva Precision-Recall no produce ningun umbral
            evaluable (dataset degenerado de una unica clase).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # ``precision_recall_curve`` devuelve un punto extra (recall=0, umbral
    # implicito = +inf) sin threshold asociado: se excluye para alinear los
    # arrays con ``thresholds`` antes de calcular el F1 por punto.
    precision_aligned = precision[:-1]
    recall_aligned = recall[:-1]

    if len(thresholds) == 0:
        raise ValueError(
            "La curva Precision-Recall no produjo umbrales evaluables (dataset "
            "degenerado: revise que ambas clases esten representadas)."
        )

    denom = precision_aligned + recall_aligned
    f1_scores = np.divide(
        2 * precision_aligned * recall_aligned,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    best_idx = int(np.argmax(f1_scores))
    return (
        float(thresholds[best_idx]),
        float(f1_scores[best_idx]),
        float(precision_aligned[best_idx]),
        float(recall_aligned[best_idx]),
        precision,
        recall,
    )


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    best_precision: float,
    best_recall: float,
    best_f1: float,
    best_threshold: float,
    output_path: Path,
) -> None:
    """Genera y guarda el plot de la curva Precision-Recall con el optimo marcado.

    Args:
        precision: Array de precision de la curva completa.
        recall: Array de recall de la curva completa.
        best_precision: Precision en el umbral optimo.
        best_recall: Recall en el umbral optimo.
        best_f1: F1-Score en el umbral optimo.
        best_threshold: Umbral que maximiza el F1-Score.
        output_path: Ruta de archivo destino de la imagen (se crean los
            directorios padre si hace falta).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#1f77b4", linewidth=2, label="Curva Precision-Recall")
    ax.scatter(
        [best_recall],
        [best_precision],
        color="#d62728",
        zorder=5,
        s=60,
        label=f"Optimo F1: umbral={best_threshold:.4f}, F1={best_f1:.4f}",
    )
    ax.set_xlabel("Recall (Sensibilidad)")
    ax.set_ylabel("Precision")
    ax.set_title("Fase 1: Curva Precision-Recall (scores calibrados via Platt Scaling)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_threshold_report(
    n_positive: int,
    n_negative: int,
    is_calibrated: bool,
    best_threshold: float,
    best_f1: float,
    best_precision: float,
    best_recall: float,
    output_path: Path,
) -> None:
    """Imprime el reporte del umbral empirico en una tabla ASCII legible."""
    lines = [
        ("Secuencias positivas evaluadas", str(n_positive)),
        ("Secuencias negativas evaluadas", str(n_negative)),
        ("Scores calibrados (Platt Scaling)", "SI" if is_calibrated else "NO (sigmoide sin calibrar)"),
        ("Umbral optimo (F1 maximo)", f"{best_threshold:.4f}"),
        ("F1-Score en el umbral optimo", f"{best_f1:.4f}"),
        ("Precision en el umbral optimo", f"{best_precision:.4f}"),
        ("Recall en el umbral optimo", f"{best_recall:.4f}"),
        ("Curva Precision-Recall exportada en", str(output_path)),
    ]
    width = max(len(label) for label, _ in lines) + 2
    value_width = max(len(value) for _, value in lines) + 2
    total_width = width + value_width

    print("\n┌" + "─" * total_width + "┐")
    print(f"│{'UMBRAL EMPIRICO OPTIMO (MAXIMIZACION DE F1-SCORE)':^{total_width}}│")
    print("├" + "─" * total_width + "┤")
    for label, value in lines:
        print(f"│ {label:<{width}}{value:>{value_width - 1}} │")
    print("└" + "─" * total_width + "┘\n")


def _parse_arguments() -> argparse.Namespace:
    """Define la interfaz de linea de comandos de la evaluacion standalone."""
    parser = argparse.ArgumentParser(
        description="Determinacion empirica del umbral de corte de Fase 1 via curva Precision-Recall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--positive",
        type=Path,
        default=Settings.TRAINING_DATA_DIR / "test_positive.fasta",
        help="FASTA de secuencias positivas del dataset de validacion ciego.",
    )
    parser.add_argument(
        "-n",
        "--negative",
        type=Path,
        default=Settings.TRAINING_DATA_DIR / "test_negative.fasta",
        help="FASTA de secuencias negativas del dataset de validacion ciego.",
    )
    parser.add_argument(
        "-o",
        "--output-plot",
        type=Path,
        default=Settings.PROCESSED_DIR / "precision_recall_curve.png",
        help="Ruta de salida para el plot de la curva Precision-Recall.",
    )
    parser.add_argument("--min-length", type=int, default=Settings.MIN_SEQUENCE_LENGTH)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Punto de entrada standalone: ``python -m src.evaluation.evaluate_threshold``."""
    args = _parse_arguments()

    if args.offline:
        Settings.apply_offline_mode()
    Settings.apply_thread_limits()

    try:
        engine = AntigenicityCNNEngine(threshold=Settings.ANTIGENICITY_THRESHOLD)
        y_true, y_scores = compute_calibrated_scores(
            engine, args.positive, args.negative, args.min_length
        )

        if not engine.is_calibrated:
            logger.warning(
                "El motor de Fase 1 no tiene una calibracion de Platt cargada: el umbral "
                "resultante se basa en un sigmoide SIN CALIBRAR. Ejecute "
                "'python -m src.training.trainer' antes de confiar en este resultado."
            )

        best_threshold, best_f1, best_precision, best_recall, precision, recall = (
            find_best_f1_threshold(y_true, y_scores)
        )

        plot_precision_recall_curve(
            precision, recall, best_precision, best_recall, best_f1, best_threshold, args.output_plot
        )

        print_threshold_report(
            n_positive=y_true.count(1),
            n_negative=y_true.count(0),
            is_calibrated=engine.is_calibrated,
            best_threshold=best_threshold,
            best_f1=best_f1,
            best_precision=best_precision,
            best_recall=best_recall,
            output_path=args.output_plot,
        )

        logger.info(
            "Umbral empirico optimo=%.4f (F1=%.4f, Precision=%.4f, Recall=%.4f). Plot guardado en '%s'.",
            best_threshold,
            best_f1,
            best_precision,
            best_recall,
            args.output_plot,
        )
        return 0
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Fallo fatal durante la evaluacion del umbral: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
