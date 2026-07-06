"""Suite de auditoria cientifica: valida el motor de antigenicidad contra IEDB.

Ingiere un FASTA positivo (epitopos/antigenos confirmados, tipicamente
exportados de IEDB) y un FASTA negativo (proteinas housekeeping/intracelulares
que no deberian presentar antigenicidad de superficie) y genera un reporte
estadistico riguroso: matriz de confusion, sensibilidad, especificidad, tasa
de falsos positivos (FPR) y ROC-AUC, calculados con ``scikit-learn`` sobre los
scores continuos del motor bajo auditoria.
"""

import argparse
import gc
import random
import sys
import time
from pathlib import Path
from typing import List, Protocol, Sequence

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.config.settings import Settings
from src.engines.antigenicity_cnn import AntigenicityCNNEngine
from src.engines.epitope_engine import NativeESM2Engine, compute_sliding_windows
from src.models import AntigenicityResult, BenchmarkReport, SequenceRecord
from src.utils.fasta_parser import CANONICAL_AA, FastaParser
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)


class AntigenicityScorer(Protocol):
    """Contrato minimo que debe cumplir cualquier motor auditable por esta suite."""

    def run(self, items: Sequence[SequenceRecord]) -> List[AntigenicityResult]:
        ...


class BenchmarkSuite:
    """Ejecuta la auditoria estadistica de un motor de antigenicidad frente a IEDB."""

    def __init__(self, scorer: AntigenicityScorer, threshold: "float | None" = None):
        """Inicializa la suite con el motor a auditar.

        Args:
            scorer: Motor que implementa ``run(items) -> List[AntigenicityResult]``.
                Tipicamente una instancia de
                :class:`~src.engines.antigenicity_cnn.AntigenicityCNNEngine`.
            threshold: Umbral de decision a evaluar. Si es ``None``, se usa el
                umbral propio del motor si esta expuesto como atributo
                ``threshold``, o en su defecto ``Settings.ANTIGENICITY_THRESHOLD``.
        """
        self.scorer = scorer
        self.threshold = threshold if threshold is not None else getattr(
            scorer, "threshold", Settings.ANTIGENICITY_THRESHOLD
        )

    def run(
        self,
        positive_fasta: Path,
        negative_fasta: Path,
        min_length: int = Settings.MIN_SEQUENCE_LENGTH,
    ) -> BenchmarkReport:
        """Ejecuta la auditoria completa sobre un par de FASTA etiquetados.

        Args:
            positive_fasta: FASTA de secuencias positivas conocidas (IEDB).
            negative_fasta: FASTA de secuencias negativas conocidas
                (housekeeping/intracelulares).
            min_length: Longitud minima de saneamiento aplicada a ambos FASTA.

        Returns:
            :class:`~src.models.BenchmarkReport` con todas las metricas.

        Raises:
            ValueError: Si alguno de los dos FASTA, tras el saneamiento, queda
                vacio (no es posible calcular una matriz de confusion valida).
        """
        logger.info("Cargando corpus positivo (IEDB) desde '%s'.", positive_fasta)
        positive_records = FastaParser.parse(positive_fasta, min_length=min_length)
        logger.info("Cargando corpus negativo (housekeeping) desde '%s'.", negative_fasta)
        negative_records = FastaParser.parse(negative_fasta, min_length=min_length)

        if not positive_records:
            raise ValueError(f"El FASTA positivo '{positive_fasta}' no aporto secuencias validas.")
        if not negative_records:
            raise ValueError(f"El FASTA negativo '{negative_fasta}' no aporto secuencias validas.")

        logger.info(
            "Auditoria: %d secuencias positivas, %d secuencias negativas, umbral=%.4f.",
            len(positive_records),
            len(negative_records),
            self.threshold,
        )

        positive_scores = self.scorer.run(positive_records)
        negative_scores = self.scorer.run(negative_records)

        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_scores = [r.score for r in positive_scores] + [r.score for r in negative_scores]
        y_pred = [1 if score >= self.threshold else 0 for score in y_scores]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
        false_positive_rate = float(fp) / (fp + tn) if (fp + tn) > 0 else 0.0

        if len(set(y_true)) < 2:
            logger.warning("Solo una clase presente; ROC-AUC no esta definido. Se reporta NaN.")
            roc_auc = float("nan")
            fpr_curve: List[float] = []
            tpr_curve: List[float] = []
        else:
            roc_auc = float(roc_auc_score(y_true, y_scores))
            fpr_curve_arr, tpr_curve_arr, _ = roc_curve(y_true, y_scores)
            fpr_curve = fpr_curve_arr.tolist()
            tpr_curve = tpr_curve_arr.tolist()

        report = BenchmarkReport(
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            sensitivity=sensitivity,
            specificity=specificity,
            false_positive_rate=false_positive_rate,
            roc_auc=roc_auc,
            threshold_used=self.threshold,
            n_positive=len(positive_records),
            n_negative=len(negative_records),
            fpr_curve=fpr_curve,
            tpr_curve=tpr_curve,
        )

        logger.info(
            "Auditoria completada: Sens=%.4f Espec=%.4f FPR=%.4f ROC-AUC=%.4f",
            report.sensitivity,
            report.specificity,
            report.false_positive_rate,
            report.roc_auc,
        )
        return report


def print_benchmark_report(report: BenchmarkReport) -> None:
    """Imprime el reporte de auditoria en una tabla ASCII legible en consola.

    Args:
        report: Reporte generado por :meth:`BenchmarkSuite.run`.
    """
    lines = [
        ("Umbral de decision evaluado", f"{report.threshold_used:.4f}"),
        ("Secuencias positivas (IEDB)", str(report.n_positive)),
        ("Secuencias negativas (housekeeping)", str(report.n_negative)),
        ("Verdaderos Positivos (TP)", str(report.true_positives)),
        ("Verdaderos Negativos (TN)", str(report.true_negatives)),
        ("Falsos Positivos (FP)", str(report.false_positives)),
        ("Falsos Negativos (FN)", str(report.false_negatives)),
        ("Sensibilidad (Recall)", f"{report.sensitivity:.4f}"),
        ("Especificidad", f"{report.specificity:.4f}"),
        ("Tasa de Falsos Positivos (FPR)", f"{report.false_positive_rate:.4f}"),
        ("ROC-AUC", f"{report.roc_auc:.4f}"),
    ]
    width = max(len(label) for label, _ in lines) + 2

    print("\n┌" + "─" * (width + 20) + "┐")
    print(f"│{'REPORTE DE AUDITORIA CIENTIFICA':^{width + 20}}│")
    print("├" + "─" * (width + 20) + "┤")
    for label, value in lines:
        print(f"│ {label:<{width}}{value:>18} │")
    print("└" + "─" * (width + 20) + "┘\n")


def run_macromolecular_stress_test(length: int = 5000, seed: int = 2024) -> bool:
    """Prueba de estabilidad masiva: verifica el procesamiento de macromoleculas extremas.

    Genera una secuencia sintetica de ``length`` aminoacidos -- muy por encima
    del limite fisico de contexto de ESM-2 (1022 aa) -- y la procesa de
    extremo a extremo (Fase 1 + Fase 2 con Sliding Window Stitcher), bajo el
    mismo protocolo de memoria del resto del pipeline (``torch.no_grad()``,
    mini-lotes dinamicos, liberacion explicita). Verifica que:

    1. El numero de residuos devueltos coincide EXACTAMENTE con ``length``
       (cero truncamiento, sin importar cuantas ventanas hicieron falta).
    2. No se produce ninguna excepcion (OOM u otra) durante la inferencia.

    Args:
        length: Longitud de la secuencia sintetica a generar (aa). Los
            valores tipicos de auditoria van de 3000 a 5000 aa (superan
            ampliamente proteinas gigantes reales, p. ej. la glucoproteina
            Spike de SARS-CoV-2 a 1273 aa o ortologos de superficie de
            Plasmodium falciparum de varios miles de aa).
        seed: Semilla para la generacion deterministica de la secuencia
            sintetica (reproducibilidad sin dependencia de red).

    Returns:
        ``True`` si la prueba supero la cobertura completa sin errores.
    """
    rng = random.Random(seed)
    canonical_alphabet = sorted(CANONICAL_AA)
    synthetic_sequence = "".join(rng.choice(canonical_alphabet) for _ in range(length))
    record = SequenceRecord(
        id=f"STRESS_TEST_{length}AA",
        sequence=synthetic_sequence,
        description="Secuencia sintetica de estres macromolecular",
    )

    expected_windows = compute_sliding_windows(
        length, Settings.ESM_SLIDING_WINDOW_SIZE, Settings.ESM_SLIDING_WINDOW_OVERLAP
    )

    logger.info(
        "STRESS TEST MACROMOLECULAR: procesando secuencia sintetica de %d aa "
        "(%d ventanas esperadas, ventana=%d aa, overlap=%d aa)...",
        length,
        len(expected_windows),
        Settings.ESM_SLIDING_WINDOW_SIZE,
        Settings.ESM_SLIDING_WINDOW_OVERLAP,
    )

    start_time = time.time()
    passed = False
    processed_length = 0

    try:
        antigenicity_engine = AntigenicityCNNEngine(threshold=Settings.ANTIGENICITY_THRESHOLD)
        phase1_results = antigenicity_engine.run([record])
        del antigenicity_engine
        gc.collect()

        predictor = NativeESM2Engine()
        try:
            results = predictor.predict(phase1_results)
        finally:
            predictor.close()
            gc.collect()

        if results:
            processed_length = len(results[0].residues)
            passed = processed_length == length
    except Exception as exc:
        logger.error("STRESS TEST FALLIDO: excepcion durante la inferencia: %s", exc)
        passed = False

    elapsed = time.time() - start_time

    print("\n┌" + "─" * 71 + "┐")
    print(f"│{'STRESS TEST MACROMOLECULAR (SLIDING WINDOW STITCHER)':^71}│")
    print("├" + "─" * 71 + "┤")
    print(f"│ Longitud de entrada                  : {length:>8} aa{'':<20} │")
    print(f"│ Residuos procesados                  : {processed_length:>8} aa{'':<20} │")
    print(f"│ Ventanas utilizadas                  : {len(expected_windows):>8}{'':<23} │")
    print(f"│ Tiempo total de inferencia           : {elapsed:>8.2f} s{'':<21} │")
    veredicto = "SI (0% truncamiento)" if passed else "NO — FALLO DETECTADO"
    print(f"│ Cobertura completa sin truncamiento  : {veredicto:<31} │")
    print("└" + "─" * 71 + "┘\n")

    if passed:
        logger.info(
            "STRESS TEST SUPERADO: %d/%d aa procesados (100%%) en %d ventanas, %.2fs, sin OOM.",
            processed_length,
            length,
            len(expected_windows),
            elapsed,
        )
    else:
        logger.error(
            "STRESS TEST FALLIDO: %d/%d aa procesados.", processed_length, length
        )

    return passed


def _parse_arguments() -> argparse.Namespace:
    """Define la interfaz de linea de comandos para la ejecucion standalone."""
    parser = argparse.ArgumentParser(
        description="Suite de Auditoria Cientifica del motor de antigenicidad (SOTA-B-Epitope-Pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--positive", type=Path, default=None, help="FASTA de secuencias positivas (IEDB)."
    )
    parser.add_argument(
        "-n",
        "--negative",
        type=Path,
        default=None,
        help="FASTA de secuencias negativas (housekeeping/intracelulares).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=Settings.ANTIGENICITY_THRESHOLD,
        help="Umbral de decision de antigenicidad a auditar.",
    )
    parser.add_argument(
        "--stress-test-length",
        type=int,
        default=None,
        help="Si se especifica, ejecuta el Stress Test macromolecular con una secuencia "
        "sintetica de esta longitud (aa) en lugar de la auditoria IEDB/housekeeping.",
    )
    return parser.parse_args()


def main() -> int:
    """Punto de entrada standalone: ``python -m src.validation.benchmark_suite``."""
    args = _parse_arguments()

    if args.stress_test_length is not None:
        try:
            passed = run_macromolecular_stress_test(length=args.stress_test_length)
            return 0 if passed else 1
        except Exception as exc:
            logger.critical("Fallo fatal durante el stress test macromolecular: %s", exc)
            return 1

    if args.positive is None or args.negative is None:
        logger.critical(
            "Se requieren -p/--positive y -n/--negative para la auditoria IEDB/housekeeping "
            "(o --stress-test-length para el stress test macromolecular)."
        )
        return 1

    try:
        scorer = AntigenicityCNNEngine(threshold=args.threshold)
        suite = BenchmarkSuite(scorer=scorer, threshold=args.threshold)
        report = suite.run(args.positive, args.negative)
        print_benchmark_report(report)
        return 0
    except Exception as exc:
        logger.critical("Fallo fatal durante la auditoria: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
