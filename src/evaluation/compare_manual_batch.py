"""Evalua nuestro modelo (Fase 1) sobre el batch manual de 30 fragmentos y,
si ya existe el CSV con los resultados manuales de BepiPred-3.0, calcula su
AUC en el mismo batch para comparacion directa.

El ground truth binario se deriva del prefijo del header (POS_/NEG_) generado
por ``generate_validation_batch.py`` -- son las mismas etiquetas reales de
IEDB/housekeeping del test set, no una simulacion.
"""

import csv
import sys
from pathlib import Path
from typing import Dict

from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.engines.antigenicity_cnn import AntigenicityCNNEngine
from src.utils.fasta_parser import FastaParser

BATCH_FASTA = Path("data/evaluation/bepipred_manual_batch.fasta")
BEPIPRED_RESULTS_CSV = Path("data/evaluation/bepipred_manual_results.csv")


def true_label_from_id(sequence_id: str) -> int:
    return 1 if sequence_id.startswith("POS_") else 0


def load_bepipred_manual_results(csv_path: Path) -> Dict[str, int]:
    """Lee un CSV de dos columnas: id,bepipred_binary (0 o 1)."""
    results: Dict[str, int] = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            results[row["id"]] = int(row["bepipred_binary"])
    return results


def main() -> int:
    records = FastaParser.parse(BATCH_FASTA, min_length=1)
    if len(records) != 30:
        print(f"AVISO: se esperaban 30 secuencias, se encontraron {len(records)}.")

    y_true = [true_label_from_id(r.id) for r in records]

    engine = AntigenicityCNNEngine()
    results = engine.run(records)
    y_score_ours = [r.score for r in results]

    roc_auc_ours = roc_auc_score(y_true, y_score_ours)
    pr_auc_ours = average_precision_score(y_true, y_score_ours)
    print(f"\n=== Nuestro pipeline (Fase 1, N={len(records)}) ===")
    print(f"ROC-AUC={roc_auc_ours:.4f}  PR-AUC={pr_auc_ours:.4f}")

    if not BEPIPRED_RESULTS_CSV.exists():
        print(
            f"\nEsperando resultados manuales de BepiPred. Crea '{BEPIPRED_RESULTS_CSV}' "
            "con columnas 'id,bepipred_binary' (id=POS_1..POS_15/NEG_1..NEG_15, "
            "bepipred_binary=0 o 1 segun si BepiPred-3.0 marco esa secuencia como "
            "epitopo) y volve a correr este script."
        )
        return 0

    bepipred_results = load_bepipred_manual_results(BEPIPRED_RESULTS_CSV)
    missing = [r.id for r in records if r.id not in bepipred_results]
    if missing:
        print(f"\nAVISO: faltan resultados de BepiPred para {len(missing)} ids: {missing}")
        return 1

    y_score_bepipred = [bepipred_results[r.id] for r in records]
    roc_auc_bepipred = roc_auc_score(y_true, y_score_bepipred)
    pr_auc_bepipred = average_precision_score(y_true, y_score_bepipred)

    print(f"\n=== BepiPred-3.0 (manual, N={len(records)}) ===")
    print(f"ROC-AUC={roc_auc_bepipred:.4f}  PR-AUC={pr_auc_bepipred:.4f}")
    print(
        "\nNOTA: el AUC de BepiPred aqui viene de llamadas binarias (0/1), no de un score "
        "continuo -- equivale a (Sensibilidad+Especificidad)/2, no a una curva ROC real. "
        "No es estrictamente comparable punto a punto contra el AUC continuo de nuestro "
        "pipeline; comparalo como 'tasa de acierto global' en el mismo batch."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
