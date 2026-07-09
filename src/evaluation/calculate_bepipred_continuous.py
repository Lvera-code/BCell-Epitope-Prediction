"""Aisla si el colapso binario de BepiPred-3.0 en fragmentos aislados (10-25aa)
es falla arquitectonica (sin señal real) o mala calibracion de umbral.

Usa el score continuo maximo por fragmento (max de 'BepiPred-3.0 score' entre
sus residuos) contra el ground truth real (prefijo POS_/NEG_ del batch), sin
aplicar el umbral 0.1512 -- eso es exactamente lo que separa "sin señal" de
"señal mal calibrada".
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

RAW_CSV = Path("/mnt/c/Users/USUARIO/Downloads/bepipred3_results/raw_output.csv")


def base_id(accession: str) -> str:
    return accession.split()[0]


def true_label(seq_id: str) -> int:
    return 1 if seq_id.startswith("POS_") else 0


def main() -> int:
    scores_by_id = defaultdict(list)
    with open(RAW_CSV, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = base_id(row["Accession"])
            scores_by_id[seq_id].append(float(row["BepiPred-3.0 score"]))

    ids = sorted(scores_by_id, key=lambda s: (s.split("_")[0], int(s.split("_")[1])))
    y_true = [true_label(i) for i in ids]
    y_score = [max(scores_by_id[i]) for i in ids]

    print(f"{'id':<8}{'max_score':>12}{'label':>8}")
    for seq_id, score, label in zip(ids, y_score, y_true):
        print(f"{seq_id:<8}{score:>12.4f}{label:>8}")

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print(f"\nAUC Continuo BepiPred-3.0: {roc_auc:.4f}")
    print(f"PR-AUC Continuo BepiPred-3.0: {pr_auc:.4f}")

    if roc_auc < 0.60:
        diagnostico = "AUC < 0.60 -> el modelo carece de señal intrinseca en fragmentos aislados."
    elif roc_auc > 0.70:
        diagnostico = "AUC > 0.70 -> el modelo tiene señal, pero el umbral 0.1512 esta gravemente descalibrado para 1D corto."
    else:
        diagnostico = "0.60 <= AUC <= 0.70 -> zona ambigua: señal debil, no concluyente en ninguna direccion."
    print(f"Diagnostico Forense: {diagnostico}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
