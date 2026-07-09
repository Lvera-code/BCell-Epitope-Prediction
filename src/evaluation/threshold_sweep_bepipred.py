"""Barrido de umbral de Fase 2 (ResidueClassifier) usando BepiPred-3.0 como ground truth.

Para cada proteina nativa completa evaluada (Spike, GAPDH, Lisozima C), calcula
ROC-AUC y PR-AUC entre los scores continuos del ResidueClassifier y la mascara
binaria oficial de BepiPred-3.0, y reporta el umbral optimo (indice de Youden y
F1-optimo) tanto por proteina como agrupado.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.compare_bepipred import load_pipeline_predictions, parse_bepipred_case_encoded

TARGETS: List[Tuple[str, str, str]] = [
    ("Spike", "data/bepipred3/Bcell_linepitope_top_20pct_preds.fasta", "data/processed/fase3/residuos_detalle.csv"),
    ("GAPDH", "data/bepipred3/Bcell_linepitope_top_20pct_preds.fasta", "data/processed/fase3/residuos_detalle.csv"),
    ("Lisozima", "data/bepipred3/lisozima_bepipred.fasta", "data/processed/fase3_lisozima/residuos_detalle.csv"),
]

SEQUENCE_IDS = {
    "Spike": "CONTROL_POSITIVO_SPIKE_P0DTC2",
    "GAPDH": "CONTROL_NEGATIVO_GAPDH_P04406",
    "Lisozima": "CONTROL_POSITIVO_LISOZIMA_P00698",
}


def build_arrays(sequence_id: str, bepipred_fasta: Path, pipeline_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    bepipred_sequences = parse_bepipred_case_encoded(bepipred_fasta)
    raw_seq = bepipred_sequences[sequence_id]
    predictions = load_pipeline_predictions(pipeline_csv, sequence_id)

    y_true, y_score = [], []
    for position, residue in enumerate(raw_seq, start=1):
        if position not in predictions:
            continue
        our_residue, our_prob, _ = predictions[position]
        if our_residue.upper() != residue.upper():
            continue
        y_true.append(1 if residue.isupper() else 0)
        y_score.append(our_prob)
    return np.array(y_true), np.array(y_score)


def youden_optimal(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return thresholds[idx], j[idx], tpr[idx], fpr[idx]


def f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    if len(thresholds) == 0:
        return float("nan"), 0.0, 0.0, 0.0
    return thresholds[idx], f1[idx], precision[idx], recall[idx]


def report(label: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    if len(set(y_true.tolist())) < 2:
        print(f"{label}: solo una clase presente, AUC no definido.")
        return
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    baseline_pr = y_true.mean()
    thr_j, j, tpr_j, fpr_j = youden_optimal(y_true, y_score)
    thr_f1, f1, prec_f1, rec_f1 = f1_optimal(y_true, y_score)

    print(f"\n=== {label} (n={len(y_true)}, positivos BepiPred={int(y_true.sum())}) ===")
    print(f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  (baseline PR por prevalencia={baseline_pr:.4f})")
    print(f"Umbral optimo Youden={thr_j:.4f}  J={j:.4f}  TPR={tpr_j:.4f}  FPR={fpr_j:.4f}")
    print(f"Umbral optimo F1={thr_f1:.4f}  F1={f1:.4f}  Precision={prec_f1:.4f}  Recall={rec_f1:.4f}")


def main() -> int:
    per_protein_arrays = {}
    for label, bepipred_fasta, pipeline_csv in TARGETS:
        sequence_id = SEQUENCE_IDS[label]
        y_true, y_score = build_arrays(sequence_id, Path(bepipred_fasta), Path(pipeline_csv))
        per_protein_arrays[label] = (y_true, y_score)
        report(label, y_true, y_score)

    pooled_true = np.concatenate([v[0] for v in per_protein_arrays.values()])
    pooled_score = np.concatenate([v[1] for v in per_protein_arrays.values()])
    report("POOLED (Spike + GAPDH + Lisozima)", pooled_true, pooled_score)

    return 0


if __name__ == "__main__":
    sys.exit(main())
