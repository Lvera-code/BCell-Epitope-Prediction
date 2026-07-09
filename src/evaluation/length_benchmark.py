"""Benchmark de degradacion de Fase 1 (antigenicidad) por bin de longitud.

Evalua unicamente nuestro pipeline (AntigenicityCNNEngine, scores calibrados
via Platt Scaling) sobre el panel de test real (IEDB positivo / housekeeping
negativo), agrupado por longitud de secuencia. No incluye una columna de
BepiPred: no existen predicciones de BepiPred para este panel de 309
fragmentos (BepiPred solo se corrio manualmente para 3 proteinas nativas
completas en una sesion aparte), y fabricar esos numeros seria dato inventado.
"""

import sys
from pathlib import Path
from typing import List, Tuple

from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.engines.antigenicity_cnn import AntigenicityCNNEngine
from src.utils.fasta_parser import FastaParser

BINS: List[Tuple[int, int]] = [(0, 25), (26, 50), (51, 80), (81, 120), (121, 10**9)]


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"


def main() -> int:
    positive_records = FastaParser.parse(Path("data/training/test_positive.fasta"), min_length=1)
    negative_records = FastaParser.parse(Path("data/training/test_negative.fasta"), min_length=1)

    engine = AntigenicityCNNEngine()
    positive_results = engine.run(positive_records)
    negative_results = engine.run(negative_records)

    samples = [(len(r.record.sequence), r.score, 1) for r in positive_results]
    samples += [(len(r.record.sequence), r.score, 0) for r in negative_results]

    print(f"\n{'Bin (aa)':<12}{'N':>6}{'N_pos':>8}{'N_neg':>8}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for lo, hi in BINS:
        subset = [(score, label) for length, score, label in samples if lo <= length <= hi]
        n_pos = sum(1 for _, label in subset if label == 1)
        n_neg = sum(1 for _, label in subset if label == 0)
        label_str = bin_label(lo, hi)

        if n_pos == 0 or n_neg == 0:
            reason = "vacio" if not subset else "una sola clase"
            print(f"{label_str:<12}{len(subset):>6}{n_pos:>8}{n_neg:>8}{'N/D':>10}{'N/D':>10}   <- {reason}, AUC no definible")
            continue

        y_true = [label for _, label in subset]
        y_score = [score for score, _ in subset]
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        print(f"{label_str:<12}{len(subset):>6}{n_pos:>8}{n_neg:>8}{roc_auc:>10.4f}{pr_auc:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
