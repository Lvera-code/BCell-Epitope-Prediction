"""Convierte el raw_output.csv real de BepiPred-3.0 (score por residuo) en
una llamada binaria por secuencia para el batch manual de validacion.

Regla: si ALGUN residuo de la secuencia alcanza el umbral por defecto de
BepiPred-3.0 (0.1512), la secuencia completa se marca como positiva (1).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

RAW_CSV = Path("/mnt/c/Users/USUARIO/Downloads/bepipred3_results/raw_output.csv")
OUTPUT_CSV = Path("data/evaluation/bepipred_manual_results.csv")
THRESHOLD = 0.1512


def base_id(accession: str) -> str:
    return accession.split()[0]


def main() -> int:
    scores_by_id = defaultdict(list)
    with open(RAW_CSV, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = base_id(row["Accession"])
            scores_by_id[seq_id].append(float(row["BepiPred-3.0 score"]))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "bepipred_binary"])
        for seq_id in sorted(scores_by_id, key=lambda s: (s.split("_")[0], int(s.split("_")[1]))):
            scores = scores_by_id[seq_id]
            binary = 1 if max(scores) >= THRESHOLD else 0
            writer.writerow([seq_id, binary])
            print(f"{seq_id}: max_score={max(scores):.4f} n_residuos={len(scores)} -> {binary}")

    print(f"\nExportado a '{OUTPUT_CSV}' ({len(scores_by_id)} secuencias).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
