"""Fase 3: cruce posicion a posicion contra el ground truth real de BepiPred-3.0.

Lee el FASTA oficial de salida del servidor BepiPred-3.0
(``Bcell_linepitope_top_Xpct_preds.fasta``), donde el propio servidor codifica
epitopo/no-epitopo en la capitalizacion de cada residuo (documentado en
https://services.healthtech.dtu.dk/services/BepiPred-3.0/2-Instructions.php),
y lo cruza contra ``residuos_detalle.csv`` exportado por
``python -m src.main`` para las mismas secuencias.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.stats import mannwhitneyu, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def parse_bepipred_case_encoded(fasta_path: Path) -> Dict[str, str]:
    """Extrae la secuencia cruda (sin normalizar case) por ID de header."""
    sequences: Dict[str, str] = {}
    current_id = None
    with open(fasta_path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences[current_id] = ""
            elif current_id is not None:
                sequences[current_id] += line
    return sequences


def load_pipeline_predictions(csv_path: Path, sequence_id: str) -> Dict[int, Tuple[str, float, bool]]:
    """Carga (residuo, probabilidad, is_epitope) por posicion para un sequence_id."""
    predictions: Dict[int, Tuple[str, float, bool]] = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sequence_id"] != sequence_id:
                continue
            position = int(row["position"])
            predictions[position] = (
                row["residue"],
                float(row["epitope_probability"]),
                row["is_epitope"] == "True",
            )
    return predictions


def compare_sequence(
    label: str,
    bepipred_seq: str,
    predictions: Dict[int, Tuple[str, float, bool]],
) -> None:
    tp = tn = fp = fn = mismatches = 0
    rows: List[Tuple[int, str, bool, bool, float]] = []

    for position, residue in enumerate(bepipred_seq, start=1):
        bepipred_is_epitope = residue.isupper()
        if position not in predictions:
            continue
        our_residue, our_prob, our_is_epitope = predictions[position]
        if our_residue.upper() != residue.upper():
            mismatches += 1
            continue

        if our_is_epitope and bepipred_is_epitope:
            tp += 1
        elif our_is_epitope and not bepipred_is_epitope:
            fp += 1
        elif not our_is_epitope and bepipred_is_epitope:
            fn += 1
        else:
            tn += 1
        rows.append((position, residue.upper(), bepipred_is_epitope, our_is_epitope, our_prob))

    total = tp + tn + fp + fn
    if total == 0:
        print(f"\n{label}: sin posiciones comparables (revisar alineacion).")
        return

    agreement = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    print(f"\n=== {label} ({total} posiciones comparadas, {mismatches} descartadas por desalineacion de residuo) ===")
    print(f"{'':20}{'BepiPred: epitopo':>20}{'BepiPred: no-epitopo':>22}")
    print(f"{'Pipeline: epitopo':20}{tp:>20}{fp:>22}")
    print(f"{'Pipeline: no-epitopo':20}{fn:>20}{tn:>22}")
    print(
        f"Concordancia={agreement:.4f}  Sensibilidad(vs BepiPred)={sensitivity:.4f}  "
        f"Especificidad(vs BepiPred)={specificity:.4f}  Precision(vs BepiPred)={precision:.4f}"
    )

    report_continuous_vs_binary(rows)
    report_peak_proximity(label, rows)


def report_continuous_vs_binary(rows: List[Tuple[int, str, bool, bool, float]]) -> None:
    """Correlaciona nuestra probabilidad continua contra la mascara binaria de BepiPred.

    IMPORTANTE: BepiPred-3.0 no publico aqui sus scores continuos (solo el FASTA
    binario top-X%), asi que esto es continuo-vs-binario, no continuo-vs-continuo.
    Pearson sobre una variable binaria es matematicamente identico al coeficiente
    punto-biserial; Spearman queda degenerado por los empates masivos en el lado
    binario y se reporta solo como referencia complementaria.
    """
    probs = [r[4] for r in rows]
    bepipred_binary = [1 if r[2] else 0 for r in rows]

    if len(set(bepipred_binary)) < 2:
        print("Correlacion continuo-vs-binario: no calculable (BepiPred no tiene ambas clases aqui).")
        return

    r_pb, p_pb = pearsonr(probs, bepipred_binary)
    rho, p_rho = spearmanr(probs, bepipred_binary)

    probs_pos = [p for p, b in zip(probs, bepipred_binary) if b == 1]
    probs_neg = [p for p, b in zip(probs, bepipred_binary) if b == 0]
    u_stat, p_u = mannwhitneyu(probs_pos, probs_neg, alternative="greater")

    print(
        f"Correlacion punto-biserial (Pearson, continuo-vs-binario)={r_pb:.4f} (p={p_pb:.4g})  "
        f"Spearman={rho:.4f} (p={p_rho:.4g}, degenerado por empates binarios)"
    )
    print(
        f"Mann-Whitney U (H1: nuestro score es mayor en posiciones BepiPred-epitopo): "
        f"U={u_stat:.1f} p={p_u:.4g}  "
        f"media_prob(BepiPred=epitopo)={sum(probs_pos)/len(probs_pos):.4f}  "
        f"media_prob(BepiPred=no-epitopo)={sum(probs_neg)/len(probs_neg):.4f}"
    )


def report_peak_proximity(label: str, rows: List[Tuple[int, str, bool, bool, float]]) -> None:
    """Para cada hit nuestro, distancia (en residuos) al epitopo BepiPred mas cercano."""
    bepipred_positions = [r[0] for r in rows if r[2]]
    our_hits = [r for r in rows if r[3]]

    if not our_hits:
        print(f"{label}: el pipeline no marco ningun residuo como epitopo; no hay picos que analizar.")
        return
    if not bepipred_positions:
        print(f"{label}: BepiPred no marco ningun residuo aqui; no hay referencia para distancia.")
        return

    print(f"Analisis de picos ({len(our_hits)} hits del pipeline vs {len(bepipred_positions)} residuos BepiPred-epitopo):")
    for position, residue, _, _, prob in our_hits:
        nearest = min(abs(position - bp) for bp in bepipred_positions)
        print(f"  pos={position:>5} residuo={residue} prob={prob:.4f}  distancia_al_epitopo_BepiPred_mas_cercano={nearest} aa")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fase 3: cruce contra ground truth real de BepiPred-3.0.")
    parser.add_argument("--bepipred-fasta", type=Path, default=Path("data/bepipred3/Bcell_linepitope_top_20pct_preds.fasta"))
    parser.add_argument("--pipeline-csv", type=Path, default=Path("data/processed/fase3/residuos_detalle.csv"))
    args = parser.parse_args()

    bepipred_sequences = parse_bepipred_case_encoded(args.bepipred_fasta)

    for sequence_id in bepipred_sequences:
        predictions = load_pipeline_predictions(args.pipeline_csv, sequence_id)
        if not predictions:
            print(f"\n{sequence_id}: no se encontraron predicciones del pipeline en '{args.pipeline_csv}'.")
            continue
        compare_sequence(sequence_id, bepipred_sequences[sequence_id], predictions)

    return 0


if __name__ == "__main__":
    sys.exit(main())
