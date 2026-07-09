"""Genera un sub-muestreo reproducible de fragmentos cortos (<=25aa) para
evaluacion manual cruzada contra el servidor web de BepiPred-3.0.

BepiPred no expone una API de lote automatizada para fragmentos aislados: el
usuario debe correrlos uno a uno en el servidor web. Este script selecciona
15 positivos + 15 negativos (semilla fija para reproducibilidad) del test set
real y los exporta con cabeceras cortas y trazables al ID original.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.fasta_parser import FastaParser

RANDOM_SEED = 42
N_PER_CLASS = 15
MIN_LENGTH = 10
MAX_LENGTH = 25

POSITIVE_FASTA = Path("data/training/test_positive.fasta")
NEGATIVE_FASTA = Path("data/training/test_negative.fasta")
OUTPUT_FASTA = Path("data/evaluation/bepipred_manual_batch.fasta")


def main() -> int:
    positive_records = [
        r for r in FastaParser.parse(POSITIVE_FASTA, min_length=1) if MIN_LENGTH <= len(r.sequence) <= MAX_LENGTH
    ]
    negative_records = [
        r for r in FastaParser.parse(NEGATIVE_FASTA, min_length=1) if MIN_LENGTH <= len(r.sequence) <= MAX_LENGTH
    ]

    if len(positive_records) < N_PER_CLASS or len(negative_records) < N_PER_CLASS:
        raise ValueError(
            f"No hay suficientes secuencias en [{MIN_LENGTH}-{MAX_LENGTH}]aa: "
            f"{len(positive_records)} positivas, {len(negative_records)} negativas disponibles, "
            f"se necesitan {N_PER_CLASS} de cada clase."
        )

    rng = random.Random(RANDOM_SEED)
    sampled_positive = rng.sample(positive_records, N_PER_CLASS)
    sampled_negative = rng.sample(negative_records, N_PER_CLASS)

    OUTPUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FASTA, "w") as handle:
        for idx, record in enumerate(sampled_positive, start=1):
            handle.write(f">POS_{idx} original_id={record.id}\n{record.sequence}\n")
        for idx, record in enumerate(sampled_negative, start=1):
            handle.write(f">NEG_{idx} original_id={record.id}\n{record.sequence}\n")

    print(f"Exportadas {N_PER_CLASS} positivas + {N_PER_CLASS} negativas (semilla={RANDOM_SEED}) a '{OUTPUT_FASTA}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
