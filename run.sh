#!/usr/bin/env bash
# Wrapper de arranque para pipeline.py: prepara el entorno (interprete de
# BepiPred-3.0 en su venv dedicado, PATH con blastp/pandas) y reenvia todos
# los argumentos recibidos tal cual a pipeline.py.
#
# Uso identico a pipeline.py, por ejemplo:
#   ./run.sh --input fasta_inputs/secuencia.fasta
#   ./run.sh --input fasta_inputs/secuencia.fasta --alelo-extra "DRB1_1602"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="/home/enzo/miniconda3/envs/cnb_pipeline/bin:$PATH"
export BEPIPRED_PYTHON_BIN="$SCRIPT_DIR/.venv-bepipred/bin/python"

python3 pipeline.py "$@"
