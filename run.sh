#!/usr/bin/env bash
# Wrapper de arranque para pipeline.py: si existe el venv dedicado de
# BepiPred-3.0 (.venv-bepipred/, ver README.md - Seccion de Instalacion) y
# quien invoca este script no fijo ya BEPIPRED_PYTHON_BIN por su cuenta,
# apunta el pipeline a ese interprete. Reenvia todos los argumentos
# recibidos tal cual a pipeline.py.
#
# Uso identico a pipeline.py, por ejemplo:
#   ./run.sh --input fasta_inputs/secuencia.fasta
#   ./run.sh --input fasta_inputs/secuencia.fasta --alelo-extra "DRB1_1602"
#
# No modifica PATH: blastp, tcsh y el interprete de Python de este proyecto
# deben estar disponibles en el entorno desde el que se invoca este script
# (ver README.md - Seccion de Instalacion). Si falta alguno, pipeline.py se
# detiene con un mensaje claro indicando exactamente que falta y como
# resolverlo, en vez de fallar con una traza opaca.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BEPIPRED_VENV_PYTHON="$SCRIPT_DIR/.venv-bepipred/bin/python"
if [[ -z "${BEPIPRED_PYTHON_BIN:-}" && -x "$BEPIPRED_VENV_PYTHON" ]]; then
    export BEPIPRED_PYTHON_BIN="$BEPIPRED_VENV_PYTHON"
fi

exec python3 pipeline.py "$@"
