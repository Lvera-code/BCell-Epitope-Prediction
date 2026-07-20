#!/usr/bin/env bash
# Wrapper de arranque para pipeline.py: para cada motor con instalacion local
# dedicada (BepiPred-3.0, ScanNet), si esa instalacion existe en su ubicacion
# esperada Y quien invoca este script no fijo ya la variable correspondiente
# por su cuenta, la fija automaticamente. DiscoTope-3.0 no necesita nada aqui:
# sus defaults en Settings ya son rutas relativas a la raiz del proyecto
# (.venv-discotope/, DiscoTope-3.0/), que 'cd "$SCRIPT_DIR"' mas abajo ya deja
# resueltas sin overrides. Reenvia todos los argumentos recibidos tal cual a
# pipeline.py.
#
# Uso identico a pipeline.py, por ejemplo:
#   ./run.sh --input fasta_inputs/secuencia.fasta
#   ./run.sh --input fasta_inputs/estructura.pdb --pdb-mode structure_only
#   ./run.sh --input fasta_inputs/secuencia.fasta --alelo-extra "DRB1_1602"
#
# No modifica PATH: blastp, tcsh y el interprete de Python de este proyecto
# deben estar disponibles en el entorno desde el que se invoca este script
# (ver README.md - Seccion de Instalacion). Si falta alguna instalacion,
# pipeline.py se detiene con un mensaje claro indicando exactamente que falta
# y como resolverlo, en vez de fallar con una traza opaca.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BEPIPRED_VENV_PYTHON="$SCRIPT_DIR/.venv-bepipred/bin/python"
if [[ -z "${BEPIPRED_PYTHON_BIN:-}" && -x "$BEPIPRED_VENV_PYTHON" ]]; then
    export BEPIPRED_PYTHON_BIN="$BEPIPRED_VENV_PYTHON"
fi

# ScanNet requiere Python 3.6.12 exacto: en la practica solo se consigue via
# conda (ver README.md - Seccion 7), nunca un venv comun. Si el entorno conda
# 'scannet_env' existe en la ubicacion default de SCANNET_PYTHON_BIN, se
# asume que el runtime 'venv' esta listo para usarse y se prefiere sobre el
# default 'docker' de Settings.SCANNET_RUNTIME (mas conveniente cuando no se
# valido el layout de la imagen Docker en esta maquina, ver README.md).
SCANNET_VENV_PYTHON="$HOME/miniconda3/envs/scannet_env/bin/python"
if [[ -z "${SCANNET_RUNTIME:-}" && -x "$SCANNET_VENV_PYTHON" ]]; then
    export SCANNET_RUNTIME="venv"
fi

exec python3 pipeline.py "$@"
