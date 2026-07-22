#!/usr/bin/env bash
# Descarga el modelo AlphaFold DB (ultima version) + FASTA canonico para un
# accession de UniProt, verifica que la cobertura del modelo es completa
# (sin huecos respecto al FASTA canonico) y corre el pipeline en modo
# structure_and_sequence (los 4 motores de antigenicidad: BepiPred + EpiDope
# + DiscoTope + ScanNet).
#
# Uso:
#   ./run_uniprot_target.sh P51665
#   ./run_uniprot_target.sh P51665 PSMD7
#
# El segundo argumento (nombre corto, opcional) solo se usa para nombrar los
# ficheros de salida; si se omite, se usa el accession solo.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <UNIPROT_ACCESSION> [nombre_corto]" >&2
    exit 1
fi

ACC="$1"
LABEL="${2:-$ACC}_${ACC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PDB_FILE="fasta_inputs/${LABEL}_AF.pdb"
FASTA_FILE="fasta_inputs/${LABEL}.fasta"

echo "== Descargando FASTA canonico de UniProt (${ACC}) =="
curl -s --max-time 15 "https://rest.uniprot.org/uniprotkb/${ACC}.fasta" -o "$FASTA_FILE"
cat "$FASTA_FILE"

echo ""
echo "== Consultando ultima version del modelo AlphaFold DB para ${ACC} =="
PDB_URL=$(curl -s --max-time 15 "https://alphafold.ebi.ac.uk/api/prediction/${ACC}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['pdbUrl'])")
echo "URL: $PDB_URL"

echo ""
echo "== Descargando modelo AlphaFold =="
curl -s --max-time 30 "$PDB_URL" -o "$PDB_FILE"
ls -la "$PDB_FILE"

echo ""
echo "== Verificando cobertura completa (residuos con coordenadas vs. longitud canonica) =="
python3 -c "
import sys

fasta_len = 0
with open('$FASTA_FILE') as f:
    for line in f:
        if not line.startswith('>'):
            fasta_len += len(line.strip())

resnums = set()
with open('$PDB_FILE') as f:
    for line in f:
        if line.startswith('ATOM'):
            resnums.add(int(line[22:26]))

print(f'Longitud FASTA canonico: {fasta_len} aa')
print(f'Residuos con coordenadas en el modelo: {len(resnums)} aa (min={min(resnums)}, max={max(resnums)})')
if len(resnums) != fasta_len:
    print('AVISO: el modelo NO cubre la longitud completa -- considera correr tambien')
    print('       el camino 1 (FASTA) por separado para no perder las regiones sin resolver.')
else:
    print('Cobertura completa: el camino 3 (structure_and_sequence) ya es suficiente.')
"

echo ""
echo "== Corriendo pipeline (camino 3: structure_and_sequence, los 4 motores) =="
./run.sh --input "$PDB_FILE" --pdb-mode structure_and_sequence
