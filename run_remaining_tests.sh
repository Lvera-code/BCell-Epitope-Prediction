#!/bin/bash
# Comandos para correr los casos de validacion pendientes.
# Ejecutar desde la raiz de B-Cell-Epitope-Prediction/, con el venv del pipeline activo.
set -euo pipefail
cd "$(dirname "$0")"

# --- Controles negativos (proteinas humanas -- deben rechazarse por autoinmunidad) ---
python3 pipeline.py --input fasta_inputs/negative_controls.fasta   # GAPDH+PSMD7+PODXL+THBS2+SLC8A1, camino FASTA (rapido)
python3 pipeline.py --input fasta_inputs/PSMD7_P51665_AF.pdb        # mismo PSMD7, camino estructura (DiscoTope+ScanNet)
python3 pipeline.py --input fasta_inputs/PODXL_O00592_AF.pdb        # idem PODXL
python3 pipeline.py --input fasta_inputs/THBS2_P35442_AF.pdb        # idem THBS2
python3 pipeline.py --input fasta_inputs/SLC8A1_P32418_AF.pdb       # idem SLC8A1 (caso ya confirmado que Fase 3b SI descarta candidatos aca)

# --- Casos extremos ---
python3 pipeline.py --input fasta_inputs/multi_distinct_test.fasta  # 2 proteinas sin relacion en un mismo FASTA
python3 pipeline.py --input fasta_inputs/MonkeyPoxSequences.fasta   # 6 accessions reales (MPXV) en un mismo FASTA

# --- Funcionamiento de todas las fases ---
python3 pipeline.py --input fasta_inputs/GP120.fasta                # camino FASTA completo (BepiPred+EpiDope, ya validado con hits bnAb reales)
python3 pipeline.py --input fasta_inputs/7c4s.pdb                   # camino estructura completo (BepiPred+EpiDope+DiscoTope+ScanNet a la vez)

# --- Clados VIH ---
python3 pipeline.py --input fasta_inputs/HIV_clade_B_reference.fasta  # JR-FL, YU2, BaL, SF162 (GP120.fasta ya cubre HXB2/clado B)
python3 pipeline.py --input fasta_inputs/HIV_clade_C_reference.fasta  # 93IN905, DU422, CAP210, 97ZA009, CAP206
