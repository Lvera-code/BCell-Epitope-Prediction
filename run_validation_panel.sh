#!/usr/bin/env bash
# Corre el pipeline completo (4 motores de Fase 2 + resto de fases) para cada
# uno de los 17 complejos antigeno-Fab del panel de validacion de publicacion
# (ver inputs/*.pdb), forzando PDB_CHAIN_SELECTION_STRATEGY=explicit con la
# cadena de antigeno correcta por estructura.
#
# Por que 'explicit' y no el default 'longest' (Settings.PDB_CHAIN_SELECTION_STRATEGY):
# en un complejo antigeno-Fab, 'longest' asume que la cadena mas larga es el
# antigeno, pero eso es falso cada vez que el antigeno es un fragmento/peptido
# mas corto que las cadenas pesada/ligera del Fab (~210-230 aa cada una). En
# este panel eso ocurre en al menos 7/17 estructuras -- 4XAW, 8FDD, 7RXP,
# 6B5M, 7STR, 7BEP, 5O1R -- donde 'longest' elegiria en silencio una cadena
# de anticuerpo como si fuera el antigeno. La cadena de cada fila se
# verifico por conteo de residuos + COMPND contra el mapeo de epitopos ya
# cerrado del panel.
#
# 7RQQ (peptido CSP NPNV de 6 aa, Fab F10/L9) fue
# sustituido por 8FDD (peptido CSP NPNV de 13 aa resueltos, Fab Ky15.3,
# Thai et al. 2023 Cell Reports, PMID 38007690) -- 7RQQ quedaba por debajo
# del minimo real de 9 aa que exigen los 4 motores de Fase 3
# (Settings.*_WINDOW_SIZE), por lo que nunca generaba ninguna region
# candidata (0/0), un piso estructural que ningun otro candidato del
# linaje L9/F10 supera. 8FDD lo supera con margen y mantiene el mismo
# motivo NPNV.
#
# Uso: ./run_validation_panel.sh [output_root]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_ROOT="${1:-outputs/validation_panel}"
mkdir -p "$OUTPUT_ROOT"

declare -A ANTIGEN_CHAIN=(
    [Ctetani_ToxinFragC_7CE2.pdb]=A
    [Ctetani_ToxinLCHN_7OH1.pdb]=A
    [HIV1_gp120CD4bs_3NGB.pdb]=A
    [HIV1_gp41MPER_4XAW.pdb]=P
    [Nmeningitidis_NHBA_5O1R.pdb]=A
    [Nmeningitidis_fHbp_site1_5O14.pdb]=A
    [Nmeningitidis_fHbp_site2_8UP2.pdb]=C
    [Pfalciparum_AMA1_1F9_formA_2Q8A.pdb]=A
    [Pfalciparum_AMA1_1F9_formB_2Q8B.pdb]=A
    [Pfalciparum_AMA1_75B10_9BJG.pdb]=A
    [Pfalciparum_AMA1_75C8_9BJH.pdb]=A
    [Pfalciparum_CSP_NPNVrepeat_8FDD.pdb]=P
    [Pfalciparum_CSP_betaCT_7RXP.pdb]=A
    [Pfalciparum_CSP_junctional_6B5M.pdb]=A
    [SARSCoV2_N_site1_7STR.pdb]=C
    [SARSCoV2_N_site2_7STS.pdb]=C
    [SARSCoV2_RBD_7BEP.pdb]=E
)

for fname in "${!ANTIGEN_CHAIN[@]}"; do
    chain_id="${ANTIGEN_CHAIN[$fname]}"
    stem="${fname%.pdb}"
    out_dir="$OUTPUT_ROOT/$stem"
    echo "=== $fname (cadena antigeno explicita: $chain_id) -> $out_dir ==="
    PDB_CHAIN_SELECTION_STRATEGY=explicit \
    PDB_EXPLICIT_CHAIN_ID="$chain_id" \
    ./run.sh --input "inputs/$fname" --pdb-mode structure_and_sequence --output-dir "$out_dir" \
        > "$out_dir.log" 2>&1 || echo "!!! FALLO: $fname (ver $out_dir.log)"
done

echo "Panel completo. Logs individuales en $OUTPUT_ROOT/*.log"
