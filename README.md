# Pipeline de Descubrimiento de Epítopos Vacunales

Orquestador de terminal (`pipeline.py`) que procesa una secuencia de proteína
(FASTA) a través de 5 fases estrictas hasta producir una lista de péptidos
candidatos a vacuna, validados por antigenicidad, ausencia de homología con
el proteoma humano (autoinmunidad) e inmunogenicidad (presentación por MHC-I).

## Flujo de trabajo (5 fases)

1. **Saneamiento FASTA** — valida y limpia la secuencia de entrada.
2. **Antigenicidad** — BepiPred-3.0 ejecutado **100% en local** (subprocess
   sobre el código fuente oficial de DTU Health Tech), con auto-caché local
   en CSV.
3. **Mapeo de epítopos** — agrupa localmente regiones de residuos contiguos
   por encima de un umbral de score.
4. **Filtro de tolerancia** — BLASTp local contra el proteoma humano, descarta
   péptidos con alta homología (riesgo de autoinmunidad).
5. **Inmunogenicidad** — predicción de afinidad HLA (IC50) vía NetMHCpan o
   MHCflurry.

Todos los resultados intermedios y el reporte final se guardan en
`fasta_outputs/`.

## Instalación

Sigue estos pasos **en orden** antes de la primera ejecución.

### 1. Entorno Python

```bash
pip install -r requirements.txt
```

### 2. BepiPred-3.0 local (obligatorio para la Fase 2)

> ⚠️ **Se abandonó la estrategia vía API/nube (BioLib)** por la latencia
> impredecible de los cold-start de los contenedores ESM-2 bajo carga
> pública (peticiones que no llegaban a completar ni tras ~1h de espera). La
> Fase 2 ahora ejecuta BepiPred-3.0 **enteramente en tu máquina** vía
> `subprocess`, sin ninguna llamada de red.

**a) Descarga manual obligatoria.** Por restricciones de licencia académica,
DTU Health Tech no permite redistribuir el código fuente de BepiPred-3.0: no
está incluido en este repositorio (ver `.gitignore`). Debes solicitarlo tú
mismo desde:

[https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=bepipred&version=3.0&packageversion=3.0b&platform=src](https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=bepipred&version=3.0&packageversion=3.0b&platform=src)

**b) Colócalo en la raíz del proyecto** de forma que quede así:

```
DiffSBDD/
└── bepipred-3.0b.src/
    └── BepiPred3_src/
        ├── bepipred3_CLI.py
        ├── bp3/
        └── requirements.txt
```

Si prefieres otra ubicación, define la variable de entorno `BEPIPRED_HOME`
apuntando a la carpeta que contiene `bepipred3_CLI.py` — el pipeline nunca
asume una ruta fija.

**c) Instala sus dependencias en un venv APARTE.** BepiPred-3.0 fija
versiones antiguas (`torch==1.12.0`, `numpy==1.20.2`) que pueden chocar con
las de este proyecto:

```bash
python -m venv .venv-bepipred
source .venv-bepipred/bin/activate
pip install -r bepipred-3.0b.src/BepiPred3_src/requirements.txt
deactivate
```

> ⚠️ **Nota (confirmado en Python 3.10+):** los pines exactos del
> `requirements.txt` de BepiPred (de 2022) no tienen wheels precompiladas
> para Python 3.10+ y fallan al compilar `numpy==1.20.2` desde código fuente
> (`Broken toolchain: cannot link a simple C program`) si no tienes un
> compilador de C instalado. Si te pasa esto, instala versiones modernas en
> su lugar — el código de `bp3/bepipred3.py` funciona igual con ellas:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install fair-esm plotly pandas
> ```

Luego apunta el pipeline a ese intérprete:

```bash
export BEPIPRED_PYTHON_BIN=$(pwd)/.venv-bepipred/bin/python
```

(Si omites este paso, se usa el mismo intérprete que corre `pipeline.py`.)

> La primera ejecución descargará los modelos ESM-2 (~2.5 GB) al caché de
> `torch hub`; puede tardar varios minutos según tu conexión y hardware
> (CPU vs GPU). Si el script `bepipred3_CLI.py` no se encuentra en la ruta
> configurada, el pipeline se detiene con un mensaje claro indicando este
> mismo enlace de descarga, en vez de fallar con una traza opaca.

### 3. NCBI BLAST+ y base de datos del proteoma humano (obligatorio para la Fase 4)

La Fase 4 corre **BLASTp en local**, nunca en la nube. Necesitas:

**a) Instalar el binario `blastp` / `makeblastdb`** (paquete NCBI BLAST+):

```bash
# conda (recomendado)
conda install -c bioconda blast

# o Debian/Ubuntu
sudo apt install ncbi-blast+
```

**b) Descargar el proteoma humano e indexarlo localmente** en
`reference_db/`:

```bash
mkdir -p reference_db
# Descarga el proteoma de referencia de Homo sapiens (UniProt) como FASTA:
# https://www.uniprot.org/proteomes/UP000005640 -> "Download" -> FASTA (canónico)
# Guárdalo como reference_db/human_proteome.fasta

makeblastdb -in reference_db/human_proteome.fasta \
            -dbtype prot \
            -out reference_db/human_proteome_db
```

**c) Ruta configurable, sin hardcoding.** La Fase 4 lee la ruta de la base de
datos desde la variable de entorno `BLAST_HUMAN_DB` (por defecto
`reference_db/human_proteome_db`, el layout del paso anterior). Si usas otra
ubicación:

```bash
export BLAST_HUMAN_DB=/ruta/a/tu/base_de_datos
```

Si `blastp` no está en el `PATH` o la base de datos configurada no existe, la
Fase 4 se detiene con un mensaje claro (igual que la Fase 2 con BepiPred), en
vez de fallar con una traza opaca.

**d) Selección dinámica de algoritmo.** Cada péptido candidato se enruta
automáticamente según su longitud: `< 30 aa` usa `-task blastp-short`
(word_size y matriz de sustitución ajustados para péptidos cortos, guía
estándar de NCBI); `>= 30 aa` usa `-task blastp` clásico. Configurable vía
`BLAST_SHORT_PEPTIDE_MAX_LEN`.

Sin este paso, la Fase 4 se detiene con un error explicando exactamente qué
falta (`blastp` en el PATH o la base de datos en `reference_db/`).

### 4. Motor de inmunogenicidad (Fase 5)

Elige uno según el flag `--inmuno`:

- **`mhcflurry`** (recomendado, 100% local vía pip):
  ```bash
  pip install mhcflurry
  mhcflurry-downloads fetch
  ```
- **`netmhcpan`**: binario propietario de DTU Health Tech, requiere licencia
  académica y descarga manual desde su sitio oficial. Instálalo y añádelo a
  tu `PATH` como `netMHCpan`.

## Uso

```bash
# Coloca tu(s) FASTA en fasta_inputs/, luego:
python pipeline.py --input fasta_inputs/secuencia.fasta --inmuno netmhcpan --alelo "HLA-A*02:01"

# --alelo es opcional (por defecto: HLA-A02:01)
python pipeline.py --input fasta_inputs/secuencia.fasta --inmuno mhcflurry
```

Resultados en `fasta_outputs/`:

| Archivo | Fase | Contenido |
|---|---|---|
| `<nombre>_clean.fasta` | 1 | FASTA saneado enviado a las fases siguientes |
| `<nombre>_bepipred_raw.csv` | 2 | Scores crudos por residuo (caché) |
| `<nombre>_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente |
| `<nombre>_blast_report.csv` | 4 | Veredicto de tolerancia (Segura / Autoinmunidad) |
| `candidatos_finales.csv` | 5 | **Reporte final** de candidatos con IC50 y veredicto |
