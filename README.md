# Pipeline de Descubrimiento de Epítopos Vacunales

Orquestador de terminal (`pipeline.py`) que procesa una secuencia de proteína
(FASTA) a través de 5 fases estrictas hasta producir una lista de péptidos
candidatos a vacuna, validados por antigenicidad (unión anotada de dos
motores independientes), ausencia de homología con el proteoma humano
(autoinmunidad) e inmunogenicidad T-helper (presentación MHC-II /
HLA-DR/DQ/DP, célula CD4+). La predicción MHC-I (CD8+, vía MHCflurry o
NetMHCpan) fue descartada metodológicamente: ver
`src/engines/netmhciipan_engine.py`.

## Flujo de trabajo (5 fases)

1. **Saneamiento FASTA** — valida y limpia la secuencia de entrada.
2. **Antigenicidad** — dos motores independientes ejecutados **100% en
   local**, cada uno con su propio auto-caché en CSV: BepiPred-3.0
   (subprocess sobre el código fuente oficial de DTU Health Tech) y EpiDope
   (subprocess directo sobre el binario del entorno conda dedicado, código
   abierto, sin licencia académica).
3. **Mapeo de epítopos y unión lógica anotada** — para cada motor, una
   ventana deslizante local (9 aa, tolerante a hasta 2 residuos por debajo
   del umbral por ventana, con fusión de ventanas solapadas/adyacentes)
   sobre sus scores de antigenicidad. Luego, TODA región detectada por
   BepiPred y/o por EpiDope avanza a la Fase 4 (`src/engines/consensus.py`):
   las regiones que solapan entre motores se **fusionan** (`start` mínimo,
   `end` máximo, sin recortar a la intersección, incluye fusión transitiva
   de cadenas de regiones), quedando marcadas en la columna `origen` como
   `'Consenso'` (fusión de ambos motores), `'BepiPred'` o `'EpiDope'` (un
   solo motor). Filtro de longitud inquebrantable: se descarta cualquier
   región final menor a 9 aa antes de la Fase 4.
4. **Filtro de tolerancia** — BLASTp local contra el proteoma humano, con
   E-value seleccionado dinámicamente por longitud del péptido (laxo para
   péptidos cortos, estricto para dominios/proteínas completas), descarta
   péptidos con alta homología (riesgo de autoinmunidad).
5. **Promiscuidad T-helper (MHC-II)** — NetMHCIIpan-4.3 ejecutado **100% en
   local** (subprocess) contra un panel de 27 alelos HLA-DR/DQ/DP de
   referencia del IEDB (`IEDB_REFERENCE_PANEL`). Cada péptido se enruta
   según su longitud: `<= 40 aa` en modo péptido exacto (una fila de salida
   por péptido); `> 40 aa` en modo proteína (NetMHCIIpan desliza
   internamente una ventana y evalúa todos los núcleos de unión candidatos
   dentro del fragmento — el modo péptido exacto revienta con un *buffer
   overflow* del binario para entradas > 56 aa). Un péptido/núcleo se
   aprueba solo si clasifica como aglutinador fuerte o débil (SB/WB, %Rank
   por defecto de NetMHCIIpan) en al menos 3 alelos distintos del panel.

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

### 3. EpiDope local (obligatorio para la Fase 2/3, segundo motor de antigenicidad)

A diferencia de BepiPred-3.0 y NetMHCIIpan-4.3, **EpiDope es código abierto**
(licencia MIT, [github.com/rnajena/EpiDope](https://github.com/rnajena/EpiDope)
— fork activamente mantenido, sucesor de github.com/flomock/EpiDope) y no
requiere solicitud académica. Fija, sin embargo, un stack de dependencias
antiguo y muy sensible a versiones exactas (Python 3.6, TensorFlow 1.13,
Keras 2.3, PyTorch 0.4, AllenNLP 0.7.2 para embeddings ELMo) incompatible con
el resto del pipeline: requiere un entorno conda dedicado, mismo patrón que
`.venv-bepipred`. Los pesos del modelo y los embeddings ELMo vienen
empaquetados en el propio repo — la inferencia es **100% local**, sin
ninguna llamada de red ni API externa.

**a) Crea el entorno conda dedicado** a partir del `epidope.yml` oficial del
repo (no instales los paquetes a mano: esa resolución de dependencias es
frágil y versión por versión termina en un entorno inconsistente):

```bash
git clone https://github.com/rnajena/EpiDope.git /tmp/EpiDope
conda env create -f /tmp/EpiDope/epidope.yml -p .conda-epidope
```

(El repo pesa varios cientos de MB por los embeddings ELMo empaquetados; la
creación del entorno puede tardar varios minutos por la cantidad de paquetes
pineados exactamente.)

**b) Ruta configurable, sin hardcoding.** Si instalaste EpiDope en otra
ubicación, apunta el pipeline con una de estas variables de entorno (en
orden de precedencia):

```bash
export EPIDOPE_BIN=/ruta/al/ejecutable/epidope       # bypass total de conda
# o
export EPIDOPE_CONDA_ENV=mi-entorno-epidope           # por nombre de entorno
# o
export EPIDOPE_CONDA_PREFIX=/ruta/a/.conda-epidope     # por prefijo de ruta (default)
```

Si el entorno no está instalado, la Fase 2 se detiene con un mensaje claro
indicando exactamente estos comandos, en vez de fallar con una traza opaca.

### 4. NCBI BLAST+ y base de datos del proteoma humano (obligatorio para la Fase 4)

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

**d) Selección dinámica de algoritmo y E-value.** Cada péptido candidato se
enruta automáticamente según su longitud: `<= 30 aa` usa `-task blastp-short`
con E-value laxo (50, evita que la estadística de BLAST descarte como "no
significativos" hits idénticos de péptidos cortos); `31–100 aa` usa `-task
blastp` con E-value 0.1; `> 100 aa` usa `-task blastp` con E-value 0.05
(estricto, evita ruido de homologías irrelevantes en consultas largas).
Configurable vía `BLAST_SHORT_PEPTIDE_MAX_LEN`, `BLAST_MEDIUM_PEPTIDE_MAX_LEN`
y `BLAST_EVALUE_SHORT` / `BLAST_EVALUE_MEDIUM` / `BLAST_EVALUE_LONG`.

Sin este paso, la Fase 4 se detiene con un error explicando exactamente qué
falta (`blastp` en el PATH o la base de datos en `reference_db/`).

### 5. NetMHCIIpan-4.3 local (obligatorio para la Fase 5)

Binario propietario de DTU Health Tech (predicción MHC-II / HLA-DR),
requiere licencia académica y descarga manual — igual patrón que BepiPred.

**a) Descarga manual obligatoria.** Solicítalo en
`https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/` (sección
"Downloads", requiere cuenta académica). No existe un `data.tar.gz` público
separado: el paquete `.tar.gz` que entrega DTU ya incluye la carpeta
`data/` completa (pseudosecuencias, listas de alelos, pesos de la red).

**b) Instalación.**
```bash
tar -xvf netMHCIIpan-4.3.Linux.tar.gz
mv netMHCIIpan-4.3 /ruta/al/proyecto/DiffSBDD/netMHCIIpan-4.3
```
Edita la línea `NMHOME` al inicio del script `netMHCIIpan-4.3/netMHCIIpan`
con la ruta absoluta de instalación (paso manual obligatorio según el propio
instructivo de DTU, no se puede resolver por variable de entorno):
```tcsh
setenv NMHOME /ruta/absoluta/a/DiffSBDD/netMHCIIpan-4.3
```

**c) Dependencia de sistema.** El script `netMHCIIpan` es un wrapper en
`tcsh` (no `bash`): instala el intérprete si no lo tienes (`apt-get install
tcsh` en Debian/Ubuntu).

**d) Panel de alelos.** La Fase 5 nunca evalúa un único alelo: usa
`IEDB_REFERENCE_PANEL` (27 alelos HLA-DR/DRB3/DRB4/DRB5/DQ/DP más
representativos de referencia poblacional del IEDB, ver
`src/engines/netmhciipan_engine.py`) para estimar cobertura poblacional
amplia. Un péptido se aprueba (`'Candidato Valido'`) solo si clasifica SB o
WB en al menos `NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES` (3 por defecto) alelos
distintos del panel.

## Uso

```bash
# Coloca tu(s) FASTA en fasta_inputs/, luego:
python pipeline.py --input fasta_inputs/secuencia.fasta

# El panel de 27 alelos HLA-DR/DQ/DP (IEDB_REFERENCE_PANEL) se evalúa siempre
# por defecto, sin necesidad de especificar nada. Para anexar un alelo extra:
python pipeline.py --input fasta_inputs/secuencia.fasta --alelo-extra "DRB1_1602"

# Los umbrales/longitud mínima de la Fase 3 son independientes por motor
# (las escalas de score de BepiPred y EpiDope no son comparables):
python pipeline.py --input fasta_inputs/secuencia.fasta \
    --bepipred-threshold 0.1512 --bepipred-min-length 9 \
    --epidope-threshold 0.818 --epidope-min-length 9
```

Resultados en `fasta_outputs/`:

| Archivo | Fase | Contenido |
|---|---|---|
| `<nombre>_clean.fasta` | 1 | FASTA saneado enviado a las fases siguientes |
| `<nombre>_bepipred_raw.csv` | 2 | Scores crudos por residuo de BepiPred-3.0 (caché) |
| `<nombre>_epidope_raw.csv` | 2 | Scores crudos por residuo de EpiDope (caché) |
| `<nombre>_bepipred_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (BepiPred-3.0) |
| `<nombre>_epidope_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (EpiDope) |
| `<nombre>_union_epitopes.csv` | 3 | Unión anotada BepiPred ∪ EpiDope (columna `origen`: `Consenso`/`BepiPred`/`EpiDope`), entrada de la Fase 4 |
| `<nombre>_blast_report.csv` | 4 | Veredicto de tolerancia (Segura / Autoinmunidad) |
| `netmhciipan_raw_peptide_mode.xls` | 5 | Salida cruda de NetMHCIIpan-4.3 en modo péptido exacto (multi-alelo), para trazabilidad |
| `netmhciipan_raw_protein_mode.xls` | 5 | Salida cruda de NetMHCIIpan-4.3 en modo proteína/ventana deslizante (péptidos > 40 aa), para trazabilidad |
| `candidatos_finales.csv` | 5 | **Reporte final** de candidatos con promiscuidad HLA-DR/DQ/DP y veredicto |
