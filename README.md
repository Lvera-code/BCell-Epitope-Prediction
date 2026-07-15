# Pipeline de Descubrimiento de Epítopos Vacunales

Orquestador de terminal (`pipeline.py`) que procesa una secuencia de proteína
(FASTA) a través de 5 fases estrictas hasta producir una lista de péptidos
candidatos a vacuna, validados por antigenicidad (unión anotada de dos
motores independientes), ausencia de homología con el proteoma humano
(riesgo de autoinmunidad) e inmunogenicidad T-helper (presentación MHC-II /
HLA-DR/DQ/DP, célula CD4+). Todas las fases corren **100% en local**
(subprocess sobre binarios/paquetes instalados en tu máquina): el pipeline
nunca hace una llamada de red durante la inferencia. Predice exclusivamente
presentación T-helper (MHC-II/CD4+); no incluye predicción MHC-I/CD8+.

## Flujo de trabajo (5 fases)

1. **Saneamiento FASTA** — valida y limpia la(s) secuencia(s) de entrada
   (`fasta_inputs/`, admite FASTA multi-registro con varias proteínas).
2. **Antigenicidad** — dos motores independientes ejecutados en local, cada
   uno con su propio auto-caché en CSV: BepiPred-3.0 (subprocess sobre el
   código fuente oficial de DTU Health Tech) y EpiDope (subprocess sobre el
   binario del entorno conda dedicado, código abierto).
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
   péptidos con alta homología (`status = 'Autoinmunidad'`, umbral
   `BLAST_IDENTITY_THRESHOLD`, 75% por defecto).
5. **Promiscuidad T-helper (MHC-II)** — NetMHCIIpan-4.3 ejecutado en local
   contra un panel de 27 alelos HLA-DR/DQ/DP de referencia del IEDB
   (`IEDB_REFERENCE_PANEL`). Cada péptido se enruta según su longitud:
   `<= 40 aa` en modo péptido exacto (una fila de salida por péptido);
   `> 40 aa` en modo proteína (NetMHCIIpan desliza internamente una ventana
   de 15 aa y evalúa todos los núcleos de unión candidatos dentro del
   fragmento — el modo péptido exacto revienta con un *buffer overflow* del
   binario para entradas > 55 aa con el panel de 27 alelos que usa este
   pipeline, de ahí el margen de seguridad de 40).

   **Fiabilidad para síntesis/validación experimental.** NetMHCIIpan marca
   un alelo como `Inverted=1` cuando su procedimiento de alineación del
   núcleo de unión ajusta mejor leyendo el péptido en reversa que en su
   sentido real — no hay evidencia estructural de que MHC-II presente
   péptidos "al revés"; se trata como un artefacto/limitación del
   alineador, de menor confianza que un ajuste en orientación normal. Este
   pipeline **descarta por completo** los alelos invertidos antes de
   calcular nada: no cuentan para la promiscuidad ni pueden ser el alelo
   "ganador" que determina el núcleo de 9 aa reportado. Un péptido se
   aprueba (`'Candidato Valido'`) solo si clasifica como aglutinador fuerte
   o débil (SB/WB, umbrales de %Rank por defecto de NetMHCIIpan) en al
   menos `NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES` (3 por defecto) alelos
   **en orientación normal** del panel.

   **Deduplicación de ventanas del modo proteína.** Al deslizar la ventana
   de 15 aa un residuo a la vez, un mismo núcleo de 9 aa suele "ganar" en
   varias ventanas consecutivas — no son epítopos distintos, son la misma
   predicción vista desde offsets vecinos. Las ventanas se agrupan por
   `(accession, núcleo de 9 aa, promiscuidad)` **exactos**: dentro de cada
   grupo con más de una fila se conserva únicamente la de mejor (menor)
   %Rank. Si el núcleo difiere aunque sea en 1 aminoácido, o si la
   promiscuidad difiere (con el mismo núcleo), las filas **no** se
   fusionan: son predicciones distintas y ambas se reportan, para no perder
   ninguna variante con soporte de unión real.

Todos los resultados intermedios y el reporte final se guardan en
`fasta_outputs/`.

## Instalación

Sigue estos pasos **en orden** antes de la primera ejecución.

### 1. Entorno Python

```bash
pip install -r requirements.txt
```

### 2. BepiPred-3.0 local (obligatorio para la Fase 2)

La Fase 2 ejecuta BepiPred-3.0 enteramente en tu máquina vía `subprocess`,
sin ninguna llamada de red.

**a) Descarga manual obligatoria.** Por restricciones de licencia académica,
DTU Health Tech no permite redistribuir el código fuente de BepiPred-3.0: no
está incluido en este repositorio (ver `.gitignore`). Debes solicitarlo tú
mismo desde:

[https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=bepipred&version=3.0&packageversion=3.0b&platform=src](https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=bepipred&version=3.0&packageversion=3.0b&platform=src)

**b) Colócalo en la raíz del proyecto** de forma que quede así:

```
<raiz-del-proyecto>/
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

> **Nota (confirmado en Python 3.10+):** los pines exactos del
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
empaquetados en el propio repo de EpiDope — la inferencia es 100% local, sin
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

La Fase 4 corre BLASTp en local, nunca en la nube. Necesitas:

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
mv netMHCIIpan-4.3 /ruta/al/proyecto/netMHCIIpan-4.3
```
Edita la línea `NMHOME` al inicio del script `netMHCIIpan-4.3/netMHCIIpan`
con la ruta absoluta de instalación (paso manual obligatorio según el propio
instructivo de DTU, no se puede resolver por variable de entorno):
```tcsh
setenv NMHOME /ruta/absoluta/al/proyecto/netMHCIIpan-4.3
```

**c) Dependencia de sistema.** El script `netMHCIIpan` es un wrapper en
`tcsh` (no `bash`): instala el intérprete si no lo tienes (`apt-get install
tcsh` en Debian/Ubuntu).

**d) Panel de alelos.** La Fase 5 nunca evalúa un único alelo: usa
`IEDB_REFERENCE_PANEL` (27 alelos HLA-DR/DRB3/DRB4/DRB5/DQ/DP más
representativos de referencia poblacional del IEDB, ver
`src/engines/netmhciipan_engine.py`) para estimar cobertura poblacional
amplia.

## Uso

```bash
# Coloca tu(s) FASTA en fasta_inputs/ (admite multi-registro), luego:
python pipeline.py --input fasta_inputs/secuencia.fasta

# El panel de 27 alelos HLA-DR/DQ/DP (IEDB_REFERENCE_PANEL) se evalúa siempre
# por defecto, sin necesidad de especificar nada. Para anexar alelo(s) extra
# (formato NetMHCIIpan, separados por coma SIN espacios; se valida el formato
# de inmediato, antes de correr cualquier fase):
python pipeline.py --input fasta_inputs/secuencia.fasta \
    --alelo-extra "DRB1_1602,HLA-DQA10501-DQB10201"

# Los umbrales/longitud mínima de la Fase 3 son independientes por motor
# (las escalas de score de BepiPred y EpiDope no son comparables):
python pipeline.py --input fasta_inputs/secuencia.fasta \
    --bepipred-threshold 0.1512 --bepipred-min-length 9 \
    --epidope-threshold 0.818 --epidope-min-length 9
```

Corre `python pipeline.py --help` para ver todos los flags disponibles
(umbral/E-value de BLAST, carpeta de salida, etc.), todos con su valor por
defecto documentado.

### Salida en consola

Las tablas de la Fase 5 resaltan en color el núcleo de unión de 9 aa dentro
de la ventana de 15 aa evaluada (solo en terminal, no afecta el CSV). Cuando
el FASTA de entrada tiene varias proteínas (varios `accession`), la tabla
final se ordena por proteína/posición y separa cada una con una línea
divisoria, para no leerlas como una lista continua.

### Archivos generados en `fasta_outputs/`

`<nombre>` es el nombre del FASTA de entrada sin extensión (`--input`).

| Archivo | Fase | Contenido |
|---|---|---|
| `<nombre>_clean.fasta` | 1 | FASTA saneado enviado a las fases siguientes |
| `<nombre>_bepipred_raw.csv` | 2 | Scores crudos por residuo de BepiPred-3.0 (caché) |
| `<nombre>_epidope_raw.csv` | 2 | Scores crudos por residuo de EpiDope (caché) |
| `<nombre>_bepipred_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (BepiPred-3.0) |
| `<nombre>_epidope_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (EpiDope) |
| `<nombre>_union_epitopes.csv` | 3 | Unión anotada BepiPred ∪ EpiDope (columna `origen`), entrada de la Fase 4 |
| `<nombre>_blast_report.csv` | 4 | Veredicto de tolerancia (Segura / Autoinmunidad) por región |
| `netmhciipan_raw_peptide_mode.xls` | 5 | Salida cruda de NetMHCIIpan-4.3, modo péptido exacto (`<= 40` aa), para trazabilidad. Solo se genera si hubo al menos un péptido en ese rango |
| `netmhciipan_raw_protein_mode.xls` | 5 | Salida cruda de NetMHCIIpan-4.3, modo proteína/ventana deslizante (`> 40` aa), para trazabilidad. Solo se genera si hubo al menos un fragmento en ese rango |
| `candidatos_finales.csv` | 5 | **Reporte final** (ver formato abajo) |

### Formato de `candidatos_finales.csv`

Es la salida de la Fase 5 después del cruce con la Fase 3/4 (traceback) y la
deduplicación de ventanas redundantes: solo contiene candidatos con
`veredicto == 'Candidato Valido'`, ya enriquecidos con su región de origen.

| Columna | Significado |
|---|---|
| `accession` | Identificador de la proteína de origen (primer token de la cabecera FASTA) |
| `sequence_f5` | Péptido/ventana evaluado por NetMHCIIpan (péptido completo en modo exacto, o la ventana de 15 aa ganadora en modo proteína) |
| `core_9aa` | Núcleo de unión de 9 aa del alelo con mejor %Rank **en orientación normal** (los alelos invertidos nunca determinan este valor) |
| `start` / `end` | Coordenadas absolutas en la proteína de origen (1-indexado), recalculadas por traceback contra la región de la Fase 3/4 |
| `origen` | De dónde viene la región de la Fase 3: `'Consenso'` (BepiPred + EpiDope), `'BepiPred'` o `'EpiDope'` |
| `n_alelos_promiscuos` | Cuántos alelos del panel (de 27, o más si usaste `--alelo-extra`) clasifican el péptido como SB/WB **en orientación normal** — este número decide el veredicto |
| `n_alelos_evaluados` | Tamaño total del panel evaluado |
| `min_rank_el` | Mejor (menor) %Rank entre los alelos en orientación normal |
| `bepipred_score` / `epidope_score` | Score medio de antigenicidad de la Fase 3 de cada motor (`NaN` si ese motor no detectó esa región) |

## Tests

La suite de tests (`tests/`, `pytest`) cubre la lógica pura de cada fase —
fusión de regiones en Fase 3, selección de task/E-value en Fase 4, parseo
del `.xls` de NetMHCIIpan y exclusión de alelos invertidos, traceback de
coordenadas, deduplicación de ventanas y validación de `--alelo-extra` en
Fase 5 — sin depender de BepiPred, EpiDope, BLAST+ ni NetMHCIIpan
instalados: no invoca ningún subprocess real.

```bash
pip install -r requirements-dev.txt
pytest
```
