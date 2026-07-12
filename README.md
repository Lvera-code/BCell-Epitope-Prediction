# Pipeline de Descubrimiento de Epítopos Vacunales

Orquestador de terminal (`pipeline.py`) que procesa una secuencia de proteína
(FASTA) a través de 5 fases estrictas hasta producir una lista de péptidos
candidatos a vacuna, validados por antigenicidad, ausencia de homología con
el proteoma humano (autoinmunidad) e inmunogenicidad T-helper (presentación
MHC-II / HLA-DR, célula CD4+). La predicción MHC-I (CD8+, vía MHCflurry o
NetMHCpan) fue descartada metodológicamente: ver `src/engines/netmhciipan_engine.py`.

## Flujo de trabajo (5 fases)

1. **Saneamiento FASTA** — valida y limpia la secuencia de entrada.
2. **Antigenicidad** — BepiPred-3.0 ejecutado **100% en local** (subprocess
   sobre el código fuente oficial de DTU Health Tech), con auto-caché local
   en CSV.
3. **Mapeo de epítopos** — ventana deslizante local (9 aa, tolerante a hasta
   2 residuos por debajo del umbral por ventana, con fusión de ventanas
   solapadas/adyacentes) sobre los scores de antigenicidad.
4. **Filtro de tolerancia** — BLASTp local contra el proteoma humano, con
   E-value seleccionado dinámicamente por longitud del péptido (laxo para
   péptidos cortos, estricto para dominios/proteínas completas), descarta
   péptidos con alta homología (riesgo de autoinmunidad).
5. **Promiscuidad T-helper (MHC-II)** — NetMHCIIpan-4.3 ejecutado **100% en
   local** (subprocess) contra un panel de 15 alelos HLA-DR de referencia del
   IEDB (`IEDB_DR_PANEL`). Un péptido se aprueba solo si clasifica como
   aglutinador fuerte o débil (SB/WB, %Rank por defecto de NetMHCIIpan) en al
   menos 3 alelos distintos del panel.

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

### 4. NetMHCIIpan-4.3 local (obligatorio para la Fase 5)

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
`IEDB_DR_PANEL` (15 alelos HLA-DR/DRB3/DRB4/DRB5 de referencia poblacional
del IEDB, ver `src/engines/netmhciipan_engine.py`) para estimar cobertura
poblacional amplia. Un péptido se aprueba (`'Candidato Valido'`) solo si
clasifica SB o WB en al menos `NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES` (3 por
defecto) alelos distintos del panel.

## Uso

```bash
# Coloca tu(s) FASTA en fasta_inputs/, luego:
python pipeline.py --input fasta_inputs/secuencia.fasta

# El panel de 15 alelos HLA-DR (IEDB_DR_PANEL) se evalúa siempre por
# defecto, sin necesidad de especificar nada. Para anexar un alelo extra:
python pipeline.py --input fasta_inputs/secuencia.fasta --alelo-extra "DRB1_1602"
```

Resultados en `fasta_outputs/`:

| Archivo | Fase | Contenido |
|---|---|---|
| `<nombre>_clean.fasta` | 1 | FASTA saneado enviado a las fases siguientes |
| `<nombre>_bepipred_raw.csv` | 2 | Scores crudos por residuo (caché) |
| `<nombre>_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente |
| `<nombre>_blast_report.csv` | 4 | Veredicto de tolerancia (Segura / Autoinmunidad) |
| `netmhciipan_raw.xls` | 5 | Salida cruda de NetMHCIIpan-4.3 (multi-alelo), para trazabilidad |
| `candidatos_finales.csv` | 5 | **Reporte final** de candidatos con promiscuidad HLA-DR y veredicto |
