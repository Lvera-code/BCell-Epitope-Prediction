# Pipeline de Descubrimiento de Epítopos Vacunales

Orquestador de terminal (`pipeline.py`) que procesa un FASTA de secuencia o
una estructura (PDB/mmCIF) a través de 11 fases: desde antigenicidad (unión
anotada de hasta 4 motores independientes: 2 de secuencia + 2
estructurales), enmascarado de regiones transmembrana/péptido señal (no
accesibles a anticuerpos), ausencia de homología con el proteoma humano
(riesgo de autoinmunidad), alergenicidad, N-glicosilación, inmunogenicidad
T-helper (MHC-II/HLA-DR/DQ/DP, CD4+) e inmunogenicidad T-citotóxica
(MHC-I/HLA-A/B/C, CD8+, con evidencia de corte proteasomal) y cruce con
epítopos de anticuerpos ampliamente neutralizantes (bnAb) conocidos, hasta el
**ensamblaje automático de un constructo multi-epítopo** con los mejores
candidatos y un **chequeo de alergenicidad/toxicidad/antigenicidad/péptido
señal sobre ese constructo ya ensamblado** (no por péptido individual).
Todas las fases corren **100% en local** (subprocess sobre binarios/paquetes
instalados en tu máquina): el pipeline nunca hace una llamada de red durante
la inferencia. Cada fase pesada (3b/4/4b/4c/5/5b/6/7/8) se auto-cachea por hash
de contenido de su input — reiniciar una corrida interrumpida con el mismo
input/parámetros salta directo a la fase que falló, ver "Checkpointing" más
abajo.

## Tres caminos de entrada

El tipo de archivo pasado a `--input` se detecta automáticamente
(`src/utils/input_router.py`) y decide qué motores de Fase 2 corren:

| Camino | Input | Motores activos | Notas |
|---|---|---|---|
| **1** | FASTA | BepiPred-3.0 + EpiDope | Comportamiento original, sin cambios |
| **2** | PDB/mmCIF, `--pdb-mode structure_only` | DiscoTope-3.0 + ScanNet | BepiPred/EpiDope nunca se invocan |
| **3** | PDB/mmCIF, `--pdb-mode structure_and_sequence` (default para input de estructura) | Los 4 motores | Se deriva un FASTA canónico (ATMSEQ) de la estructura para BepiPred/EpiDope |

Para input de estructura, la Fase 1 (saneamiento FASTA) se reemplaza por la
**Fase 1.5** (`src/utils/structure_parser.py`, vía `gemmi`): elige una cadena
de referencia (`PDB_CHAIN_SELECTION_STRATEGY`, `'longest'` por defecto),
extrae su secuencia **ATMSEQ** (la realmente resuelta en los átomos, no
`SEQRES`) resolviendo residuos modificados vía el CCD (MSE→M, SEP→S, etc.), y
construye un mapeo de posiciones PDB↔FASTA. Si esa secuencia derivada trae
algún residuo no canónico sin mapeo (`X`), el pipeline **no aborta**: excluye
solo a BepiPred-3.0/EpiDope de esa corrida (con aviso claro) porque BepiPred
los rechaza en bloque, pero los motores estructurales corren igual sobre el
PDB.

## Flujo de trabajo (11 fases)

1. **Saneamiento FASTA / extracción de estructura** — Camino 1: valida y
   limpia la(s) secuencia(s) de entrada (`fasta_inputs/`, admite FASTA
   multi-registro). Caminos 2/3: Fase 1.5, ver arriba.
2. **Antigenicidad** — hasta 4 motores independientes ejecutados en local
   según el camino de entrada (ver tabla arriba), cada uno con su propio
   auto-caché en CSV: BepiPred-3.0 y EpiDope (motores de secuencia, FASTA →
   score por residuo) y DiscoTope-3.0 y ScanNet (motores estructurales, PDB
   de una sola cadena → score por residuo).
3. **Mapeo de epítopos y unión lógica anotada** — para cada motor activo, una
   ventana deslizante local (9 aa, tolerante a hasta 2 residuos por debajo
   del umbral por ventana, con fusión de ventanas solapadas/adyacentes;
   umbral independiente por motor, las escalas no son comparables entre sí)
   sobre sus scores de antigenicidad. Luego, TODA región detectada por
   cualquier motor activo avanza a la Fase 4 (`src/engines/consensus.py`):
   las regiones que solapan entre motores se **fusionan** (`start` mínimo,
   `end` máximo, sin recortar a la intersección, incluye fusión transitiva
   de cadenas de regiones), quedando marcadas en la columna `origen` con
   abreviaturas de 2 letras por motor contribuyente: `Bp` (BepiPred), `Ed`
   (EpiDope), `Dt` (DiscoTope-3.0), `Sn` (ScanNet). Un único motor se
   reporta solo (p. ej. `'Bp'`); dos o tres se unen con `'+'` (p. ej.
   `'Bp+Ed'`, `'Dt+Sn'`, `'Bp+Dt'` — cualquier combinación, no solo los
   pares "naturales", para poder distinguirlas todas sin ambigüedad).
   Única excepción: cuando los 4 motores contribuyen a la vez, la etiqueta
   es `'Consenso total'` en vez de `'Bp+Ed+Dt+Sn'`. Filtro de longitud
   inquebrantable: se descarta
   cualquier región final menor a 9 aa antes de la Fase 4.

   *Nota biológica:* DiscoTope-3.0/ScanNet puntúan epítopos conformacionales
   (parches 3D potencialmente discontinuos en la secuencia lineal);
   colapsarlos a regiones contiguas vía ventana deslizante es una
   simplificación deliberada para que Fase 4/5 sigan operando sobre péptidos
   lineales sintetizables.
3b. **Enmascarado transmembrana/péptido señal** — TMbed ejecutado en local
   (`src/engines/tmbed_engine.py`), sobre la secuencia COMPLETA de cada
   accession (no por péptido candidato, a diferencia de Fase 4b/4c: TMbed
   necesita el contexto completo de la proteína para predecir topología de
   membrana). Descarta de la unión anotada de Fase 3 cualquier región que
   caiga dentro de una hélice/tira transmembrana o del péptido señal
   N-terminal — esos residuos no son accesibles a anticuerpos en la
   proteína madura/anclada a membrana, así que proponerlos como epítopo
   B-cell no tiene sentido biológico. Reusa el mismo venv/pesos ya
   instalados para el plugin Scipion `scipion-chem-tmbed` (repo hermano,
   mismo encoder ProtT5-XL-U50 que StackGlyEmbed), sin importar código de
   ese plugin (depende de `pwchem`). Reporte propio:
   `<nombre>_tmbed_regions.csv` (regiones detectadas) y
   `<nombre>_union_epitopes_masked.csv` (unión post-enmascarado, insumo real
   de la Fase 4).
4. **Filtro de tolerancia** — BLASTp local contra el proteoma humano, con
   E-value seleccionado dinámicamente por longitud del péptido (laxo para
   péptidos cortos, estricto para dominios/proteínas completas), descarta
   péptidos con alta homología (`status = 'Autoinmunidad'`, umbral
   `BLAST_IDENTITY_THRESHOLD`, 75% por defecto). Los péptidos `'Segura'`
   resultantes alimentan, en paralelo y sin depender entre sí, **todas** las
   fases siguientes (4b, 4c, 5, 5b, 6) — ninguna de ellas descarta candidatos
   de otra.
4b. **Alergenicidad** — AlgPred 2.0 ejecutado en local
   (`src/engines/algpred_engine.py`) sobre cada péptido `'Segura'`.
   Puramente informativa: señal de seguridad de la secuencia en sí
   (potencial reacción tipo I mediada por IgE), no condiciona ninguna otra
   fase. Reporte propio: `<nombre>_alergenicidad_report.csv`.
4c. **N-glicosilación** — StackGlyEmbed ejecutado en local
   (`src/engines/stackglyembed_engine.py`), stack ProteinBERT + ESM-2 650M +
   ProtT5 ya entrenado. El repo original no trae un scanner de sitios
   candidatos: este pipeline escanea internamente cada péptido `'Segura'`
   buscando el sequon canónico N-X-[S/T] (X ≠ Prolina, incluyendo sequones
   solapados) y evalúa cada sitio encontrado; péptidos sin ningún sequon se
   omiten (no producen fila). Igual de informativa que 4b. Reporte propio:
   `<nombre>_glicosilacion_report.csv`.
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
5b. **Promiscuidad T-citotóxica (MHC-I)** — NetMHCpan-4.2 ejecutado en local
   (`src/engines/netmhcpan_engine.py`), paralela e independiente de la Fase
   5 (MHC-II): son vías de presentación antigénica biológicamente distintas
   (célula presentadora profesional vs. cualquier célula nucleada, CD8+
   citotóxico vs. CD4+ helper), nunca se fusionan en un único veredicto.
   Mismo patrón que Fase 5 (panel de referencia — `NETMHCPAN_REFERENCE_PANEL`,
   23 alelos HLA-A/B/C: 12 HLA-A/B de los supertipos de Sidney et al. 2008
   (ese paper no cubre HLA-C) + 11 HLA-C comunes globalmente (Rasmussen et
   al. 2014 + criterio de frecuencia poblacional ≥1% de IEDB, ver docstring
   de `netmhcpan_engine.py` para el detalle completo) —, enrutamiento
   por longitud, buffer overflow conocido del binario (re-verificado con el
   panel ampliado: el límite no cambia), con sus propios
   umbrales de %Rank (0.5/2.0, distintos de los de MHC-II: son escalas no
   comparables). **Anotación adicional de corte proteasomal C-terminal**: cada
   candidato aceptado se cruza con NetCleave (`src/engines/netcleave_engine.py`)
   para verificar si hay evidencia de corte EXACTO en el residuo
   inmediatamente posterior al núcleo de unión — un péptido puede bindear
   MHC-I fuerte y aun así nunca generarse vía procesamiento antigénico real
   si el proteasoma no corta ahí. Es una columna informativa
   (`netcleave_c_term_match`/`netcleave_c_term_score`), no un filtro: el
   veredicto de NetMHCpan sigue siendo el único criterio de aceptación.
   Reporte propio: `<nombre>_candidatos_finales_mhc1.csv`.
6. **Cruce con bnAb conocidos** — LANL HIV Molecular Immunology Database +
   CATNAP, ejecutado en local sobre CSVs ya descargados
   (`src/engines/lanl_catnap_engine.py`), sin ningún subprocess (pandas
   puro). Reemplaza a bNAber, cuyo dominio está muerto/parqueado. Cruza cada
   péptido `'Segura'` contra los epítopos lineales conocidos de anticuerpos
   ampliamente neutralizantes (matching por subcadena, umbral mínimo
   configurable — `LANL_CATNAP_MIN_OVERLAP`, 6 aa por defecto) y anexa
   potencia de neutralización (IC50, número de virus del panel) desde CATNAP
   cuando el nombre del anticuerpo coincide. Puramente informativa, no
   filtra nada. Solo produce matches reales para entradas de la familia HIV
   Env — un reporte vacío es el resultado esperado para cualquier otra
   proteína, no un fallo. No hace alineamiento a coordenadas HXB2 ni captura
   epítopos conformacionales (fuera de alcance de un cruce por secuencia).
   Reporte propio: `<nombre>_bnab_crossref.csv`.
7. **Ensamblaje automático del constructo multi-epítopo** —
   (`src/engines/construct_assembly.py`, lógica pura, sin subprocess).
   Selecciona los mejores `CONSTRUCT_TOP_N_PER_CLASS` candidatos (3 por
   defecto) de cada clase — **B-cell** (péptidos `'Segura'` que además son
   `Non-Allergen` en Fase 4b y sin ningún sequon `Glicosilado` en Fase 4c,
   rankeados por el mejor `{motor}_score` disponible), **HTL** (`'Candidato
   Válido'` de Fase 5, colapsados por `core_9aa`) y **CTL** (`'Candidato
   Válido'` de Fase 5b, colapsados por `core_9aa`, priorizando
   `netcleave_c_term_match == True`) — y los concatena con los linkers
   estándar del campo de diseño de vacunas multi-epítopo: `AAY` intra-CTL
   (sitio de corte del proteasoma), `GPGPG` intra-HTL e inter-bloque
   (espaciador universal, Livingston et al. 2002), `KK` intra-B-cell.
   **Orden de bloques: B-cell → HTL → CTL** (sin consenso fuerte en la
   literatura sobre el orden óptimo — los linkers ya garantizan liberación
   correcta por procesamiento antigénico independiente de la posición — se
   ancla en B-cell por ser el foco humoral original del proyecto). **Sin
   adjuvante** por decisión activa (requiere criterio biológico/estratégico
   específico del patógeno/huésped, fuera de scope de este pipeline); el
   parámetro opcional `adjuvant_sequence` de `assemble_construct` permite
   agregarlo más adelante sin rediseñar nada (linker rígido `EAAAK`, Arai
   et al. 2001). El "epítopo" insertado en los bloques HTL/CTL es
   `core_9aa` (el núcleo de unión real), no la ventana completa evaluada.
   Epítopos solapados de la MISMA clase ya se fusionan en Fase 3 (unión
   anotada); entre clases distintas deliberadamente NO se fusionan (rompería
   la semántica de los linkers). Genera `<nombre>_constructo.fasta` y
   `<nombre>_constructo_metadata.csv` (trazabilidad 100%: una fila por
   segmento —epítopo o linker—, con posición en el constructo, accession/
   posición de origen y el score que motivó la selección).
8. **Chequeo del constructo ensamblado** — 4 motores independientes
   corriendo sobre la secuencia COMPLETA del constructo (no por péptido
   individual, a diferencia de Fase 4b/4c):
   - **Alergenicidad**: AlgPred 2.0, reutilizado tal cual de la Fase 4b
     (sin instalación nueva).
   - **Toxicidad**: ToxinPred2 (`src/engines/toxinpred_engine.py`) — el
     propio grupo Raghava lo recomienda para proteínas de longitud
     completa (a diferencia de ToxinPred3.0, pensado para péptidos cortos).
   - **Antigenicidad intrínseca**: IApred (`src/engines/iapred_engine.py`)
     — reemplazo de VaxiJen (descartado: no es open-source ni tiene
     standalone/API local), único predictor open-source/local publicado
     específicamente para antigenicidad de secuencia completa (Miles
     et al. 2025).
   - **Péptido señal**: SignalP-6.0 (`src/engines/signalp_engine.py`) —
     confirma que el constructo NO tenga un péptido señal N-terminal
     predicho (esperable para un constructo de fusión sintético).
   Los 4 son informativos, ninguno filtra ni aborta el pipeline. Reporte
   combinado: `<nombre>_constructo_chequeo.csv`.

Todos los resultados intermedios y el reporte final se guardan en
`fasta_outputs/`.

### Checkpointing

Fases 3b/4/4b/4c/5/5b/6/7/8 se auto-cachean por hash de contenido de su input
(mismo mecanismo que ya usaba la Fase 2 para sus scores crudos): cada una
guarda un sidecar `<archivo>.inputhash` junto a su CSV final. Si relanzás
`pipeline.py` con el **mismo** `--input` y los **mismos** parámetros (umbral
de identidad, alelos extra, etc.), cada fase ya completada se detecta y se
salta ("`[Fase X] Checkpoint detectado...`" en la consola) en vez de
recomputar todo desde el principio — útil si una corrida larga se interrumpe
a mitad de camino (p. ej. por un OOM en una fase pesada como 4c/StackGlyEmbed,
que carga 3 modelos grandes). Cambiar cualquier parámetro que afecte el
resultado de una fase invalida su checkpoint automáticamente (y en cascada,
el de las fases que dependen de ella). El pico de memoria residente del
proceso se loggea después de las fases más pesadas (nivel `INFO`), para
diagnosticar en cuál ocurrió un OOM sin depender de herramientas externas.

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

**e) Filtro de cobertura mínima de alineamiento.** El e-value laxo de
`blastp-short` (pensado para no perder homólogos cortos reales) tiene un
efecto secundario: un fragmento minúsculo (5-6 aa) 100% idéntico dentro de
un péptido de 9-30 aa es estadísticamente esperable *por puro azar* contra
el proteoma humano completo (~11M residuos), no una homología real —
sin filtrar esto, casi cualquier péptido corto se rechazaba por
"Autoinmunidad" sin importar su origen. Un
hit solo cuenta hacia `max_pident` si su alineamiento cubre al menos
`BLAST_MIN_QUERY_COVERAGE` (90% por defecto) de la longitud del péptido
consultado.

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

### 6. DiscoTope-3.0 local (opcional, motor estructural 1/2 — Caminos 2/3)

A diferencia de BepiPred-3.0/NetMHCIIpan-4.3, **DiscoTope-3.0 es instalable
directo vía git+pip** (licencia Creative Commons, sin solicitud académica
separada): [github.com/Magnushhoie/DiscoTope-3.0](https://github.com/Magnushhoie/DiscoTope-3.0/).
Fija un stack propio (`torch`, `torch_geometric`, `xgboost`, `biotite`)
incompatible con el resto del pipeline: requiere entorno aislado dedicado,
mismo patrón que `.venv-bepipred`.

> **Nota (confirmado en este entorno):** `biotite==1.6.*` (pin exacto del
> `requirements.txt` oficial) no tiene wheel para Python < 3.11. Si tu
> `python3` por defecto es 3.10 o menor, crea el venv con un intérprete más
> nuevo (`python3.12`, etc.) — si tu sistema no tiene `python3.12-venv`
> instalado y no querés usar `sudo`, `pip install --user virtualenv` seguido
> de `python3 -m virtualenv -p python3.12 .venv-discotope` funciona sin
> privilegios de administrador.

```bash
git clone https://github.com/Magnushhoie/DiscoTope-3.0/
python3 -m venv .venv-discotope   # o virtualenv -p python3.12, ver nota arriba
.venv-discotope/bin/pip install -r DiscoTope-3.0/requirements.txt
.venv-discotope/bin/pip install ./DiscoTope-3.0

# Los pesos del ensemble XGBoost propio vienen en el repo, comprimidos:
cd DiscoTope-3.0 && python3 -c "import zipfile; zipfile.ZipFile('models.zip').extractall('.')" && cd ..
```

Variables de entorno (todas opcionales, con default razonable):
`DISCOTOPE_INSTALL_PATH` (default `DiscoTope-3.0/`), `DISCOTOPE_PYTHON_BIN`
(default `.venv-discotope/bin/python`), `DISCOTOPE_WEIGHTS_CACHE_DIR`
(cache persistente de los pesos de ESM-IF1 —descargados automáticamente en
la primera corrida, ~350 MB— fuera del repo por defecto:
`~/.cache/bcell-epitope-pipeline/discotope-weights`, para no volver a
descargarlos en cada entorno nuevo).

### 7. ScanNet local (opcional, motor estructural 2/2 — Caminos 2/3)

[github.com/jertubiana/ScanNet](https://github.com/jertubiana/ScanNet), sin
software externo más allá de su propio stack Python, pero uno **muy
antiguo**: Python 3.6.12 exacto, TensorFlow 1.14, Keras 2.2.5. Los pesos
pre-entrenados (~43 MB) ya vienen en el repo, no hace falta descargarlos
aparte.

> **Ningún sistema moderno trae ya Python 3.6.12 preinstalado.** La forma
> reproducible de conseguirlo (confirmada en este entorno) es con **conda**,
> no con un venv común (que necesita partir de un intérprete ya instalado):

```bash
git clone https://github.com/jertubiana/ScanNet
conda create -n scannet_env python=3.6.12 -y
conda run -n scannet_env pip install -r ScanNet/requirements.txt
```

Si instalaste el entorno conda con el nombre `scannet_env` (el usado arriba),
**no hace falta exportar nada**: `./run.sh` (ver "Uso" abajo) detecta ese
entorno automáticamente y usa el runtime `venv` en vez del default
`docker`. Si le pusiste otro nombre, o preferís invocar `pipeline.py`
directo sin pasar por `run.sh`:

```bash
export SCANNET_PYTHON_BIN=$HOME/miniconda3/envs/mi_entorno/bin/python
export SCANNET_RUNTIME=venv   # default es 'docker' (ver abajo)
```

**Alternativa Docker** (`SCANNET_RUNTIME=docker`, imagen oficial
`jertubiana/scannet`): evita resolver el stack antiguo a mano. **Validado**
(`docker pull jertubiana/scannet` + corrida real contra un PDB, resultados
idénticos byte a byte al runtime `venv`) — el `WORKDIR` de la imagen es
efectivamente `/ScanNet` (default de `SCANNET_DOCKER_WORKDIR`, confirmado
con `docker inspect`), y tanto `python` como `python3` resuelven al
intérprete correcto (3.6.12) dentro de la imagen. Es la alternativa mas
simple para quien no quiera lidiar con conda:

```bash
docker pull jertubiana/scannet
export SCANNET_RUNTIME=docker   # ya es el default; explicito solo si algo mas lo cambio
```

Variables de entorno: `SCANNET_RUNTIME` (`docker` default | `venv`),
`SCANNET_INSTALL_PATH` (default `ScanNet/`), `SCANNET_PYTHON_BIN`,
`SCANNET_DOCKER_IMAGE` (default `jertubiana/scannet`),
`SCANNET_DOCKER_WORKDIR`.

### 8. NetMHCpan-4.2 local (obligatorio para la Fase 5b)

Binario propietario de DTU Health Tech (predicción MHC-I / HLA-A/B/C), mismo
patrón de licencia académica y descarga manual que NetMHCIIpan-4.3.

**a) Descarga manual obligatoria.** Solicítalo en
`https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/` (sección
"Downloads", requiere cuenta académica).

**b) Instalación.**
```bash
tar -xvf netMHCpan-4.2.Linux.tar.gz
mv netMHCpan-4.2 /ruta/al/proyecto/netMHCpan-4.2
```
Edita la línea `NMHOME` al inicio del script `netMHCpan-4.2/netMHCpan` con la
ruta absoluta de instalación, igual que en NetMHCIIpan-4.3 (Sección 5b).

**c) Buffer overflow conocido.** El binario `netMHCpan` (modo péptido exacto,
`-p`) revienta con `*** buffer overflow detected ***` (exit code 0, el
wrapper `tcsh` no propaga el fallo) para péptidos > ~55 aa con el panel de 23
alelos por defecto (re-verificado al ampliar el panel de 12 a 23: el límite
no cambia, es una propiedad del largo del péptido, no de cuántos alelos se
pasan por `-a`) — el pipeline enruta automáticamente los péptidos largos a
modo proteína (ventana deslizante) para evitarlo, mismo mecanismo que
NetMHCIIpan-4.3.

Variables de entorno: `NETMHCPAN_HOME` (default `netMHCpan-4.2/`),
`NETMHCPAN_BINARY_NAME` (default `netMHCpan`).

### 9. AlgPred 2.0 local (obligatorio para la Fase 4b)

Código abierto (GPSR group, sin solicitud académica), instalable vía `pip`.
Requiere un venv aparte: fija su propio stack de dependencias.

```bash
python3 -m venv .venv-algpred
.venv-algpred/bin/pip install algpred2
```

Variables de entorno: `ALGPRED_PYTHON_BIN` (intérprete del venv),
`ALGPRED_SCRIPT_PATH` (ruta a `algpred2.py` dentro del paquete instalado —
normalmente `<venv>/lib/python3.X/site-packages/algpred2/python_scripts/algpred2.py`),
`ALGPRED_THRESHOLD` (umbral `ML_Score`, 0.3 por defecto, el mismo del propio
`algpred2.py`).

> **Bug conocido del script upstream** (verificado empíricamente): revienta
> con `ValueError: Expected 2D array, got 1D array` si el batch de entrada
> tiene EXACTAMENTE 1 secuencia. El wrapper de este pipeline lo evita
> duplicando la única secuencia antes de invocar el binario y descartando la
> fila extra del resultado — transparente, no requiere ninguna acción.

### 10. NetCleave local (obligatorio para la anotación de Fase 5b)

Código abierto (MIT, [github.com/APeriolo/NetCleave](https://github.com/APeriolo/NetCleave)),
predicción de sitios de corte proteasomal (MHC-I). Este pipeline usa
únicamente el modelo pre-entrenado bundled (`data/models/I_mass-spectrometry_HLA/`,
entrenado sobre IEDB/UniProt/UniParc) — **nunca reentrena en runtime**.

```bash
git clone https://github.com/APeriolo/NetCleave.git
python3 -m venv .venv-netcleave
.venv-netcleave/bin/pip install -r NetCleave/requirements.txt
.venv-netcleave/bin/pip install openpyxl  # pandas no trae soporte .xlsx por defecto
```

Variables de entorno: `NETCLEAVE_PYTHON_BIN`, `NETCLEAVE_SCRIPT_PATH` (ruta a
`NetCleave.py`).

### 11. StackGlyEmbed local (obligatorio para la Fase 4c)

Código abierto ([github.com/GaryChan-lab/StackGlyEmbed](https://github.com/GaryChan-lab/StackGlyEmbed)),
predicción de N-glicosilación via un stack ProteinBERT + ESM-2 650M + ProtT5
ya entrenado. El repo original espera posiciones candidatas escritas a mano
y llama a red en cada corrida (ESM-2 vía `torch.hub`, ProtT5 contra el ID
remoto del Hub) — este pipeline lo reemplaza por
`src/engines/stackglyembed_predict_local.py` (versionado en este mismo repo,
no dentro del clon externo: ver docstring del módulo), 100% offline una vez
cacheados los pesos.

**a) Clona el repo e instala su venv:**
```bash
git clone https://github.com/GaryChan-lab/StackGlyEmbed.git
python3 -m venv StackGlyEmbed/.venv-stackglyembed
StackGlyEmbed/.venv-stackglyembed/bin/pip install numpy pandas "tensorflow==2.14.*" \
    "tensorflow_addons==0.22.0" torch xgboost scikit-learn "transformers<5" h5py lxml pyfaidx
```
> `transformers>=5` exige `torch>=2.4`: si tu venv fija `torch==2.2.2` (como
> el de referencia), instalá una versión `4.x` de `transformers` o el
> backend de PyTorch queda deshabilitado en silencio (sin romper el import).

**b) Instala ProteinBERT** (no está en PyPI bajo ese nombre):
```bash
StackGlyEmbed/.venv-stackglyembed/bin/pip install --no-deps \
    "git+https://github.com/nadavbra/protein_bert.git"
```

**c) Descarga los pesos una única vez** (paso de SETUP, nunca en runtime):
```bash
StackGlyEmbed/.venv-stackglyembed/bin/python -c "
from proteinbert import load_pretrained_model
load_pretrained_model(validate_downloading=False)"  # ~183MB a ~/proteinbert_models/

HF_HUB_OFFLINE=0 StackGlyEmbed/.venv-stackglyembed/bin/python -c "
from transformers import AutoTokenizer, EsmModel
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"  # ~2.5GB al cache de HF Hub
```

**d) ProtT5**: si ya tenés los pesos de `Rostlab/prot_t5_xl_half_uniref50-enc`
descargados para otra herramienta (p. ej. TMbed), reusalos apuntando
`STACKGLYEMBED_T5_MODEL_PATH` a esa carpeta local — evita una descarga de
~3GB duplicada (mismo encoder, solo cambia precisión/empaquetado respecto al
`Rostlab/prot_t5_xl_uniref50` que pide el repo original).

Variables de entorno: `STACKGLYEMBED_PYTHON_BIN`, `STACKGLYEMBED_SCRIPT_PATH`
(default: `src/engines/stackglyembed_predict_local.py` de este repo, no
suele hacer falta tocarlo), `STACKGLYEMBED_MODELS_DIR` (carpeta `prediction/`
del clon externo, donde viven los pickles del clasificador ya entrenado),
`STACKGLYEMBED_T5_MODEL_PATH`, `STACKGLYEMBED_ESM_MODEL_NAME` (default
`facebook/esm2_t33_650M_UR50D`).

### 12. Datos LANL + CATNAP (obligatorio para la Fase 6)

Sin instalación de software: son CSVs/TSVs planos, consultados con pandas
puro, sin ningún subprocess ni llamada de red en runtime.

```bash
mkdir -p reference_db/lanl_immunology reference_db/catnap
# LANL HIV Molecular Immunology Database, tabla "Antibody":
# https://www.hiv.lanl.gov/components/sequence/HIV/asearch/query_one.comp?se_id=ab
# -> exportar como reference_db/lanl_immunology/ab_all.csv

# CATNAP (neutralización, antibodies):
# https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp
# -> exportar como reference_db/catnap/abs_<fecha>.txt
```

Variables de entorno: `LANL_AB_ALL_PATH`, `CATNAP_ABS_PATH`,
`LANL_CATNAP_MIN_OVERLAP` (umbral mínimo de solapamiento de subcadena, 6 aa
por defecto — para epítopos de referencia más cortos que el umbral, se exige
el match completo, nunca uno más laxo que el propio epítopo).

### 13. ToxinPred2 (obligatorio para la Fase 8, toxicidad del constructo)

Código abierto (Raghava group), instalable vía `pip`. El modelo (Random
Forest, ONNX) y el binario `blastp` que usa internamente vienen EMBEBIDOS en
el paquete — cero descarga aparte.

```bash
python3 -m venv .venv-toxinpred2
.venv-toxinpred2/bin/pip install toxinpred2
```

> **Nota (confirmado en este entorno):** creá el venv con **Python 3.10**,
> no con el `python3` por defecto del sistema si es 3.13+. El script
> empaquetado escribe un archivo intermedio con
> `to_csv(..., sep="\n")` — `pandas>=2` lo rechaza (`ValueError: bad
> delimiter value`), y no hay ningún flag de CLI que evite ese paso. Hace
> falta pinear versiones compatibles entre sí:
> ```bash
> .venv-toxinpred2/bin/pip install "pandas==1.5.3" "numpy<2"
> ```
> (pandas 1.5.3 está compilado contra la ABI de numpy<2; sin este segundo
> pin, el import de pandas revienta con
> `ValueError: numpy.dtype size changed...`).

Variables de entorno: `TOXINPRED2_PYTHON_BIN`, `TOXINPRED2_BINARY_NAME`
(default `toxinpred2`), `TOXINPRED2_THRESHOLD` (0.6 por defecto, el mismo
default del propio `toxinpred2`).

### 14. IApred (obligatorio para la Fase 8, antigenicidad intrínseca del constructo)

Código abierto ([github.com/sebamiles/IApred](https://github.com/sebamiles/IApred),
Miles et al. 2025), reemplazo de VaxiJen (descartado: no es open-source ni
tiene standalone/API local). SVM puro sobre features fisicoquímicas, sin
PyTorch/TensorFlow.

```bash
git clone https://github.com/sebamiles/IApred.git
python3 -m venv IApred/.venv-iapred
IApred/.venv-iapred/bin/pip install -r IApred/requirements.txt
```

> **Nota (confirmado en este entorno):** el `requirements.txt` del repo está
> incompleto — `functions.py` importa, sin declararlas en ningún lado,
> `imbalanced-learn`, `matplotlib` y `seaborn`. Instalalas a mano:
> ```bash
> IApred/.venv-iapred/bin/pip install imbalanced-learn matplotlib seaborn
> ```

Variables de entorno: `IAPRED_PYTHON_BIN`, `IAPRED_HOME` (raíz del clon —
`models_folder` dentro del script original es una ruta RELATIVA al cwd, así
que el wrapper siempre invoca el subproceso con `cwd=IAPRED_HOME`),
`IAPRED_SCRIPT_NAME` (default `IApred.py`).

### 15. SignalP-6.0 (obligatorio para la Fase 8, péptido señal del constructo)

Binario propietario de DTU Health Tech (licencia académica, mismo patrón que
BepiPred-3.0/NetMHCIIpan-4.3/NetMHCpan-4.2).

**a) Descarga manual obligatoria.** Solicitalo en
`https://services.healthtech.dtu.dk/services/SignalP-6.0/` (sección
"Downloads", requiere cuenta académica). Elegí el modo **`slow_sequential`**
(mismo footprint de RAM que `fast`, ~6x más lento, pensado para máquinas sin
GPU — el modo `slow` en paralelo requiere >14GB libres).

**b) Instalación.**
```bash
tar -xvf signalp-6.0i.slow_sequential.tar.gz
python3 -m venv .venv-signalp   # Python 3.10, ver nota abajo
.venv-signalp/bin/pip install ./signalp-6-package
.venv-signalp/bin/pip install "numpy<2"   # ABI, ver nota abajo
```

> **Notas (confirmadas en este entorno):**
> - Creá el venv con **Python 3.10**: el `requirements.txt` del paquete fija
>   `torch>1.7.0,<2`, sin wheel ya en el índice CPU-only oficial de PyTorch
>   para instalaciones modernas — se instala desde PyPI normal en su lugar,
>   que sí tiene wheels para Python 3.10.
> - `pip install` arrastra `numpy>=2` (vía `matplotlib`, sin pin superior en
>   `requirements.txt`), incompatible con la ABI de `torch==1.13` — mismo
>   tipo de bug que ToxinPred2 (`RuntimeError: Numpy is not available`).
> - **No hace falta seguir el paso 4 del `README.md` oficial** (copiar los
>   pesos dentro del paquete instalado): apuntá `SIGNALP_MODEL_DIR`
>   directo a la carpeta que ya contiene `sequential_models_signalp6/`
>   (el flag `--model_dir` del propio `signalp6` la usa tal cual) — evita
>   duplicar ~9.2GB.

Variables de entorno: `SIGNALP_PYTHON_BIN`, `SIGNALP_BINARY_NAME` (default
`signalp6`), `SIGNALP_MODEL_DIR`, `SIGNALP_ORGANISM` (`other` por defecto).

### 16. TMbed (obligatorio para la Fase 3b, enmascarado transmembrana/péptido señal)

Código abierto (Apache-2.0, Bernhofer & Rost 2022), pip-instalable — pero sus
pesos del encoder ProtT5-XL-U50 (~2.4 GB) no vienen bundled y normalmente se
descargan de HuggingFace en el primer uso, lo cual no está permitido por la
política local-only/no-scraping de este proyecto: ambas piezas se instalan
una sola vez, a mano.

Este pipeline **reusa el mismo venv/pesos** ya instalados para el plugin
Scipion hermano `scipion-chem-tmbed` (ver su `README.rst`), sin instalar
nada nuevo si ese plugin ya está configurado.

**a) Venv dedicado.**
```bash
python3 -m venv /path/to/venv-tmbed
/path/to/venv-tmbed/bin/pip install tmbed
```

**b) Pesos ProtT5-XL-U50**, descargados una sola vez (p. ej. desde una
máquina con acceso a internet, o vía
`huggingface-cli download Rostlab/prot_t5_xl_uniref50`) en una carpeta local
con `config.json`, `model.safetensors`, `spiece.model`,
`special_tokens_map.json` y `tokenizer_config.json`. **Mismo encoder que
reusa StackGlyEmbed** (Sección 11, `STACKGLYEMBED_T5_MODEL_PATH`): si ya
tenés esos pesos descargados para StackGlyEmbed, apuntá `TMBED_MODEL_DIR` a
la misma carpeta en vez de duplicar ~2.4 GB.

Variables de entorno: `TMBED_PYTHON_BIN`, `TMBED_BINARY_NAME` (default
`tmbed`), `TMBED_MODEL_DIR`, `TMBED_USE_GPU` (`0`/`1`, `0` por defecto —
máquina CPU-only), `TMBED_THREADS` (`4` por defecto), `TMBED_MIN_REGION_LENGTH`
(`1` por defecto, sin filtro de longitud mínima).

## Uso

`./run.sh` es un wrapper fino sobre `pipeline.py` (mismos argumentos) que
autodetecta las instalaciones locales de BepiPred-3.0 y ScanNet (venv/conda)
si están en su ubicación por defecto, para no tener que exportar nada a
mano. DiscoTope-3.0 nunca necesita exports: sus defaults ya son relativos a
la raíz del proyecto. Preferilo sobre invocar `python pipeline.py` directo
salvo que necesites apuntar a instalaciones en otras rutas (ahí sí usá las
variables de entorno de las Secciones 2-7).

```bash
# Coloca tu(s) FASTA en fasta_inputs/ (admite multi-registro), luego:
./run.sh --input fasta_inputs/secuencia.fasta

# Input de estructura (Caminos 2/3, requiere DiscoTope-3.0/ScanNet instalados,
# ver Secciones 6 y 7). El tipo de archivo se detecta automáticamente:
./run.sh --input fasta_inputs/estructura.pdb --pdb-mode structure_only
./run.sh --input fasta_inputs/estructura.pdb
# Sin --pdb-mode, se usa Settings.PDB_PROCESSING_MODE (default 'structure_and_sequence'),
# asi que la segunda linea de arriba ya corre los 4 motores sin necesitar el flag.

# El panel de 27 alelos HLA-DR/DQ/DP (IEDB_REFERENCE_PANEL) se evalúa siempre
# por defecto, sin necesidad de especificar nada. Para anexar alelo(s) extra
# (formato NetMHCIIpan, separados por coma SIN espacios; se valida el formato
# de inmediato, antes de correr cualquier fase):
./run.sh --input fasta_inputs/secuencia.fasta \
    --alelo-extra "DRB1_1602,HLA-DQA10501-DQB10201"

# Los umbrales/longitud mínima de la Fase 3 son independientes por motor
# (las escalas de score no son comparables entre sí):
./run.sh --input fasta_inputs/secuencia.fasta \
    --bepipred-threshold 0.1512 --bepipred-min-length 9 \
    --epidope-threshold 0.818 --epidope-min-length 9 \
    --discotope-threshold 0.90 --discotope-min-length 9 \
    --scannet-min-length 9
# ScanNet es distinto: por defecto (sin --scannet-threshold) usa un umbral
# ADAPTATIVO por percentil, calculado por accession (ScanNet no publica un
# umbral absoluto oficial, a diferencia de DiscoTope-3.0). Para forzar un
# valor fijo en vez del adaptativo:
./run.sh --input fasta_inputs/estructura.pdb --scannet-threshold 0.15
```

Corre `python pipeline.py --help` (o `./run.sh --help`) para ver todos los flags disponibles
(umbral/E-value de BLAST, carpeta de salida, etc.), todos con su valor por
defecto documentado.

### Salida en consola

Las tablas de la Fase 5 (y Fase 5b) resaltan en color el núcleo de unión de
9 aa dentro de la ventana de 15 aa evaluada (solo en terminal, no afecta el
CSV). Cuando el FASTA de entrada tiene varias proteínas (varios
`accession`), las tablas de Fase 2 (BepiPred-3.0/EpiDope/DiscoTope-3.0/
ScanNet), Fase 3b y Fase 6 separan cada proteína con una línea divisoria,
para no leerlas como una lista continua.

La Fase 3b, además de la tabla de regiones transmembrana/péptido señal
detectadas, imprime también qué regiones de la unión anotada se
descartaron por solaparse con ellas (`accession`/`start`/`end`/`tipo`),
cuando corresponde.

La Fase 6 corta la columna `Secuencia` cada 40 caracteres en vez de
estirar la tabla a lo ancho (un péptido HIV Env candidato puede medir 70+
aa); el tramo que matchea el epítopo bnAb se resalta en amarillo. Los
nombres de anticuerpo/dominio se acortan semánticamente (nunca una palabra
cortada a la mitad) y se limpia cualquier markup HTML crudo que traiga el
dato original de LANL (p. ej. `<sub>`).

La Fase 8 (chequeo del constructo) solo muestra score/veredicto/resumen
por motor, sin repetir la secuencia completa del constructo (ya visible en
la tabla de Fase 7).

### Archivos generados en `fasta_outputs/`

`<nombre>` es el nombre del archivo de entrada sin extensión (`--input`).

| Archivo | Fase | Contenido |
|---|---|---|
| `<nombre>_clean.fasta` | 1 | (Camino 1) FASTA saneado enviado a las fases siguientes |
| `<nombre>_derived.fasta` | 1.5 | (Caminos 2/3) FASTA canónico ATMSEQ derivado de la estructura |
| `<nombre>_chain_<cadena>.pdb` | 1.5 | (Caminos 2/3) PDB de una sola cadena (la elegida), input real de los motores estructurales |
| `<nombre>_position_mapping.csv` | 1.5 | (Caminos 2/3) Mapeo de posiciones PDB↔FASTA derivado |
| `<nombre>_bepipred_raw.csv` | 2 | Scores crudos por residuo de BepiPred-3.0 (caché) |
| `<nombre>_epidope_raw.csv` | 2 | Scores crudos por residuo de EpiDope (caché) |
| `<nombre>_discotope_raw.csv` | 2 | Scores crudos por residuo de DiscoTope-3.0 (caché) |
| `<nombre>_scannet_raw.csv` | 2 | Scores crudos por residuo de ScanNet (caché) |
| `<nombre>_bepipred_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (BepiPred-3.0) |
| `<nombre>_epidope_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (EpiDope) |
| `<nombre>_discotope_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (DiscoTope-3.0) |
| `<nombre>_scannet_epitopes.csv` | 3 | Regiones de epítopo mapeadas localmente (ScanNet) |
| `<nombre>_union_epitopes.csv` | 3 | Unión anotada de los motores activos (columna `origen`), entrada de la Fase 3b |
| `<nombre>_tmbed_raw.pred` | 3b | Salida cruda de TMbed (formato de 3 líneas por proteína), para trazabilidad |
| `<nombre>_tmbed_regions.csv` | 3b | Regiones transmembrana/péptido señal detectadas (`accession`/`start`/`end`/`type`) |
| `<nombre>_union_epitopes_masked.csv` | 3b | Unión anotada tras descartar regiones solapadas con TM/péptido señal, entrada real de la Fase 4 |
| `<nombre>_blast_report.csv` | 4 | Veredicto de tolerancia (Segura / Autoinmunidad) por región |
| `<nombre>_algpred_raw.csv` | 4b | Salida cruda de AlgPred 2.0, para trazabilidad |
| `<nombre>_alergenicidad_report.csv` | 4b | Veredicto de alergenicidad (Allergen / Non-Allergen) por péptido `'Segura'` |
| `<nombre>_stackglyembed_features.csv` | 4c | Features crudos (ProteinBERT+ESM-2+ProtT5) de StackGlyEmbed, para trazabilidad |
| `<nombre>_stackglyembed_raw.csv` | 4c | Predicción cruda de StackGlyEmbed (0/1 + probabilidad), para trazabilidad |
| `<nombre>_glicosilacion_report.csv` | 4c | Veredicto de N-glicosilación (Glicosilado / No glicosilado) por sequon candidato |
| `<nombre>_netmhciipan_raw_peptide_mode.xls` | 5 | Salida cruda de NetMHCIIpan-4.3, modo péptido exacto (`<= 40` aa), para trazabilidad. Solo se genera si hubo al menos un péptido en ese rango |
| `<nombre>_netmhciipan_raw_protein_mode.xls` | 5 | Salida cruda de NetMHCIIpan-4.3, modo proteína/ventana deslizante (`> 40` aa), para trazabilidad. Solo se genera si hubo al menos un fragmento en ese rango |
| `<nombre>_candidatos_finales.csv` | 5 | **Reporte final MHC-II** (ver formato abajo) |
| `<nombre>_netmhcpan_raw_peptide_mode.xls` / `_protein_mode.xls` | 5b | Salida cruda de NetMHCpan-4.2, mismo criterio que NetMHCIIpan |
| `<nombre>_netcleave_raw.csv` | 5b | Salida cruda de NetCleave (todas las ventanas de corte evaluadas, no solo las que matchean) |
| `<nombre>_candidatos_finales_mhc1.csv` | 5b | **Reporte final MHC-I**, con anotación `netcleave_c_term_match`/`netcleave_c_term_score` |
| `<nombre>_bnab_crossref.csv` | 6 | Cruce con epítopos de bnAb conocidos (vacío si la entrada no es HIV Env, es el resultado esperado) |
| `<nombre>_constructo.fasta` | 7 | Secuencia del constructo multi-epítopo ensamblado |
| `<nombre>_constructo_metadata.csv` | 7 | Trazabilidad 100%: una fila por segmento (epítopo o linker), posición en el constructo, accession/posición de origen, score que motivó la selección |
| `<nombre>_constructo_algpred_raw.csv` | 8 | Salida cruda de AlgPred2 sobre el constructo |
| `<nombre>_constructo_toxinpred_raw.csv` | 8 | Salida cruda de ToxinPred2 sobre el constructo |
| `<nombre>_constructo_iapred_raw.csv` | 8 | Salida cruda de IApred sobre el constructo |
| `<nombre>_constructo_signalp_raw.txt` | 8 | Salida cruda de SignalP-6.0 sobre el constructo |
| `<nombre>_constructo_chequeo.csv` | 8 | **Reporte combinado del constructo**: alergenicidad + toxicidad + antigenicidad intrínseca + péptido señal |
| `<archivo>.inputhash` | 3b/4/4b/4c/5/5b/6/7/8 | Sidecar de checkpointing (hash del input de esa fase), ver "Checkpointing" arriba |

### Formato de `<nombre>_candidatos_finales.csv`

Es la salida de la Fase 5 después del cruce con la Fase 3/4 (traceback) y la
deduplicación de ventanas redundantes: solo contiene candidatos con
`veredicto == 'Candidato Valido'`, ya enriquecidos con su región de origen.

| Columna | Significado |
|---|---|
| `accession` | Identificador de la proteína de origen (primer token de la cabecera FASTA) |
| `sequence_f5` | Péptido/ventana evaluado por NetMHCIIpan (péptido completo en modo exacto, o la ventana de 15 aa ganadora en modo proteína) |
| `core_9aa` | Núcleo de unión de 9 aa del alelo con mejor %Rank **en orientación normal** (los alelos invertidos nunca determinan este valor) |
| `start` / `end` | Coordenadas absolutas en la proteína de origen (1-indexado), recalculadas por traceback contra la región de la Fase 3/4 |
| `origen` | Motores de la Fase 3 que detectaron la región, abreviados (`Bp`/`Ed`/`Dt`/`Sn`) y unidos por `'+'` (p. ej. `'Bp'`, `'Bp+Ed'`, `'Dt+Sn'`, `'Bp+Dt'`); `'Consenso total'` si contribuyen los 4 |
| `n_alelos_promiscuos` | Cuántos alelos del panel (de 27, o más si usaste `--alelo-extra`) clasifican el péptido como SB/WB **en orientación normal** — este número decide el veredicto |
| `n_alelos_evaluados` | Tamaño total del panel evaluado |
| `min_rank_el` | Mejor (menor) %Rank entre los alelos en orientación normal |
| `<motor>_score` | Una columna por cada motor activo en esa corrida (p. ej. `bepipred_score`, `discotope_score`, ...): score medio de antigenicidad de la Fase 3 (`NaN` si ese motor no detectó esa región) |

`<nombre>_candidatos_finales_mhc1.csv` (Fase 5b) tiene el mismo formato
(análogo MHC-I: `NETMHCPAN_REFERENCE_PANEL` en vez del panel de MHC-II,
umbrales de %Rank propios), más dos columnas exclusivas:
`netcleave_c_term_match` (bool, ¿hay un corte proteasomal predicho EXACTO en
el residuo inmediatamente posterior al núcleo de unión?) y
`netcleave_c_term_score` (score crudo de ese corte, `NA` si no hubo match).

### Formato de `<nombre>_constructo_metadata.csv` y `<nombre>_constructo_chequeo.csv`

`_constructo_metadata.csv` (Fase 7) tiene una fila por SEGMENTO del
constructo (epítopo o linker, en el orden en que aparecen): `block`
(`'B-cell'`/`'HTL'`/`'CTL'`/`'Linker (...)'`/`'Adjuvante'`), `sequence`,
`start`/`end` (1-indexado, posición dentro del constructo — no de la
proteína de origen), `source_accession`/`source_start`/`source_end`
(`None` para segmentos de linker) y `source_score_note` (resumen legible
del score que motivó la selección de ese epítopo). Concatenar la columna
`sequence` en orden reconstruye exactamente `<nombre>_constructo.fasta`.

`_constructo_chequeo.csv` (Fase 8) tiene una única fila (el constructo
completo) con las columnas de los 4 motores combinadas: `algpred_score`/
`algpred_veredicto`, `toxinpred_score`/`toxinpred_veredicto`,
`iapred_score`/`iapred_categoria`, `signalp_prediction`/
`signalp_prob_other`/`signalp_prob_sp`/`signalp_cs_position`.

## Tests

La suite de tests (`tests/`, `pytest`, 219 tests) cubre la lógica pura de
cada fase — enrutamiento de input (FASTA vs. estructura), extracción de
estructura (residuos modificados, selección de cadena), unión anotada de N
motores en Fase 3, selección de task/E-value en Fase 4, parseo del `.xls` de
NetMHCIIpan y exclusión de alelos invertidos, traceback de coordenadas,
deduplicación de ventanas y validación de `--alelo-extra`/`--pdb-mode` — sin
depender de BepiPred, EpiDope, DiscoTope-3.0, ScanNet, BLAST+ ni NetMHCIIpan
instalados: no invoca ningún subprocess real (los tests de integración de
los 3 caminos mockean los 4 motores). Mismo criterio para las Fases
4b/4c/5b/6/8: `test_algpred_engine.py` (workaround de batch=1),
`test_netcleave_engine.py` (matching de corte C-terminal exacto),
`test_stackglyembed_engine.py` (scanner de sequones N-X-[S/T]),
`test_toxinpred_engine.py` (workaround de batch=1), `test_iapred_engine.py`,
`test_signalp_engine.py` (parseo de `prediction_results.txt`) mockean
`subprocess.run` para no depender de los venvs/modelos externos;
`test_lanl_catnap_engine.py` no necesita mockear nada (el motor nunca
invoca un subprocess, es pandas puro sobre CSVs). `test_construct_assembly.py`
cubre la Fase 7 completa (selección top-N por clase, dedup por `core_9aa`,
linkers, bloques vacíos, adjuvante opcional) con el invariante de
trazabilidad (`"".join(metadata_df['sequence']) == construct_sequence`)
verificado explícitamente.

```bash
pip install -r requirements-dev.txt
pytest
```

Estos tests unitarios validan la lógica de cada motor de forma aislada, no
que el pipeline completo funcione de punta a punta con los binarios/venvs
reales instalados — para eso, correr `pipeline.py` contra un input real (ver
"Uso" arriba) sigue siendo necesario. `STATUS.md` documenta la última
validación end-to-end real (corrida completa de las 11 fases contra
`fasta_inputs/GP120.fasta`, un HIV-1 Env real).
