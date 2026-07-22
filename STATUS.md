# STATUS — pipeline standalone de descubrimiento de epitopos vacunales

Ultima actualizacion: 2026-07-22. Documento de estado limpio (no diario de
sesion): refleja el estado FINAL verificado, no el historial de idas y
vueltas para llegar ahi. Mantener actualizado al final de cada sesion
futura — si algo cambia, ACTUALIZAR la seccion correspondiente en vez de
agregar una nueva marcada "resuelto mas tarde".

## Tabla A — pipeline B-cell de 5 fases (ya portado a Scipion-Chem)

| Fase original | Destino real | Rama | Estado |
|---|---|---|---|
| Fase 2/3 BepiPred | `scipion-chem-bepipred` | `feat/gap-tolerant-window-mode` | Hecho, PR pendiente |
| Fase 2/3 EpiDope | `scipion-chem-fork` (= `Lvera-code/scipion-chem`) | `feat/epidope-protocol` | Hecho, PR pendiente |
| Fase 3 consenso | `ProtOperateSeqROI` generico | — | No requiere codigo |
| Fase 4 BLASTp | `scipion-chem-blast` | `feat/batch-roi-input` | Hecho, PR pendiente |
| Fase 5 NetMHCIIpan | `scipion-chem-netmhciipan` | `master` | Hecho y validado |
| — | `scipion-chem-bcellepitope` | `chore/migrate-to-upstream-plugins` | Deprecado, solo referencia |

## Tabla B — plugin Scipion `scipion-chem-tmbed`

Construido y validado: `scipion3 test tmbed.tests...` pasa.

## Tabla C — herramientas wireadas al SCRIPT STANDALONE (`pipeline.py`)

Estado real verificado. Todas corren 100% local (subprocess sobre un venv
dedicado o binario instalado), sin ninguna llamada de red en tiempo de
ejecucion.

| Herramienta | Instalacion | Motor Python (`src/engines/`) | Fase | Notas |
|---|---|---|---|---|
| NetMHCpan-4.2 (MHC-I) | `B-Cell-Epitope-Prediction/netMHCpan-4.2/` | `netmhcpan_engine.py` | 5b, paralela a Fase 5 (MHC-II), NO fusionada (vias de presentacion antigenica distintas) | Buffer overflow del binario en modo peptido exacto para entradas >55aa con el panel de 12 alelos (verificado empiricamente, exit code 0 silencioso) -- enrutado automaticamente a modo proteina para evitarlo. Sin columna `Inverted` (a diferencia de NetMHCIIpan, verificado, no asumido). |
| AlgPred 2.0 (alergenicidad) | venv en `scipion-chem-algpred/.venv-algpred` | `algpred_engine.py` | 4b (per-peptido) y reusado en 8 (constructo completo) | Bug real del script upstream: revienta con `ValueError` si el batch tiene exactamente 1 secuencia (bug de reshape de sklearn). Workaround: se duplica la secuencia y se descarta la fila extra. En Fase 8 este es el camino NORMAL (siempre 1 secuencia por corrida), no un caso de borde. |
| NetCleave (cleavage MHC-I) | venv en `scipion-chem-netcleave/.venv-netcleave`, modelo pre-entrenado bundled | `netcleave_engine.py` | Anotacion dentro del reporte de Fase 5b | Verifica si hay un corte proteasomal EXACTO en el residuo inmediatamente posterior al candidato aceptado por NetMHCpan (no solo "hay algun corte en la region"). Señal complementaria, no filtro. El .xlsx de salida se nombra `<stem>_<primer-token-del-header-fasta>_NetCleave.xlsx`; el wrapper usa glob, no el nombre exacto. |
| StackGlyEmbed (N-glicosilacion) | Repo clonado en `StackGlyEmbed/` (venv `.venv-stackglyembed`), `protein_bert` instalado `--no-deps`, ProteinBERT/ESM-2 650M/ProtT5 cacheados localmente | `stackglyembed_engine.py` (scanner de secuones propio) + `src/engines/stackglyembed_predict_local.py` (extraccion+prediccion, reemplaza los scripts originales que llamaban a red) | 4c (per-peptido) | `StackGlyEmbed/` es un repo git anidado (su propio `.git`): git NO permite des-ignorar un archivo dentro de un repo anidado con ningun patron de `.gitignore` -- por eso `stackglyembed_predict_local.py` vive en `src/engines/` (arbol versionado normal), no dentro del clon. ESM-2 vía `transformers.EsmModel` (offline real) en vez de `torch.hub.load(...)` del script original (pega red siempre). ProtT5 REUSA los pesos de TMbed (`Rostlab/prot_t5_xl_half_uniref50-enc`, mismo encoder). |
| LANL Immunology DB + CATNAP (bnAb cross-ref) | CSVs locales en `reference_db/` | `lanl_catnap_engine.py` (pandas puro, sin subprocess) | 6, informativa (solo relevante para HIV Env) | Reemplaza a bNAber (dominio muerto/parqueado). Cruce de subcadena (longest-common-substring) contra los 771 epitopos lineales de `ab_all.csv` con epitopo reportable (de 3799 registros totales; el resto son conformacionales, fuera de alcance). Umbral configurable `LANL_CATNAP_MIN_OVERLAP` (6 aa default). Validado con bnAbs reales (10E8, 2F5, Z13e1, m66) con IC50 real cruzado desde CATNAP. |
| Ensamblaje de constructo | N/A (logica pura) | `construct_assembly.py` | 7 | Ver Tabla D. |
| ToxinPred2 (toxicidad del constructo) | `pip install toxinpred2` en venv Python 3.10 dedicado (`.venv-toxinpred2/`) | `toxinpred_engine.py` | 8 | Modelo ONNX + blastp + base MERCI EMBEBIDOS en el wheel, cero descarga aparte. Venv Python 3.10 + `pandas==1.5.3` + `numpy<2` pineados (el script empaquetado usa `to_csv(sep="\n")`, que pandas>=2 rechaza; ABI de numpy>=2 rompe pandas 1.5.3). Mismo bug de batch=1 que AlgPred2. |
| IApred (antigenicidad intrinseca del constructo) | `git clone github.com/sebamiles/IApred` + venv propio (`IApred/.venv-iapred/`) | `iapred_engine.py` | 8 | Reemplaza a VaxiJen (no open-source, sin standalone/API local). SVM puro sobre features fisicoquimicas. `requirements.txt` del repo esta incompleto (faltan `imbalanced-learn`/`matplotlib`/`seaborn`, instalados a mano). `models_folder` es ruta relativa al cwd: subprocess siempre con `cwd=IAPRED_HOME`. |
| SignalP-6.0 (peptido señal del constructo) | Paquete DTU Health Tech (licencia academica), copiado a `signalp-6.0/` (9.2GB) + venv Python 3.10 dedicado (`.venv-signalp/`) | `signalp_engine.py` | 8 | Modo `slow-sequential` (mismo RAM que `fast`, ~6x mas lento, para CPU sin GPU). Pesos referenciados por `--model_dir` directo, sin duplicar. Venv Python 3.10 + `torch>1.7,<2` + `numpy<2`. Bug de parseo real: `prediction_results.txt` trae 2 lineas de comentario `#`, no 1 -- se usa `comment="#"` en `pd.read_csv`, no un `skiprows` fijo. |

## Tabla D — Fase 7 (ensamblaje de constructo) + Fase 8 (chequeo del constructo)

Resuelve el pedido original de Carlos: alergenicidad/toxicidad/antigenicidad
evaluadas sobre el CONSTRUCTO MULTI-EPITOPO FINAL ensamblado, no por
peptido individual (eso ya lo cubre Fase 4b/4c, insuficiente por si solo
para este pedido).

**Fase 7 — ensamblaje (`construct_assembly.py`, logica pura, sin subprocess):**

- Selecciona **top-3 candidatos por clase** (`Settings.CONSTRUCT_TOP_N_PER_CLASS`,
  configurable por variable de entorno, no expuesto como flag de CLI):
  - **B-cell**: de `safe_df` (Fase 4 'Segura'), excluye `Allergen` (Fase 4b)
    y cualquier peptido con >=1 sequon `Glicosilado` (Fase 4c); rankea por
    el mejor `{motor}_score` disponible.
  - **HTL/CTL**: de los `'Candidato Valido'` de Fase 5/5b, colapsa por
    `core_9aa` (misma logica que la deduplicacion de ventanas de
    NetMHCIIpan/NetMHCpan) quedandose con la mejor fila; CTL ademas
    prioriza `netcleave_c_term_match == True` antes que promiscuidad/%Rank.
- **Linkers** (convencion estandar del campo, no regla biologica fija):
  `AAY` intra-CTL (sitio de corte del proteasoma), `GPGPG` intra-HTL e
  inter-bloque (espaciador universal, Livingston et al. 2002), `KK`
  intra-B-cell.
- **Orden de bloques: B-cell → HTL → CTL.** Decision final del usuario
  (2026-07-22): sin consenso fuerte en la literatura sobre orden optimo (los
  linkers ya garantizan liberacion correcta por procesamiento, independiente
  de la posicion); se ancla en B-cell por ser el foco humoral original del
  proyecto (bnAb/HIV).
- **Sin adjuvante.** Decision activa del usuario de NO incluir uno en esta
  version (la eleccion de adjuvante -beta-defensina, PADRE, flagelina,
  L7/L12, etc.- requiere criterio biologico/estrategico especifico del
  patogeno/huesped, fuera de scope). Hook de diseño ya implementado
  (`adjuvant_sequence` en `assemble_construct`, con linker rigido EAAAK) para
  agregarlo sin rediseñar si se decide mas adelante.
- **Sin fusion de epitopos solapados entre clases.** Decision final del
  usuario: fusionar epitopos de clases distintas (p. ej. un B-cell que se
  solapa en posicion con un HTL) romperia la semantica de los linkers -cada
  bloque espera un peptido de ESA clase, no un hibrido-. La fusion
  INTRA-clase ya la resuelve la Fase 3 (union de regiones solapadas del
  mismo tipo de motor, antes de que las clases se separen).
- Metadata 100% trazable (`<input_stem>_constructo_metadata.csv`): una fila
  por segmento (epitopo o linker), con posicion en el constructo, accession/
  posicion de origen, y el score que motivo la seleccion. Invariante
  verificado en tests y en corridas reales:
  `"".join(metadata_df['sequence']) == construct_sequence`.

**Fase 8 — chequeo (4 motores, ver Tabla C):** AlgPred2 (reusado, sin
instalacion nueva), ToxinPred2, IApred, SignalP-6.0. Los 4 son informativos
(ninguno filtra ni aborta el pipeline).

**Nota historica:** ToxDL2 fue evaluado como alternativa de toxicidad y
**descartado definitivamente** (necesitaba dominios InterPro que el repo no
calcula localmente, sin un paso automatizado viable sin llamadas de red).
ToxinPred2 cubre el chequeo de toxicidad del constructo sin ese problema.
El clon (`ToxDL2/`) fue eliminado del proyecto; no queda ninguna accion
pendiente relacionada.

## Validacion realizada

**End-to-end con datos reales** (`fasta_inputs/GP120.fasta`, HIV-1 Env real,
861 aa — elegido porque contiene sequones N-glico reales y epitopos de bnAb
conocidos, para forzar resultados biologicamente sensatos en vez de
sinteticos):

| Fase | Resultado real |
|---|---|
| 1 (saneamiento) | 1 registro, 861 aa, OK |
| 2 (BepiPred+EpiDope) | ambos corren |
| 3 (union) | 10 regiones (9 Ed, 1 Bp+Ed) |
| 4 (BLASTp) | 10/10 'Segura' |
| 4b (AlgPred2) | 5 Allergen / 5 Non-Allergen |
| 4c (StackGlyEmbed) | 10 sequones evaluados, 6/10 'Glicosilado' |
| 5 (NetMHCIIpan, MHC-II) | 20/108 candidatos promiscuos |
| 5b (NetMHCpan, MHC-I) + NetCleave | 7/452 candidatos promiscuos, 7/7 con corte C-terminal confirmado |
| 6 (bnAb cross-ref) | 8/10 peptidos coinciden, 3 con neutralizante confirmado (2F5/Z13e1/m66, MPER de gp41 -- biologicamente correcto) |
| 7 (ensamblaje) | Constructo de 127 aa (3 B-cell + 3 HTL + 3 CTL) |
| 8 (chequeo constructo) | Non-Allergen, Non-Toxin, antigenicidad intrinseca "Low" (IApred), sin peptido señal |

**Camino de estructura** (`fasta_inputs/7c4s.pdb`, modo `structure_and_sequence`,
los 4 motores de Fase 2 a la vez): Fase 3 produjo regiones con origenes
mixtos (`Bp+Ed+Sn`, `Ed+Dt`, confirma que la union de 4 motores simultaneos
funciona), y las Fases 4b-8 corrieron sin fallar sobre los candidatos
resultantes. PIPELINE COMPLETADO sin errores en ambos caminos.

**Checkpointing** (Fase 4/4b/4c/5/5b/6/7/8, auto-cache por hash de contenido
del input de cada fase): verificado con corridas de 2 pasadas — segunda
pasada instantanea (38s -> 0.4s en un caso real), y que cambiar un
parametro invalida el checkpoint en cascada correctamente.

**Suite de tests** (`pytest tests/`): 201 tests, sin depender de ningun
venv/binario externo instalado (logica pura + `subprocess.run` mockeado).

## Restriccion no negociable

Todo debe correr local, nunca llamadas de red en tiempo de ejecucion del
pipeline. Instalacion/descarga de pesos es un paso de SETUP unico, no
runtime. Cada wrapper nuevo se audita para que nunca dispare `requests`,
`from_pretrained`/`torch.hub.load` sin ruta local, ni equivalente.

## Decisiones de diseño vigentes (para no repreguntar)

- **MHC-I (NetMHCpan-4.2) es una fase independiente en paralelo a MHC-II**
  (Fase 5b vs. Fase 5), nunca fusionadas: son vias de presentacion
  antigenica distintas.
- **bNAber reemplazado por LANL Immunology DB** (no existe un mirror
  recuperable de bNAber).
- **AlgPred2 = Fase 4b (per-peptido, paralela a 5/5b), NetCleave = anotacion
  dentro de Fase 5b, StackGlyEmbed = Fase 4c (per-peptido), bnAb cross-ref =
  Fase 6 (informativa)**: alergenicidad/N-glicosilacion son propiedades de
  la secuencia en si, no atadas a una via de presentacion; el corte
  proteasomal SI esta mecanicamente atado a MHC-I.
- **Fase 4b (per-peptido) NO sustituye el chequeo a nivel de constructo**:
  son preguntas distintas ("es seguro este peptido candidato" vs. "es
  seguro el constructo final ensamblado"). Resuelto con Fase 7/8, ver
  Tabla D.
- **NetMHCpan-4.2 vive dentro de `B-Cell-Epitope-Prediction/`**; AlgPred2/
  NetCleave/StackGlyEmbed NO se movieron (viven en sus repos hermanos):
  mover un venv arrastra rutas absolutas embebidas y puede romperlo — se
  referencian por ruta absoluta configurable en `Settings`.
- **StackGlyEmbed/IApred son repos git anidados** (su propio `.git`): git no
  permite des-ignorar un archivo dentro de un repo anidado con ningun
  patron de `.gitignore`. Si hace falta escribir codigo de integracion
  propio (como con StackGlyEmbed), ese codigo vive en `src/engines/`, nunca
  dentro del clon. Si el CLI original ya sirve tal cual sin modificarlo
  (como con IApred), no hay problema en ignorar el repo entero.
- **ToxDL2 evaluado y descartado** (ver Tabla D): ToxinPred2 cubre el
  chequeo de toxicidad del constructo sin el problema de dominios InterPro
  que bloqueaba a ToxDL2.
- **VaxiJen descartado** (no open-source, sin standalone/API local),
  reemplazado por IApred.

## Testing exhaustivo de robustez (2026-07-22)

Pedido explicito del usuario: confirmar que el pipeline es "irrompible" con
casos extremos reales, no solo el camino feliz. Corridas/pruebas realizadas
esta ronda (mas alla de la validacion end-to-end ya documentada arriba):

- **Camino de estructura con Fase 7/8** (no probado hasta ahora): `7c4s.pdb`
  produjo 2 candidatos B-cell y CERO HTL/CTL -- el constructo se ensamblo
  correctamente con un unico bloque (38 aa), sin bloques vacios colgando ni
  linkers huerfanos. Fase 8 corrio sin fallar sobre ese constructo corto.
- **Constructo completamente vacio** (via `--identity-threshold 1`, fuerza
  0 candidatos 'Segura'): Fase 7/8 lo manejan limpiamente, `PIPELINE
  COMPLETADO` sin error.
- **FASTA multi-registro** (`MonkeyPoxSequences.fasta`, 6 proteinas):
  constructo de 314 aa ensamblado correctamente cruzando candidatos de 3
  accessions distintas (`WEN68160.1`, `AGR38652.1`, `AGR38316.1`),
  trazabilidad verificada exacta.
- **Errores de input** (archivo inexistente, FASTA vacio, FASTA sin
  cabecera `>`): los 3 casos terminan con mensaje de error claro y
  `exit code 1`, nunca una traza cruda sin manejar.
- **Secuencias extremas directas contra los 4 motores de Fase 8** (AlgPred2,
  ToxinPred2, IApred, SignalP-6.0): 1 aa, 2-3 aa, homopolimero (30x 'A'),
  secuencia de 1000 aa. Sin crashes en ningun caso.

**Bug real encontrado y corregido en esta ronda:** IApred exige un MINIMO
de 20 aa (verificado leyendo `IApred.py`) -- para secuencias mas cortas
escribe el texto literal `'Sequence too short'` en la columna de score (no
un numero). `iapred_engine.py` no lo manejaba: `print_iapred_report`
formatea el score con `:.4f`, que revienta con `TypeError` sobre un string.
Esto es un escenario REAL, no hipotetico: un constructo con un unico
B-cell candidato corto (Fase 3 permite regiones desde 9 aa) cae bajo ese
umbral -- confirmado reproduciendolo con `fase_8_chequeo_constructo`
directamente sobre un constructo real de 15 aa. Corregido: el score se
coacciona a `NaN` explicito (`pd.to_numeric(errors='coerce')`, nunca un
string mezclado en la columna) con una categoria informativa
(`'No evaluado (secuencia < 20 aa)'`), y el formateador de tabla tolera
`NaN`. 3 tests de regresion agregados (`test_iapred_engine.py`).

Suite completa tras esta ronda: **201 tests**, sin regresiones.

## Auditoria de Scipion-readiness (2026-07-22)

Decision de secuenciacion vigente: standalone-script-first, integracion a
Scipion en una sesion aparte (ver Tabla A/B para lo que YA esta portado de
la version anterior del pipeline). Esta sesion NO escribio protocolos
Scipion nuevos -- eso sigue fuera de alcance -- pero se audito que la
arquitectura actual no introduzca nada que complique esa migracion futura:

- **Sin `argparse` dentro de `src/engines/`**: el unico modulo con
  `argparse` es `stackglyembed_predict_local.py`, que es un SCRIPT
  standalone invocado por subprocess (nunca importado como modulo Python),
  exactamente igual que los CLIs de AlgPred2/NetCleave/ToxinPred2/IApred/
  SignalP-6.0 -- no es logica de orquestacion mezclada con un motor.
- **Sin `input()` interactivo** en ningun motor (confirmado con grep) --
  el unico punto que en teoria podia pedir confirmacion interactiva
  (`ProteinBERT.load_pretrained_model`) se invoca siempre con
  `download_model_dump_if_not_exists=False`.
- **Sin estado global mutable** a nivel de modulo en ningun engine (solo
  constantes: `_OUTPUT_COLUMNS`, paneles de referencia, etc.) -- corridas
  concurrentes/paralelas de protocolos no compartirian estado por accidente.
- **I/O de archivos confinado a los parametros explicitos** (`output_dir`,
  directorios temporales): ningun motor escribe fuera de esas rutas.
- **Retornos siempre DataFrames tipados con columnas documentadas** — base
  natural para construir Sets/Objects de Scipion.
- **Errores via jerarquia tipada** (`EngineExecutionError` y subclases,
  `src/utils/exceptions.py`), no excepciones genericas -- un protocolo
  Scipion puede capturarlas y reportarlas limpio en la GUI.
- **Separacion `predict_*` (logica) vs. `print_*_report` (solo consola)**:
  un protocolo Scipion llamaria unicamente a las primeras.

Conclusion: no hace falta ningun refactor previo a portar estos motores a
protocolos Scipion cuando llegue esa sesion.

## Siguiente sesion

No queda ningun item de scope bloqueado o pendiente de decision del pipeline
standalone. Lo unico fuera de alcance de este documento:

1. **Integracion a Scipion**: decision de secuenciacion tomada explicitamente
   por el usuario — standalone-script-first, Scipion-integration despues, en
   una sesion aparte (ver Tabla A/B para lo que YA esta portado, y
   "Auditoria de Scipion-readiness" arriba).
2. Test individual de PSMD7/PODXL/THBS2/SLC8A1 (proteinas humanas de prueba,
   camino FASTA) — no bloqueante, mismo camino de codigo ya confirmado con
   GP120.
