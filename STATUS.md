# STATUS — extension del pipeline standalone (7 herramientas del scope de Carlos)

Ultima actualizacion: 2026-07-21, cierre de sesion por tiempo (no por crash).
Formato de tablas igual al usado para reconstruir el estado tras el OOM
anterior — mantener actualizado al final de cada sesion futura.

## Por que existe este documento

La sesion anterior a esta se corto por un OOM durante una corrida, obligando
a un `wsl --shutdown`. Esta sesion reconstruyo el estado completo desde cero
(Paso 0: git status/branch en 5 repos, activacion de venvs, verificacion de
binarios, re-corrida de tests Scipion) antes de tocar nada. Esta sesion
**tambien** se cierra por limite de tiempo del usuario, no por crash — pero
se deja este documento igual, para no depender de memoria de contexto en la
proxima.

## Tabla A — pipeline B-cell de 5 fases (ya portado a Scipion-Chem)

Sin cambios esta sesion. Confirmado intacto en el Paso 0 (git status/branch
limpios en los 5 repos, HEAD coincide, tests `tmbed`/`netmhciipan` de
Scipion vueltos a correr y pasan).

| Fase original | Destino real | Rama | Estado |
|---|---|---|---|
| Fase 2/3 BepiPred | `scipion-chem-bepipred` | `feat/gap-tolerant-window-mode` | Hecho, PR pendiente |
| Fase 2/3 EpiDope | `scipion-chem-fork` (= `Lvera-code/scipion-chem`) | `feat/epidope-protocol` | Hecho, PR pendiente |
| Fase 3 consenso | `ProtOperateSeqROI` generico | — | No requiere codigo |
| Fase 4 BLASTp | `scipion-chem-blast` | `feat/batch-roi-input` | Hecho, PR pendiente |
| Fase 5 NetMHCIIpan | `scipion-chem-netmhciipan` | `master` | Hecho y validado |
| — | `scipion-chem-bcellepitope` | `chore/migrate-to-upstream-plugins` | Deprecado, solo referencia |

## Tabla B — plugin Scipion `scipion-chem-tmbed`

Sin cambios esta sesion (construido y validado en la sesion anterior,
re-verificado en el Paso 0 de esta): `scipion3 test tmbed.tests...` pasa.

## Tabla C — extension del SCRIPT STANDALONE (objetivo de esta sesion)

Alcance: encadenar las 7 herramientas del scope de Carlos al `pipeline.py`
standalone (NO Scipion — esa integracion queda para una sesion aparte). Esta
tabla es el estado real verificado al cierre de esta sesion.

| Herramienta | Instalacion | Motor Python (`src/engines/`) | Wireado a `pipeline.py` | Notas |
|---|---|---|---|---|
| NetMHCpan-4.2 (MHC-I) | OK, movido a `B-Cell-Epitope-Prediction/netMHCpan-4.2/` | `netmhcpan_engine.py` — **completo, probado con datos reales y con la pipeline real** (`./run.sh --input OVA_test.fasta`) | **SI** — `fase_5b_tc_promiscuidad`, paralela a Fase 5 (MHC-II), NO fusionada (ver ADR) | Bug de buffer overflow del binario replicado y verificado empiricamente (55aa OK / 57aa crash con el panel de 12 alelos `NETMHCPAN_REFERENCE_PANEL`, exit code 0 silencioso). Sin columna `Inverted` (verificado, no asumido). |
| AlgPred 2.0 (alergenicidad) | OK, venv en `scipion-chem-algpred/.venv-algpred` | `algpred_engine.py` — **completo, probado con datos reales** | **NO** — motor listo, falta decidir en que fase engancha (¿filtro adicional en Fase 4? ¿anotacion informativa en Fase 3?) | Bug real del script upstream: revienta con `ValueError` si el batch tiene exactamente 1 secuencia (bug de reshape de sklearn en su propio codigo). Workaround: se duplica la secuencia y se descarta la fila extra — ya implementado y probado. |
| NetCleave (cleavage MHC-I/II) | OK, venv en `scipion-chem-netcleave/.venv-netcleave`, modelo pre-entrenado bundled (`data/models/I_mass-spectrometry_HLA/`, NO requiere reentrenar) | `netcleave_engine.py` — **completo, probado con datos reales** | **NO** — mismo motivo que AlgPred2 | Detalle no documentado verificado leyendo el codigo fuente: el .xlsx de salida se nombra `<stem>_<primer-token-del-header-fasta>_NetCleave.xlsx`, no `<stem>_NetCleave.xlsx` como asumiria uno por el `--help`. El wrapper usa glob, no el nombre exacto, para no depender de este detalle interno. Requirio instalar `openpyxl` en el entorno `cnb_pipeline` (pandas no trae soporte .xlsx por defecto). |
| StackGlyEmbed (N-glico) | Parcial. Repo clonado en `B-Cell-Epitope-Prediction/StackGlyEmbed/`, venv `.venv-stackglyembed` con torch/xgboost/sklearn/transformers/tensorflow instalados | **NO** | **NO** | Bug real encontrado y resuelto: `tensorflow_addons` (dependencia de ProteinBERT) esta descontinuado, solo compatible hasta TF 2.14 -- `pip install tensorflow` normal trae 2.21 e incompatibiliza. Resuelto pineando `tensorflow==2.14.*` + `tensorflow_addons==0.22.0` en este venv aislado (no afecta el TF 2.21 de NetCleave, venv separado). **Pendiente:** instalar `proteinbert` (NO esta en PyPI bajo ese nombre, hay que instalar desde `git+https://github.com/nadavbra/protein_bert.git`), localizar/descargar sus pesos pre-entrenados (Zenodo o GitHub, sin verificar aun), y sobre todo **parchear `extractFeatures.py`**: llama a `T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', ...)` y `torch.hub.load("facebookresearch/esm:main", ...)` que intentan red EN CADA CORRIDA si no se parchean a rutas locales (ProtT5 ya lo tenemos de TMbed, reusable; ESM-2 650M hay que descargarlo una vez y forzar carga offline despues). Clasificador final (KNN+SVM+XGB x10 folds + meta-SVM, ~500MB) SI esta bundled en el repo, no requiere descarga. |
| ToxDL 2.0 (toxicidad) | Repo clonado en `B-Cell-Epitope-Prediction/ToxDL2/`, sin venv | **NO** | **NO** | **Diferido explicitamente por el usuario** (2026-07-21): necesita dominios InterPro por proteina que el repo no calcula localmente (solo ejemplo con IDs pegados a mano); el usuario evaluara opciones (vector cero degradado / InterProScan local / consulta puntual a InterPro en setup) antes de decidir. Checkpoint del modelo SI esta bundled (~94MB: `ToxDL2_model.pth` + embeddings de dominio). Usa ESM-2 650M -- el MISMO checkpoint que StackGlyEmbed necesita, compartible si se retoma. `device='cuda:0'` hardcodeado en `parameters/test_000.py`, hay que parchearlo (esta maquina no tiene GPU). |
| LANL + CATNAP (bnAb cross-ref) | **Datos descargados**, `reference_db/catnap/` (12 archivos, 28MB) + `reference_db/lanl_immunology/` (6 archivos, 29MB) | **NO** | **NO** | bNAber (fuente original pedida) esta **muerta**: dominio parqueado/hijackeado (confirmado por Wayback Machine, ya asi desde antes de mayo 2025). Reemplazado por LANL HIV Molecular Immunology Database (`ab_all.csv`, 1790+ registros de anticuerpos, mismo origen que alimentaba a bNAber) + CATNAP (neutralizacion IC50/IC80, secuencias, germlines). Falta escribir el motor de consulta (pandas puro sobre estos CSVs locales, sin red) que cruce epitopos candidatos con `env_feature.txt` (anotaciones HXB2) y las secuencias de `ab_all.csv`/`virseqs_aa*.fasta`. |

## Decisiones de diseno tomadas esta sesion (para no repreguntar)

1. **ADR de 2026-07-12 (descartar MHC-I) REVERTIDO 2026-07-21** — decision
   explicita del usuario. MHC-I (NetMHCpan-4.2) vuelve al pipeline pero
   **como fase independiente en paralelo** (Fase 5b), nunca fusionado con el
   veredicto de T-helper/MHC-II de la Fase 5 original (son vias de
   presentacion antigenica distintas). Documentado en el docstring de
   `netmhciipan_engine.py` y `netmhcpan_engine.py`.
2. **bNAber reemplazado por LANL Immunology DB**, no por un mirror/snapshot
   de bNAber (no existe ninguno recuperable). Decision del usuario
   ("busca una alternativa que cumpla la misma funcion").
3. **ToxDL2 diferido**, no descartado. El usuario evaluara como resolver el
   problema de los dominios InterPro antes de retomarlo.
4. **NetMHCpan-4.2 vive dentro de `B-Cell-Epitope-Prediction/`** (movido
   desde `/home/enzo/software/`), por consistencia con el resto de
   instalaciones del pipeline standalone. AlgPred2/NetCleave/StackGlyEmbed
   NO se movieron (viven en sus repos `scipion-chem-*` hermanos): son venvs
   Python, mover un venv arrastra rutas absolutas embebidas en
   `pyvenv.cfg`/`bin/activate`/scripts de consola y puede romperlo; se
   referencian por ruta absoluta configurable en `Settings` en su lugar.

## Restriccion no negociable (recordada explicitamente por el usuario a mitad de sesion)

Todo debe correr local, nunca llamadas de red en tiempo de ejecucion del
pipeline. Instalacion/descarga de pesos es un paso de SETUP unico, no
runtime -- pero hay que verificar que ningun wrapper termine llamando a
`from_pretrained`/`torch.hub.load` sin fijar una ruta local + variables de
entorno offline (`HF_HUB_OFFLINE=1`, `TORCH_HOME` local). Pendiente
explicitamente para StackGlyEmbed (ver tabla arriba); TMbed ya lo hace bien
(usa `--model-dir` local siempre, nunca decide por si mismo).

## Siguiente sesion — orden sugerido (no vinculante)

1. Decidir donde enganchan AlgPred2/NetCleave en las fases del pipeline
   (ambos motores ya estan listos, es una decision de diseno, no de
   instalacion).
2. Terminar StackGlyEmbed: `pip install git+https://github.com/nadavbra/protein_bert.git`,
   descargar/verificar pesos de ProteinBERT, descargar ESM-2 650M una vez y
   parchear `extractFeatures.py` para carga 100% local/offline, escribir
   `src/engines/stackglyembed_engine.py` (necesita ademas un scanner de
   sequones N-X-[S/T] propio, el repo no lo trae).
3. Escribir el motor de consulta bnAb (`src/engines/lanl_catnap_engine.py`
   o similar) sobre los CSVs ya descargados.
4. Retomar ToxDL2 segun lo que decida el usuario sobre InterPro.
5. Orquestacion con manejo de memoria (batching, no cargar varios modelos
   pesados en simultaneo -- TMbed/StackGlyEmbed/ToxDL2 comparten ESM-2/ProtT5,
   posible reuso de un unico proceso "server" en vez de recargar el modelo
   por cada engine si el diseño lo permite) + logging de RAM + checkpointing,
   dado el OOM que origino todo esto.
6. Test end-to-end con PSMD7/PODXL/THBS2/SLC8A1 (y gp120 si da el tiempo).
