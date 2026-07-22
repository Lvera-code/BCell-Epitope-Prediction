# STATUS — extension del pipeline standalone (7 herramientas del scope de Carlos)

Ultima actualizacion: 2026-07-22. Formato de tablas igual al usado para
reconstruir el estado tras el OOM original — mantener actualizado al final
de cada sesion futura.

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
| NetMHCpan-4.2 (MHC-I) | OK, movido a `B-Cell-Epitope-Prediction/netMHCpan-4.2/` | `netmhcpan_engine.py` — **completo, probado con datos reales y con la pipeline real** (`./run.sh --input OVA_test.fasta`) | **SI** — `fase_5b_tc_promiscuidad`, paralela a Fase 5 (MHC-II), NO fusionada (ver ADR). Ademas anotada con `netcleave_engine.annotate_cterm_cleavage` (ver fila NetCleave). | Bug de buffer overflow del binario replicado y verificado empiricamente (55aa OK / 57aa crash con el panel de 12 alelos `NETMHCPAN_REFERENCE_PANEL`, exit code 0 silencioso). Sin columna `Inverted` (verificado, no asumido). |
| AlgPred 2.0 (alergenicidad) | OK, venv en `scipion-chem-algpred/.venv-algpred` | `algpred_engine.py` — **completo, probado con datos reales** | **SI (2026-07-22)** — `fase_4b_alergenicidad`, corre sobre el `safe_df` de Fase 4 en paralelo a Fase 5/5b (senal de seguridad, no atada a ninguna via de presentacion), CSV propio `<input_stem>_alergenicidad_report.csv` | Bug real del script upstream: revienta con `ValueError` si el batch tiene exactamente 1 secuencia (bug de reshape de sklearn en su propio codigo). Workaround: se duplica la secuencia y se descarta la fila extra — ya implementado y probado. |
| NetCleave (cleavage MHC-I/II) | OK, venv en `scipion-chem-netcleave/.venv-netcleave`, modelo pre-entrenado bundled (`data/models/I_mass-spectrometry_HLA/`, NO requiere reentrenar) | `netcleave_engine.py` — **completo, probado con datos reales** | **SI (2026-07-22)** — anotacion `netcleave_c_term_match`/`netcleave_c_term_score` dentro del reporte de Fase 5b (MHC-I): corre sobre los mismos peptidos del `safe_df`, verifica si NetCleave predice un corte proteasomal EXACTO en el residuo inmediatamente posterior al candidato aceptado por NetMHCpan (no solo "hay algun corte en la region"). Senal complementaria, no filtro -- el veredicto de NetMHCpan sigue siendo el unico criterio de 'Candidato Valido'. | Detalle no documentado verificado leyendo el codigo fuente: el .xlsx de salida se nombra `<stem>_<primer-token-del-header-fasta>_NetCleave.xlsx`, no `<stem>_NetCleave.xlsx` como asumiria uno por el `--help`. El wrapper usa glob, no el nombre exacto, para no depender de este detalle interno. Requirio instalar `openpyxl` en el entorno `cnb_pipeline` (pandas no trae soporte .xlsx por defecto). |
| StackGlyEmbed (N-glico) | OK. Repo clonado en `B-Cell-Epitope-Prediction/StackGlyEmbed/` (venv `.venv-stackglyembed`), `protein_bert` instalado (`--no-deps`, para no romper el pineo de TF), dump de ProteinBERT descargado (`~/proteinbert_models/default.pkl`, 183MB), ESM-2 650M descargado via HF Hub (2.5GB, offline confirmado con `HF_HUB_OFFLINE=1`) | `stackglyembed_engine.py` (scanner de secuones N-X-[S/T] propio + subprocess) + `src/engines/stackglyembed_predict_local.py` (extraccion de features + prediccion, reemplaza `extractFeatures.py`+`predict.py` originales) — **completo, probado end-to-end con secuencias sinteticas (positivas/negativas/solapadas), 100% offline confirmado** | **NO** — motor listo, falta decidir en que fase engancha (mismo tipo de decision pendiente que tuvieron AlgPred2/NetCleave hasta hoy) | `transformers` bajado de 5.14.1 a 4.46.3 en este venv: la version que trae `protein_bert` como dependencia transitiva exige `torch>=2.4`, incompatible con el `torch==2.2.2` pineado por StackGlyEmbed (deshabilitaba silenciosamente el backend de PyTorch, sin romper el import). ESM-2 se carga via `transformers.EsmModel` (offline-friendly) en vez de `torch.hub.load(...)` del script original (pega red SIEMPRE, no solo la primera vez). ProtT5 REUSA los pesos ya descargados para TMbed (`scipion-chem-tmbed/tmbed_src/tmbed/models/t5/`, `Rostlab/prot_t5_xl_half_uniref50-enc` -- mismo encoder que el `Rostlab/prot_t5_xl_uniref50` que pedia el script original, verificado que carga y produce `d_model=1024` igual). **Descubrimiento importante:** `StackGlyEmbed/` es un repo git anidado (tiene su propio `.git`) -- git NO permite des-ignorar un archivo especifico dentro de un repo anidado con ningun patron de `.gitignore` (se probo el patron en cascada `dir/* + !dir/sub/ + dir/sub/* + !dir/sub/archivo`, no funciono: git colapsa el directorio entero como opaco). Por eso el script de integracion vive en `src/engines/stackglyembed_predict_local.py` (arbol versionado normal), NO dentro de `StackGlyEmbed/`, y recibe la carpeta de pickles del clasificador (`STACKGLYEMBED_MODELS_DIR`) por parametro en vez de asumirla relativa a si mismo. |
| ToxDL 2.0 (toxicidad) | Repo clonado en `B-Cell-Epitope-Prediction/ToxDL2/`, sin venv | **NO** | **NO** | **Diferido explicitamente por el usuario** (2026-07-21): necesita dominios InterPro por proteina que el repo no calcula localmente (solo ejemplo con IDs pegados a mano); el usuario evaluara opciones (vector cero degradado / InterProScan local / consulta puntual a InterPro en setup) antes de decidir. Checkpoint del modelo SI esta bundled (~94MB: `ToxDL2_model.pth` + embeddings de dominio). Usa ESM-2 650M -- el MISMO checkpoint que StackGlyEmbed necesita, compartible si se retoma. `device='cuda:0'` hardcodeado en `parameters/test_000.py`, hay que parchearlo (esta maquina no tiene GPU). |
| LANL + CATNAP (bnAb cross-ref) | **Datos descargados**, `reference_db/catnap/` (12 archivos, 28MB) + `reference_db/lanl_immunology/` (6 archivos, 29MB) | `lanl_catnap_engine.py` — **completo, probado con datos reales** (`query_bnab_crossref`) | **NO** — motor listo, decision de enganche pendiente (mismo tipo de decision que StackGlyEmbed) | bNAber (fuente original pedida) esta **muerta**: dominio parqueado/hijackeado (confirmado por Wayback Machine, ya asi desde antes de mayo 2025). Reemplazado por LANL HIV Molecular Immunology Database (`ab_all.csv`, 3799 registros de anticuerpos, 771 con epitopo lineal reportable -- el resto son conformacionales, fuera de alcance de un cruce por secuencia) + CATNAP (`abs_*.txt`, potencia/amplitud de neutralizacion). Motor implementado como cruce de subcadena (longest-common-substring, DP simple) entre candidatos y los 771 epitopos lineales, con umbral minimo configurable (`LANL_CATNAP_MIN_OVERLAP`, default 6 -- por debajo de eso el solapamiento es ruido estadistico; para epitopos de referencia MAS CORTOS que el umbral se exige el match completo). Validado con 10E8 (bnAb real del MPER): match de 13/13 residuos, IC50 real cruzado desde CATNAP (0.506 ug/mL, panel de 1321 virus). NO hace alineamiento a HXB2 ni captura epitopos conformacionales -- deliberado, ver docstring del modulo. `env_feature.txt`/`virseqs_aa*.fasta` quedaron sin usar (el cruce por secuencia via `ab_all.csv` resulto suficiente y mas directo que reconstruir fragmentos posicion-a-posicion). |

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
   NO se movieron (viven en sus repos `scipion-chem-*` hermanos o en
   `StackGlyEmbed/`): son venvs Python, mover un venv arrastra rutas
   absolutas embebidas en `pyvenv.cfg`/`bin/activate`/scripts de consola y
   puede romperlo; se referencian por ruta absoluta configurable en
   `Settings` en su lugar.
5. **AlgPred2 = Fase 4b (paralela a Fase 5/5b), NetCleave = anotacion dentro
   del reporte de Fase 5b** (2026-07-22, decision explicita del usuario tras
   presentarle el tradeoff). Razon: alergenicidad es una propiedad de la
   secuencia en si (no atada a una via de presentacion antigenica
   particular) mientras que el corte proteasomal SI esta mecanicamente
   atado a la via MHC-I (es el paso previo a la carga en el surco), asi que
   tiene mas sentido como señal complementaria del reporte de NetMHCpan que
   como fase generica propia. StackGlyEmbed quedo con el motor completo pero
   la misma decision de enganche AUN pendiente para la proxima sesion.
6. **Fase 4b (AlgPred2) es per-peptido, NO satisface por si sola el pedido
   original de Carlos** ("alergenicidad/toxicidad/antigenicidad del
   CONSTRUCTO MULTI-EPITOPO FINAL ENSAMBLADO", no por peptido individual --
   eso ya lo cubrian, mal, los wrappers de IIITD descartados por scraping).
   Confirmado explicitamente con el usuario 2026-07-22: Fase 4b se queda
   como filtro temprano util (descarta peptidos individualmente riesgosos
   antes de ensamblar), pero **el chequeo a nivel de constructo final SIGUE
   SIN CONSTRUIRSE** y es un item de scope aparte, pendiente de que exista
   un paso de ensamblaje de secuencia (fuera del alcance actual del
   pipeline, que termina en candidatos individuales, no en un constructo).
   No asumir que Fase 4b ya cubrio este pedido.

## Restriccion no negociable (recordada explicitamente por el usuario a mitad de sesion)

Todo debe correr local, nunca llamadas de red en tiempo de ejecucion del
pipeline. Instalacion/descarga de pesos es un paso de SETUP unico, no
runtime -- pero hay que verificar que ningun wrapper termine llamando a
`from_pretrained`/`torch.hub.load` sin fijar una ruta local + variables de
entorno offline (`HF_HUB_OFFLINE=1`, `TORCH_HOME` local). Pendiente
explicitamente para StackGlyEmbed (ver tabla arriba); TMbed ya lo hace bien
(usa `--model-dir` local siempre, nunca decide por si mismo).

## Siguiente sesion — orden sugerido (no vinculante)

1. ~~Decidir donde enganchan AlgPred2/NetCleave en las fases del pipeline~~ —
   HECHO 2026-07-22 (AlgPred2 = Fase 4b, NetCleave = anotacion en Fase 5b,
   ver Tabla C y "Decisiones de diseno" arriba).
2. ~~Terminar StackGlyEmbed~~ — motor HECHO 2026-07-22 (ver Tabla C), pero
   **decidir donde engancha en las fases del pipeline sigue pendiente**
   (mismo tipo de decision que el punto 1, no se tomo esta sesion).
3. ~~Escribir el motor de consulta bnAb~~ — HECHO 2026-07-22 (ver Tabla C),
   **decidir donde engancha en el pipeline sigue pendiente** (mismo tipo de
   decision que StackGlyEmbed).
4. Retomar ToxDL2 segun lo que decida el usuario sobre InterPro (bloqueado
   en una decision del usuario, no se avanzo esta sesion).
5. Orquestacion con manejo de memoria (batching, no cargar varios modelos
   pesados en simultaneo -- TMbed/StackGlyEmbed/ToxDL2 comparten ESM-2/ProtT5,
   posible reuso de un unico proceso "server" en vez de recargar el modelo
   por cada engine si el diseño lo permite) + logging de RAM + checkpointing,
   dado el OOM que origino todo esto. Relevante ahora que StackGlyEmbed
   tambien carga ESM-2 650M + ProtT5 + ProteinBERT en el mismo proceso.
6. Test end-to-end con PSMD7/PODXL/THBS2/SLC8A1 (y gp120 si da el tiempo).
7. Chequeo de alergenicidad/toxicidad/antigenicidad a nivel de CONSTRUCTO
   FINAL ensamblado (pedido original de Carlos, ver "Decisiones de diseno"
   punto 6 -- Fase 4b per-peptido NO lo cubre). Depende de que exista un
   paso de ensamblaje de secuencia, que hoy no existe en el pipeline.
