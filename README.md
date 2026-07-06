# SOTA-B-Epitope-Pipeline

Pipeline de cribado de alto rendimiento (HTS) para epítopos lineales de células B, diseñado para ejecutarse íntegramente en CPU (Intel i7, 12 núcleos, 16 GB RAM, sin GPU) bajo Linux WSL Ubuntu.

Este documento justifica matemática y arquitectónicamente las dos decisiones de diseño que un auditor técnico cuestionaría primero: (1) por qué usar las Escalas Z de Hellberg como espacio de entrada de la 1D-CNN de Fase 1, y (2) por qué ESM-2 es superior a un modelo de propensión clásico para la Fase 2. También documenta el contrato de memoria y el patrón adaptador de motores.

## Árbol de directorios

```
DiffSBDD/
├── requirements.txt
├── README.md
├── data/
│   ├── raw/candidatos.fasta
│   └── processed/
└── src/
    ├── __init__.py
    ├── main.py
    ├── models.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    ├── engines/
    │   ├── __init__.py
    │   ├── base_engine.py
    │   ├── antigenicity_cnn.py
    │   └── epitope_engine.py
    ├── validation/
    │   ├── __init__.py
    │   └── benchmark_suite.py
    └── utils/
        ├── __init__.py
        ├── batching.py
        ├── csv_exporter.py
        ├── exceptions.py
        ├── fasta_parser.py
        ├── logger_config.py
        └── memory_profiler.py
```

## 1. Justificación matemática: Escalas Z de Hellberg como espacio de entrada de la 1D-CNN

### 1.1 El problema de la representación

Una red convolucional no puede operar directamente sobre el alfabeto discreto de 20 símbolos de aminoácidos. Cualquier codificación debe preservar **continuidad fisicoquímica**: dos aminoácidos con propiedades similares (p. ej. Leucina e Isoleucina) deben proyectarse a puntos cercanos en el espacio de entrada, de forma que los filtros convolucionales puedan generalizar a través de sustituciones conservativas — un requisito básico para detectar motifs antigénicos que toleran variabilidad de secuencia (deriva antigénica, polimorfismo entre cepas).

Un one-hot encoding (20 canales binarios) viola esta propiedad: la distancia euclídea entre cualquier par de aminoácidos es idéntica (`√2`), sin importar su similitud biológica real. El filtro convolucional tendría que *aprender* la biofísica desde cero a partir de datos, lo cual es estadísticamente ineficiente con los tamaños de dataset típicos de IEDB (cientos a miles de epítopos confirmados, no millones).

### 1.2 Construcción de las Escalas Z (Hellberg et al., 1987; Sandberg et al., 1998)

Hellberg y colaboradores midieron o compilaron **29 descriptores fisicoquímicos** independientes para cada uno de los 20 aminoácidos codificados genéticamente (volumen de Van der Waals, momentos dipolares, superficie accesible al solvente, constantes de partición octanol/agua, pKa de cadena lateral, etc.). Sobre la matriz resultante `X ∈ ℝ^(20×29)` se aplicó un **Análisis de Componentes Principales (PCA)**:

```
X = U Σ Vᵀ
```

Las tres primeras componentes principales (`z1, z2, z3`), correspondientes a los tres valores singulares dominantes de `Σ`, capturan **más del 95% de la varianza fisicoquímica total** de los 29 descriptores originales. Cada componente tiene una interpretación biológica directa:

| Componente | Interpretación biofísica dominante |
|---|---|
| `z1` | Hidrofobicidad / electronegatividad (carácter polar vs. apolar de la cadena lateral) |
| `z2` | Tamaño estérico / volumen molecular |
| `z3` | Polarizabilidad / carácter electrónico (capacidad de formar puentes de hidrógeno, carga parcial) |

Esta compresión de 29 → 3 dimensiones es exactamente lo que hace viable la codificación `(Batch, 3, N)` que alimenta la 1D-CNN: cada posición de la secuencia se convierte en un punto en `ℝ³` con significado fisicoquímico interpretable, en lugar de un vector disperso de 20 dimensiones sin estructura métrica.

### 1.3 Por qué esto es relevante para epítopos de células B

Los epítopos lineales de células B son reconocidos por anticuerpos en la **superficie accesible al solvente** de una proteína plegada. Esta accesibilidad está gobernada casi exclusivamente por `z1` (hidrofilicidad: los epítopos tienden a ser regiones hidrofílicas expuestas) y por la flexibilidad de bucle, correlacionada con `z2`/`z3` (residuos pequeños y polares como Gly, Ser, Asn favorecen bucles flexibles y accesibles). La proyección de Hellberg no es una elección arbitraria: es la base fisicoquímica mínima suficiente sobre la que operan los predictores de propensión clásicos (BepiPred 1.0, VaxiJen), y aquí se reutiliza como *feature engineering* de entrada para que la 1D-CNN aprenda combinaciones no lineales de estas tres señales — algo que un modelo lineal de covarianza cruzada (ACC) no puede capturar.

### 1.4 Por qué una 1D-CNN y no el modelo lineal ACC clásico

El modelo ACC (Auto Cross Covariance) calcula covarianzas de `z_j` desplazadas por un *lag* fijo y las combina linealmente. Esto solo captura correlaciones de **orden par** y a **distancias fijas**. Una 1D-CNN con `Conv1d(kernel=5) → BatchNorm1d → ReLU` apilada dos veces:

1. Aprende **filtros de motif** de longitud variable (receptive field acumulado de 9 posiciones tras dos capas con kernel 5), equivalente a detectar patrones tipo "hidrofílico-flexible-hidrofílico" sin fijar a priori el lag.
2. La `ReLU` introduce **no linealidad**, permitiendo combinaciones tipo AND/OR entre canales `z1, z2, z3` que el modelo lineal no puede representar.
3. El **Global Max Pooling enmascarado** (invariante a la posición y a la longitud de la secuencia) selecciona la evidencia de motif antigénico más fuerte en toda la proteína, replicando la lógica biológica de que basta con **una** región suficientemente antigénica para que la proteína completa sea inmunogénica — la misma razón por la que BepiPred y VaxiJen reportan un score agregado por proteína en la fase de cribado grueso.
4. Todo el forward pass es `O(N)` en la longitud de secuencia (frente a `O(N·lag)` del ACC), y se ejecuta en milisegundos en CPU gracias al tamaño reducido de la red (dos capas convolucionales, sin necesidad de GPU).

**Nota de honestidad científica:** la arquitectura aquí implementada es correcta y funcional, pero sus pesos se inicializan de forma determinista (semilla fija) cuando no existe un checkpoint entrenado en `ANTIGENICITY_CNN_WEIGHTS_PATH`. Antes de uso en producción, debe entrenarse con descenso de gradiente (pérdida `BCELoss`) sobre un corpus etiquetado de IEDB (positivos) y proteínas housekeeping (negativos) — precisamente el corpus que consume `src/validation/benchmark_suite.py` para auditar el modelo resultante.

## 2. Justificación matemática: ESM-2 frente a modelos de propensión clásicos (Fase 2)

### 2.1 Limitación fundamental de los modelos de propensión

Los predictores de epítopos clásicos (Parker, Chou-Fasman, Emini) asignan una probabilidad a cada residuo como una función de una **ventana local fija** (típicamamente 5-7 residuos) de escalas fisicoquímicas. Esto ignora la **estructura de largo alcance**: un epítopo conformacional o un bucle expuesto depende de interacciones con residuos que pueden estar a decenas o cientos de posiciones de distancia en la secuencia primaria, pero próximos en el espacio 3D plegado.

### 2.2 ESM-2 como aproximación al plegamiento sin estructura 3D explícita

ESM-2 (`facebook/esm2_t30_150M_UR50D`) es un *transformer* de 30 capas entrenado con el objetivo de **modelado de lenguaje enmascarado (Masked Language Modeling)** sobre cientos de millones de secuencias de UniRef50:

```
L = -E_{i ∈ M} [ log P(x_i | x_{\ V \ M}) ]
```

donde `M` es el conjunto de posiciones enmascaradas. Para predecir correctamente un residuo enmascarado, el mecanismo de **autoatención** (`Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`) debe aprender, capa a capa, dependencias de largo alcance entre posiciones — que en la práctica se ha demostrado que codifican información de **contacto estructural y coevolución**, sin que el modelo haya visto nunca una coordenada 3D durante el entrenamiento (Rives et al., 2021; Lin et al., 2023). El embedding de la última capa (`ℝ^640` para la variante `t30_150M`) es, por tanto, una representación por residuo que integra contexto global de la proteína — algo que ninguna ventana deslizante de escalas fisicoquímicas puede aproximar.

### 2.3 Por qué esto mejora la detección de epítopos frente a un score de propensión

Un epítopo de células B es, por definición estructural, una región de la superficie plegada accesible a un anticuerpo. La accesibilidad al solvente es una propiedad **emergente del plegamiento completo**, no de la secuencia local. Al clasificar sobre el embedding contextual de ESM-2 en lugar de sobre una ventana de escalas Z, el `ResidueClassifier` (MLP de dos capas: `Linear(640→128) → ReLU → Dropout → Linear(128→1) → Sigmoid`) recibe una señal que ya incorpora información aproximada de exposición estructural, mejorando la separabilidad estadística entre residuos epitópicos y no epitópicos frente a los modelos de propensión de las décadas 1980-1990 (fundamento de BepiPred 3.0/4.0 frente a BepiPred 1.0).

### 2.4 Por qué un MLP en PyTorch y no un Random Forest

Se optó por un MLP nativo de PyTorch como cabeza de clasificación (en lugar de un `RandomForestClassifier` de scikit-learn) por tres razones de ingeniería, no solo de rendimiento:

1. **Un único framework de tensores**: evita conversiones `torch.Tensor ↔ numpy` adicionales en el *hot path* de inferencia sobre miles de secuencias, reduciendo overhead de CPU.
2. **`torch.no_grad()` uniforme**: todo el forward pass (ESM-2 + cabeza) se ejecuta bajo el mismo contexto de no acumulación de gradientes, simplificando el perfilado de memoria.
3. **Serialización estable**: los pesos se guardan con `torch.save`/`state_dict`, evitando el riesgo de incompatibilidad de versiones de `pickle` entre versiones de scikit-learn en despliegues HPC de larga vida.

### 2.5 Patrón Adaptador (Strategy) entre `NativeESM2Engine` y `CLIWrapperEngine`

La interfaz `BaseEpitopePredictor` desacopla el orquestador de la implementación concreta del motor de Fase 2:

- **`NativeESM2Engine`**: carga el modelo directamente en el proceso Python (mínima latencia, ideal para iteración de desarrollo local).
- **`CLIWrapperEngine`**: delega en un binario externo vía `subprocess`, comunicándose por archivos temporales seguros (`tempfile.mkdtemp`, permisos `0o700`, limpieza garantizada en `finally`). Esto blinda el pipeline ante actualizaciones de la herramienta de referencia (p. ej. una futura migración a BepiPred-4.0 o un contenedor Scipion en un clúster HPC) sin tocar el código del orquestador: basta con que el binario externo respete el contrato documentado en el docstring de `CLIWrapperEngine` (entrada `--input/--output/--threshold`, salida CSV con columnas `sequence_id,position,residue,epitope_probability`).

`EpitopePredictorFactory` selecciona la implementación en tiempo de ejecución según `Settings.PREDICTOR_ENGINE` o el flag `--engine` de la CLI, sin que `main.py` conozca ninguna de las dos clases concretas.

## 3. Disciplina de memoria (16 GB RAM, sin GPU)

Todo cálculo con tensores en `antigenicity_cnn.py` y `epitope_engine.py` sigue el mismo protocolo:

1. **Mini-lotes dinámicos** (`src/utils/batching.py`): las secuencias se ordenan por longitud y se agrupan hasta un presupuesto de `longitud_máxima × n_secuencias` (área acolchada), no un conteo fijo de secuencias. Esto acota la memoria de cada forward pass independientemente de la varianza de longitudes del FASTA de entrada.
2. **`torch.no_grad()` estricto** en toda la superficie de inferencia (Fase 1 y Fase 2): nunca se construye el grafo de autodiferenciación, eliminando la mayor fuente de crecimiento de memoria en PyTorch.
3. **Liberación explícita** (`del tensor, mask, probs, ...`) seguida de `gc.collect()` tras cada lote, antes de procesar el siguiente.
4. **Perfilado de memoria implícito** (`src/utils/memory_profiler.py`): usa `resource.getrusage` de la librería estándar (sin dependencias externas como `psutil`) para registrar el pico de RSS tras cada lote y emitir una advertencia si supera el presupuesto configurado (`ANTIGENICITY_MEMORY_BUDGET_MB`, `ESM_MEMORY_BUDGET_MB`), permitiendo detectar una tendencia de fuga de memoria antes de que derive en un OOM en ejecuciones HTS de horas de duración.

## 4. Suite de auditoría científica

`src/validation/benchmark_suite.py` ingiere un FASTA positivo (epítopos confirmados de IEDB) y un FASTA negativo (proteínas housekeeping/intracelulares, que no deberían presentar antigenicidad de superficie), ejecuta el motor de Fase 1 sobre ambos y calcula, con `scikit-learn`:

- Matriz de confusión (`sklearn.metrics.confusion_matrix`).
- Sensibilidad (`TP / (TP + FN)`) y Especificidad (`TN / (TN + FP)`).
- Tasa de Falsos Positivos (`FP / (FP + TN) = 1 - Especificidad`).
- ROC-AUC (`sklearn.metrics.roc_auc_score`) sobre los scores continuos, independiente del umbral de decisión elegido.

Uso:

```bash
python -m src.validation.benchmark_suite -p iedb_positivos.fasta -n housekeeping_negativos.fasta -t 0.6
```

## 5. Uso de la CLI principal

```bash
python src/main.py -i data/raw/candidatos.fasta \
    -t 0.6 \
    --engine esm2 \
    --epitope-threshold 0.35 \
    --offline
```

| Flag | Descripción |
|---|---|
| `-i, --input` | Ruta al FASTA de entrada. |
| `-t, --threshold` | Umbral de antigenicidad de Fase 1. |
| `--engine` | `esm2` (nativo, desarrollo local) o `cli` (subprocess, HPC). |
| `--offline` | Fuerza `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`: sin llamadas de red al Hub. |
| `--epitope-threshold` | Umbral de probabilidad de epítopo de Fase 2. |
| `-o, --output-dir` | Directorio de salida para los CSV de resultados. |

La salida se exporta a `ranking_resumen.csv` (resumen por proteína) y `residuos_detalle.csv` (detalle por residuo), además de un reporte ASCII ejecutivo en consola.
