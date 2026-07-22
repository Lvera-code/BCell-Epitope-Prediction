"""Configuracion centralizada del pipeline: rutas, umbrales y credenciales externas.

Todos los parametros ajustables se resuelven desde variables de entorno con
valores por defecto conservadores, para permitir reconfiguracion sin tocar
codigo fuente ni comprometer credenciales al subir el repositorio a GitHub.
"""

import os
import sys
from pathlib import Path


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings:
    """Punto unico de verdad para toda configuracion del pipeline."""

    # --- Fase 2: Prediccion de Antigenicidad (BepiPred-3.0, ejecucion LOCAL) ---
    # Pivote de arquitectura (2026-07-10): se abandono definitivamente la
    # estrategia via BioLib (API/nube) por la latencia e imprevisibilidad de
    # los cold-start de los contenedores ESM-2 bajo carga publica (ver
    # historial de src/engines/bepipred_engine.py). BepiPred-3.0 ahora corre
    # 100% en local via subprocess contra el codigo fuente oficial descargado
    # manualmente (licencia academica DTU Health Tech).
    #
    # Ninguna ruta se hardcodea: todo se resuelve desde variables de entorno,
    # con defaults que asumen el paquete descomprimido en la raiz del
    # proyecto tal como lo distribuye DTU (ver README.md - Seccion de
    # Instalacion).
    BEPIPRED_HOME: Path = Path(_env_str("BEPIPRED_HOME", "bepipred-3.0b.src/BepiPred3_src"))
    BEPIPRED_CLI_SCRIPT_NAME: str = _env_str("BEPIPRED_CLI_SCRIPT_NAME", "bepipred3_CLI.py")
    # Interprete de Python a usar para invocar el CLI de BepiPred. BepiPred-3.0
    # fija versiones antiguas de sus dependencias (torch==1.12.0, numpy==1.20.2)
    # que pueden chocar con las del entorno principal del pipeline: se
    # recomienda un venv dedicado (ver README.md) y apuntar aqui a su python.
    # Por defecto usa el mismo interprete que corre pipeline.py.
    BEPIPRED_PYTHON_BIN: str = _env_str("BEPIPRED_PYTHON_BIN", sys.executable)
    # 'vt_pred' (variable threshold) o 'mjv_pred' (majority vote). Solo afecta
    # los archivos de prediccion propios de BepiPred que NO consumimos (nuestra
    # Fase 3 hace su propio agrupamiento local sobre raw_output.csv), pero el
    # flag '-pred' es obligatorio en su CLI.
    BEPIPRED_PRED_MODE: str = _env_str("BEPIPRED_PRED_MODE", "vt_pred")
    BEPIPRED_TIMEOUT_SECONDS: int = _env_int("BEPIPRED_TIMEOUT_SECONDS", 1800)
    BEPIPRED_DOWNLOAD_URL: str = (
        "https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=bepipred"
        "&version=3.0&packageversion=3.0b&platform=src"
    )

    # Umbral y longitud minima aplicados LOCALMENTE en Fase 3 (ver
    # `extract_epitopes` en bepipred_engine.py) sobre el raw_output.csv que
    # genera la ejecucion local de BepiPred-3.0.
    BEPIPRED_THRESHOLD: float = _env_float("BEPIPRED_THRESHOLD", 0.1512)
    BEPIPRED_MIN_EPITOPE_LENGTH: int = _env_int("BEPIPRED_MIN_EPITOPE_LENGTH", 9)
    # Ventana deslizante de Fase 3 (ver `extract_epitopes`): tamano fijo del
    # footprint minimo de reconocimiento de celula B, y tolerancia de gaps
    # (residuos individuales por debajo de BEPIPRED_THRESHOLD) permitida
    # dentro de cada ventana de 9 aa para no perder epitopos reales por un
    # unico residuo debil.
    BEPIPRED_WINDOW_SIZE: int = _env_int("BEPIPRED_WINDOW_SIZE", 9)
    BEPIPRED_MAX_GAP_RESIDUES: int = _env_int("BEPIPRED_MAX_GAP_RESIDUES", 2)

    # Nombre del CSV de salida crudo que escribe BepiPred-3.0 localmente
    # (confirmado leyendo bp3/bepipred3.py::create_csvfile en el codigo fuente
    # oficial: columnas 'Accession,Residue,BepiPred-3.0 score,...').
    BEPIPRED_RAW_OUTPUT_FILENAME: str = _env_str("BEPIPRED_RAW_OUTPUT_FILENAME", "raw_output.csv")

    BEPIPRED_OUTPUT_DIR: Path = Path(_env_str("BEPIPRED_OUTPUT_DIR", "produccion_resultados/bepipred3"))

    # --- Fase 2 (segundo motor): Prediccion de Antigenicidad (EpiDope, ejecucion LOCAL) ---
    # A diferencia de BepiPred-3.0 y NetMHCIIpan-4.3, EpiDope es codigo abierto
    # (licencia MIT, github.com/rnajena/EpiDope -fork activamente mantenido,
    # sucesor de github.com/flomock/EpiDope-) e instalable via conda sin
    # solicitud academica. Fija un entorno completo (``epidope.yml`` del
    # propio repo: Python 3.6, TensorFlow 1.13, Keras 2.3, PyTorch 0.4,
    # AllenNLP 0.7.2 para embeddings ELMo) incompatible con el entorno
    # principal del pipeline (mismo problema que BepiPred con torch==1.12.0):
    # requiere un entorno conda dedicado, creado EXACTAMENTE con ese
    # ``epidope.yml`` (no version por version a mano: la resolucion de
    # dependencias de ese stack es fragil), igual patron que
    # ``.venv-bepipred`` (ver README.md - Seccion de Instalacion). Los pesos
    # del modelo y los embeddings ELMo vienen empaquetados en el propio repo
    # (``epidope/epidope_weights``, ``epidope/elmo_settings``): la inferencia
    # es 100% local, sin ninguna llamada de red.
    #
    # Invocacion: si EPIDOPE_BIN apunta a un ejecutable existente, se llama
    # directamente (bypass de conda); si no, se invoca via
    # 'conda run -p EPIDOPE_CONDA_PREFIX epidope' (o -n EPIDOPE_CONDA_ENV si
    # se prefiere un entorno por nombre en vez de por prefijo de ruta).
    EPIDOPE_CONDA_PREFIX: str = _env_str("EPIDOPE_CONDA_PREFIX", ".conda-epidope")
    EPIDOPE_CONDA_ENV: str = _env_str("EPIDOPE_CONDA_ENV", "")
    EPIDOPE_BIN: str = _env_str("EPIDOPE_BIN", "")
    EPIDOPE_TIMEOUT_SECONDS: int = _env_int("EPIDOPE_TIMEOUT_SECONDS", 1800)
    EPIDOPE_DOWNLOAD_URL: str = "https://github.com/rnajena/EpiDope"

    # Umbral y longitud minima aplicados LOCALMENTE en Fase 3 (misma logica de
    # ventana deslizante que BepiPred, ver `src/engines/epitope_mapping.py`)
    # sobre los scores crudos por residuo que genera la ejecucion local de
    # EpiDope. 0.818 es el umbral por defecto del propio EpiDope (ver su
    # `epidope/cli.py`), MUY distinto en escala al de BepiPred (0.1512): no
    # son scores comparables entre si, cada motor conserva su propio umbral.
    EPIDOPE_THRESHOLD: float = _env_float("EPIDOPE_THRESHOLD", 0.818)
    EPIDOPE_MIN_EPITOPE_LENGTH: int = _env_int("EPIDOPE_MIN_EPITOPE_LENGTH", 9)
    EPIDOPE_WINDOW_SIZE: int = _env_int("EPIDOPE_WINDOW_SIZE", 9)
    EPIDOPE_MAX_GAP_RESIDUES: int = _env_int("EPIDOPE_MAX_GAP_RESIDUES", 2)

    EPIDOPE_OUTPUT_DIR: Path = Path(_env_str("EPIDOPE_OUTPUT_DIR", "produccion_resultados/epidope"))

    # --- Fase 1 / Orquestador: carpetas de entrada y salida del pipeline ---
    FASTA_INPUT_DIR: Path = Path(_env_str("FASTA_INPUT_DIR", "fasta_inputs"))
    FASTA_OUTPUT_DIR: Path = Path(_env_str("FASTA_OUTPUT_DIR", "fasta_outputs"))

    # --- Fase 1.5: Extraccion de estructura (PDB/mmCIF via gemmi, LOCAL) ---
    # Estrategia de seleccion de cadena cuando el archivo de entrada tiene mas
    # de una cadena proteica (ver `src/utils/structure_parser.py`). Nunca
    # implicita: la cadena elegida siempre se loggea con su motivo.
    #   'longest'  -> se elige la cadena con mas residuos en su polimero
    #                 (`chain.get_polymer().length()` via gemmi).
    #   'explicit' -> se usa PDB_EXPLICIT_CHAIN_ID (obligatorio en ese caso).
    PDB_CHAIN_SELECTION_STRATEGY: str = _env_str("PDB_CHAIN_SELECTION_STRATEGY", "longest")
    PDB_EXPLICIT_CHAIN_ID: str = _env_str("PDB_EXPLICIT_CHAIN_ID", "")

    # Modo de procesamiento para input de tipo estructura (ver
    # `src/engines/engine_registry.py::active_engines_for`):
    #   'structure_only'         -> solo corren los motores estructurales
    #                               (DiscoTope-3.0 + ScanNet).
    #   'structure_and_sequence' -> ademas se deriva un FASTA canonico (ATMSEQ)
    #                               de la estructura y se pasa tambien a
    #                               BepiPred-3.0 + EpiDope.
    # Default 'structure_and_sequence': maximiza cobertura (los 4 motores)
    # cuando el input es un PDB, salvo que se pida lo contrario explicitamente
    # (Setting o '--pdb-mode' en pipeline.py).
    PDB_PROCESSING_MODE: str = _env_str("PDB_PROCESSING_MODE", "structure_and_sequence")

    # --- Fase 2 (motor estructural 1/2): DiscoTope-3.0, ejecucion LOCAL ---
    # Mismo grupo (DTU Health Tech) que BepiPred-3.0, pero a diferencia de
    # este SI es instalable directo via git+pip (licencia Creative Commons,
    # sin solicitud academica separada): github.com/Magnushhoie/DiscoTope-3.0.
    # Entorno aislado dedicado (.venv-discotope) por el mismo motivo que
    # BepiPred/EpiDope: stack de dependencias propio (pytorch-geometric,
    # xgboost, biotite) que puede chocar con el resto del pipeline.
    DISCOTOPE_INSTALL_PATH: Path = Path(_env_str("DISCOTOPE_INSTALL_PATH", "DiscoTope-3.0"))
    DISCOTOPE_PYTHON_BIN: str = _env_str("DISCOTOPE_PYTHON_BIN", str(Path(".venv-discotope/bin/python")))
    # ESM-IF1 (inverse folding) descarga sus pesos via el cache de torch hub
    # en tiempo de inferencia. Se redirige ese cache (variable de entorno
    # TORCH_HOME) a una ruta persistente FUERA del repo del proyecto, para no
    # volver a descargarlos en cada corrida (ver `_build_env` en
    # discotope_engine.py). Los pesos del ensemble XGBoost propio de
    # DiscoTope-3.0 (`models.zip`) NO se cachean aqui: se descomprimen una
    # sola vez dentro de DISCOTOPE_INSTALL_PATH siguiendo la guia oficial del
    # repo, igual que BepiPred-3.0 con su paquete descargado.
    DISCOTOPE_WEIGHTS_CACHE_DIR: Path = Path(
        _env_str("DISCOTOPE_WEIGHTS_CACHE_DIR", str(Path.home() / ".cache" / "bcell-epitope-pipeline" / "discotope-weights"))
    )
    DISCOTOPE_STRUC_TYPE: str = _env_str("DISCOTOPE_STRUC_TYPE", "solved")  # 'solved' | 'alphafold'
    DISCOTOPE_TIMEOUT_SECONDS: int = _env_int("DISCOTOPE_TIMEOUT_SECONDS", 1800)
    DISCOTOPE_DOWNLOAD_URL: str = "https://github.com/Magnushhoie/DiscoTope-3.0/"

    # Umbral y longitud minima aplicados LOCALMENTE en Fase 3 (misma logica de
    # ventana deslizante que BepiPred/EpiDope) sobre 'calibrated_score' (ver
    # ADR "Por que calibrated_score" en discotope_engine.py) -- NO sobre
    # 'DiscoTope-3.0_score' cruda como en una version anterior de este motor
    # (esa version SI requeria una calibracion casera de una sola estructura
    # de ejemplo; ver historial de git para esa version si hace falta).
    #
    # 0.90 ES EL UMBRAL OFICIAL publicado por los autores para
    # 'calibrated_score' (nivel "moderate" del flag CLI
    # '--calibrated_score_epi_threshold', confirmado via el paper: Hoie et
    # al., Frontiers in Immunology 2024). Los autores publican 3 niveles de
    # referencia con recall esperado, todos validos segun el objetivo:
    #   0.40 -> "low"      (~70% recall, mas candidatos, mas falsos positivos)
    #   0.90 -> "moderate" (default, balance recall/precision)
    #   1.51 -> "higher"   (mayor precision, menos candidatos)
    # Ajustable via DISCOTOPE_THRESHOLD sin tocar codigo si se prefiere otro
    # nivel de la tabla oficial.
    DISCOTOPE_THRESHOLD: float = _env_float("DISCOTOPE_THRESHOLD", 0.90)
    DISCOTOPE_MIN_EPITOPE_LENGTH: int = _env_int("DISCOTOPE_MIN_EPITOPE_LENGTH", 9)
    DISCOTOPE_WINDOW_SIZE: int = _env_int("DISCOTOPE_WINDOW_SIZE", 9)
    DISCOTOPE_MAX_GAP_RESIDUES: int = _env_int("DISCOTOPE_MAX_GAP_RESIDUES", 2)

    DISCOTOPE_OUTPUT_DIR: Path = Path(_env_str("DISCOTOPE_OUTPUT_DIR", "produccion_resultados/discotope3"))

    # --- Fase 2 (motor estructural 2/2): ScanNet, ejecucion LOCAL ---
    # A diferencia de DiscoTope-3.0, ScanNet (github.com/jertubiana/ScanNet)
    # no requiere ningun software externo mas alla de su propio stack Python
    # (numpy/numba/scikit-learn/tensorflow/keras, Python 3.6.12) -- pero ese
    # stack SI es antiguo e incompatible con el resto del pipeline, mismo
    # motivo que EpiDope para requerir entorno aislado dedicado
    # (.venv-scannet). Runtime alternativo via Docker (imagen oficial
    # 'jertubiana/scannet'), pensado para evitar tener que resolver ese stack
    # antiguo a mano.
    #
    # AMBOS runtimes fueron instalados y validados empiricamente (2026-07-20,
    # ver ADR en scannet_engine.py): 'docker pull jertubiana/scannet' +
    # 'docker inspect' confirmaron WORKDIR=/ScanNet (el default de
    # SCANNET_DOCKER_WORKDIR, sin ajuste necesario) y una corrida real dio
    # resultados identicos byte a byte al runtime 'venv' sobre el mismo PDB.
    SCANNET_RUNTIME: str = _env_str("SCANNET_RUNTIME", "docker")  # 'docker' | 'venv'
    SCANNET_INSTALL_PATH: Path = Path(_env_str("SCANNET_INSTALL_PATH", "ScanNet"))
    # NOTA (confirmado al instalar/validar realmente el runtime 'venv'): pese
    # al nombre de la variable, en la practica NINGUN sistema moderno trae ya
    # un interprete Python 3.6.12 instalado (requisito exacto de ScanNet) del
    # que un simple 'python3 -m venv' pueda partir. Lo que si funciona de
    # forma reproducible es crear el entorno con conda, que SI distribuye
    # builds de Python 3.6.12: 'conda create -n scannet_env python=3.6.12'
    # seguido de 'pip install -r ScanNet/requirements.txt' dentro de ese
    # entorno. SCANNET_PYTHON_BIN admite cualquier interprete (venv o conda);
    # el default de abajo asume conda por ser la ruta que de verdad funciono.
    SCANNET_PYTHON_BIN: str = _env_str(
        "SCANNET_PYTHON_BIN", str(Path.home() / "miniconda3" / "envs" / "scannet_env" / "bin" / "python")
    )
    SCANNET_DOCKER_IMAGE: str = _env_str("SCANNET_DOCKER_IMAGE", "jertubiana/scannet")
    SCANNET_DOCKER_WORKDIR: str = _env_str("SCANNET_DOCKER_WORKDIR", "/ScanNet")
    SCANNET_TIMEOUT_SECONDS: int = _env_int("SCANNET_TIMEOUT_SECONDS", 1800)
    SCANNET_DOWNLOAD_URL: str = "https://github.com/jertubiana/ScanNet"

    # Umbral y longitud minima aplicados LOCALMENTE en Fase 3 sobre 'Binding
    # site probability' (columna cruda del CSV oficial, escala 0.00-1.00 por
    # residuo, salida sigmoide del modelo).
    #
    # INVESTIGADO A FONDO (2026-07-20): a diferencia de DiscoTope-3.0 (que SI
    # publica un umbral oficial via 'calibrated_score', ver
    # DISCOTOPE_THRESHOLD), los autores de ScanNet NO publican un punto de
    # corte fijo para el modelo de epitopos (revisado el paper, el repo
    # completo y su propio 'utilities/chimera.py': los unicos numeros que
    # usan son un gradiente de 8 colores para visualizacion, 0.05-1.00, no un
    # umbral de clasificacion). Tiene sentido: el score bruto de ScanNet varia
    # mucho de una cadena a otra (en la cadena de prueba real, max=0.291;
    # nada garantiza que otra cadena no llegue a 0.7) -un numero absoluto fijo
    # nunca generaliza bien entre proteinas distintas.
    #
    # Por eso el comportamiento por defecto de `extract_epitopes` (ver
    # scannet_engine.py) NO usa este valor fijo: calcula un umbral ADAPTATIVO
    # por accession, como el percentil `SCANNET_THRESHOLD_PERCENTILE` de los
    # scores de ESA cadena especifica -mismo principio que 'calibrated_score'
    # de DiscoTope-3.0 (normalizar por la distribucion propia de cada
    # antigeno en vez de un corte absoluto universal), aplicado aqui porque
    # ScanNet no lo hace por si solo-. SCANNET_THRESHOLD se conserva como
    # override MANUAL (fijo, no adaptativo) para quien prefiera un numero
    # exacto y reproducible entre corridas -- ver '--scannet-threshold' en
    # pipeline.py.
    SCANNET_THRESHOLD_PERCENTILE: float = _env_float("SCANNET_THRESHOLD_PERCENTILE", 90.0)
    SCANNET_THRESHOLD: float = _env_float("SCANNET_THRESHOLD", 0.10)
    SCANNET_MIN_EPITOPE_LENGTH: int = _env_int("SCANNET_MIN_EPITOPE_LENGTH", 9)
    SCANNET_WINDOW_SIZE: int = _env_int("SCANNET_WINDOW_SIZE", 9)
    SCANNET_MAX_GAP_RESIDUES: int = _env_int("SCANNET_MAX_GAP_RESIDUES", 2)

    SCANNET_OUTPUT_DIR: Path = Path(_env_str("SCANNET_OUTPUT_DIR", "produccion_resultados/scannet"))

    # --- Fase 4: Filtro de tolerancia inmunologica (BLASTp local) ---
    # Prefijo (sin extension) de la base de datos BLAST del proteoma humano,
    # generada localmente con 'makeblastdb'. NUNCA se hardcodea: se lee de la
    # variable de entorno BLAST_HUMAN_DB (con un default razonable que asume
    # el layout de 'reference_db/' descrito en README.md - Seccion de
    # Instalacion). Si la base de datos resuelta no existe, la Fase 4 se
    # detiene con un error accionable (ver `_check_blast_environment` en
    # blast_engine.py), igual que la validacion de instalacion de BepiPred.
    BLAST_HUMAN_DB: str = _env_str("BLAST_HUMAN_DB", "reference_db/human_proteome_db")
    BLAST_IDENTITY_THRESHOLD: float = _env_float("BLAST_IDENTITY_THRESHOLD", 75.0)
    # CONFIRMADO EMPIRICAMENTE (2026-07-20, PDBs reales 1fv2/7c4s/7lkh via
    # DiscoTope-3.0/ScanNet): 'max_pident' original tomaba el %identidad
    # maximo de CUALQUIER hit de BLAST, sin considerar cuanto del peptido
    # realmente cubria ese alineamiento. Con 'blastp-short' + evalue=50 (laxo
    # a proposito, ver BLAST_EVALUE_SHORT), un fragmento de 5-6 aa 100%
    # identico dentro de un peptido de 14-31 aa es estadisticamente esperable
    # por puro azar contra el proteoma humano completo (~11M residuos: un
    # 5-mero especifico se espera ~3 veces solo por azar) y contaba exactamente
    # igual que un homologo real de longitud completa -- rechazando por
    # "Autoinmunidad" casi cualquier peptido corto, real o no. Un hit solo
    # cuenta para 'max_pident' si su longitud de alineamiento cubre al menos
    # esta fraccion de la longitud del peptido consultado (ver
    # `_max_identity_by_query` en blast_engine.py).
    BLAST_MIN_QUERY_COVERAGE: float = _env_float("BLAST_MIN_QUERY_COVERAGE", 0.9)
    # E-value seleccionado dinamicamente por tramo de longitud del peptido
    # (ver `_select_evalue` en blast_engine.py). La estadistica de BLAST
    # penaliza a los peptidos cortos: con el e-value por defecto de blastp
    # (10), un match identico de 9-25 aa contra el proteoma humano puede
    # descartarse por "no significativo", arruinando el filtro de
    # autoinmunidad. Para secuencias largas (dominios/proteinas completas)
    # aplica el criterio contrario: ahi si un e-value laxo generaria ruido de
    # homologias irrelevantes, por lo que se usan los valores estandar de
    # BLAST (mas estrictos cuanto mas larga la consulta).
    BLAST_EVALUE_SHORT: float = _env_float("BLAST_EVALUE_SHORT", 50.0)      # <= BLAST_SHORT_PEPTIDE_MAX_LEN aa
    BLAST_EVALUE_MEDIUM: float = _env_float("BLAST_EVALUE_MEDIUM", 0.1)     # BLAST_SHORT_PEPTIDE_MAX_LEN+1 .. BLAST_MEDIUM_PEPTIDE_MAX_LEN aa
    BLAST_EVALUE_LONG: float = _env_float("BLAST_EVALUE_LONG", 0.05)        # > BLAST_MEDIUM_PEPTIDE_MAX_LEN aa
    # Umbrales de longitud (aa) que deciden tanto el algoritmo de BLASTp como
    # el tramo de E-value de cada peptido (ver `_select_task` / `_select_evalue`
    # en blast_engine.py):
    #   <= BLAST_SHORT_PEPTIDE_MAX_LEN         -> '-task blastp-short', evalue=BLAST_EVALUE_SHORT
    #   BLAST_SHORT_PEPTIDE_MAX_LEN < len <= BLAST_MEDIUM_PEPTIDE_MAX_LEN -> '-task blastp', evalue=BLAST_EVALUE_MEDIUM
    #   >  BLAST_MEDIUM_PEPTIDE_MAX_LEN        -> '-task blastp', evalue=BLAST_EVALUE_LONG
    BLAST_SHORT_PEPTIDE_MAX_LEN: int = _env_int("BLAST_SHORT_PEPTIDE_MAX_LEN", 30)
    BLAST_MEDIUM_PEPTIDE_MAX_LEN: int = _env_int("BLAST_MEDIUM_PEPTIDE_MAX_LEN", 100)

    # --- Fase 5: Inmunogenicidad T-helper (MHC-II, NetMHCIIpan-4.3 LOCAL) ---
    # Pivote metodologico: toda prediccion de presentacion MHC-I (celulas T
    # CD8+, antes servida por MHCflurry/NetMHCpan) fue descartada. La Fase 5
    # ahora evalua exclusivamente presentacion MHC-II (celulas T-helper CD4+)
    # via NetMHCIIpan-4.3 ejecutado 100% en local por subprocess, mismo
    # patron que BepiPred-3.0 (Fase 2) y BLASTp+ (Fase 4): ninguna ruta se
    # hardcodea, todo se resuelve desde variables de entorno.
    NETMHCIIPAN_HOME: Path = Path(_env_str("NETMHCIIPAN_HOME", "netMHCIIpan-4.3"))
    NETMHCIIPAN_BINARY_NAME: str = _env_str("NETMHCIIPAN_BINARY_NAME", "netMHCIIpan")
    NETMHCIIPAN_TIMEOUT_SECONDS: int = _env_int("NETMHCIIPAN_TIMEOUT_SECONDS", 600)
    NETMHCIIPAN_DOWNLOAD_URL: str = "https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/"
    # Umbrales de %Rank POR DEFECTO de NetMHCIIpan-4.3 (ver 'netMHCIIpan.1':
    # flags -rankS/-rankW del binario). SB (aglutinador fuerte): Rank_EL <=
    # NETMHCIIPAN_RANK_STRONG. WB (aglutinador debil): Rank_EL <=
    # NETMHCIIPAN_RANK_WEAK. No se pasan -rankS/-rankW al comando: se
    # replica el mismo umbral aqui, en Python, para clasificar el .xls.
    NETMHCIIPAN_RANK_STRONG: float = _env_float("NETMHCIIPAN_RANK_STRONG", 1.0)
    NETMHCIIPAN_RANK_WEAK: float = _env_float("NETMHCIIPAN_RANK_WEAK", 5.0)
    # "Promiscuidad": un epitopo T-helper se reporta como 'Candidato Valido'
    # solo si clasifica SB o WB en al menos este numero de alelos distintos
    # del panel evaluado (cobertura poblacional, no un unico alelo HLA).
    NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES: int = _env_int("NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES", 3)

    # --- Inmunogenicidad T-citotoxica (MHC-I, NetMHCpan-4.2 LOCAL) ---
    # ADR de 2026-07-12 (descartar MHC-I) REVERTIDO 2026-07-21: ver docstring
    # de netmhciipan_engine.py. Mismo patron 100% local por subprocess que el
    # resto de motores; NUNCA se hardcodea la ruta.
    NETMHCPAN_HOME: Path = Path(_env_str("NETMHCPAN_HOME", "netMHCpan-4.2"))
    NETMHCPAN_BINARY_NAME: str = _env_str("NETMHCPAN_BINARY_NAME", "netMHCpan")
    NETMHCPAN_TIMEOUT_SECONDS: int = _env_int("NETMHCPAN_TIMEOUT_SECONDS", 600)
    NETMHCPAN_DOWNLOAD_URL: str = "https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/"
    # Umbrales de %Rank POR DEFECTO de NetMHCpan-4.2 (ver 'netMHCpan.1':
    # flags -rankS/-rankW del binario) -- DISTINTOS de los de NetMHCIIpan-4.3
    # (1.0/5.0): MHC-I tiene su propia escala de %Rank, no comparable 1:1.
    NETMHCPAN_RANK_STRONG: float = _env_float("NETMHCPAN_RANK_STRONG", 0.5)
    NETMHCPAN_RANK_WEAK: float = _env_float("NETMHCPAN_RANK_WEAK", 2.0)
    NETMHCPAN_MIN_PROMISCUOUS_ALLELES: int = _env_int("NETMHCPAN_MIN_PROMISCUOUS_ALLELES", 3)
    # Longitudes de epitopo MHC-I a evaluar en modo FASTA/proteina (ventana
    # deslizante interna de NetMHCpan via '-l'): 9-mero es el largo canonico
    # mas frecuente, 8/10/11 cubren la variabilidad real observada en
    # ligandos eluidos (IEDB/CEDAR, ver docstring del binario).
    NETMHCPAN_PEPTIDE_LENGTHS: str = _env_str("NETMHCPAN_PEPTIDE_LENGTHS", "8,9,10,11")

    # --- Alergenicidad (AlgPred 2.0 LOCAL) ---
    # Instalacion propia (venv dedicado + BLAST DB + MERCI.pl bundled), open
    # source (GPSR group), 100% local por subprocess. AlgPred2 vive en
    # scipion-chem-algpred/ (repo hermano, NO dentro de este proyecto): se
    # referencia por ruta absoluta configurable, igual que cualquier otro
    # motor -- nunca hardcodeada.
    ALGPRED_PYTHON_BIN: str = _env_str(
        "ALGPRED_PYTHON_BIN",
        "/home/enzo/DiffSBDD/scipion-chem-algpred/.venv-algpred/bin/python",
    )
    ALGPRED_SCRIPT_PATH: str = _env_str(
        "ALGPRED_SCRIPT_PATH",
        "/home/enzo/DiffSBDD/scipion-chem-algpred/.venv-algpred/lib/python3.10/site-packages/algpred2/python_scripts/algpred2.py",
    )
    ALGPRED_TIMEOUT_SECONDS: int = _env_int("ALGPRED_TIMEOUT_SECONDS", 300)
    # Umbral ML_Score por defecto del propio AlgPred2 (ver 'algpred2.py -h').
    ALGPRED_THRESHOLD: float = _env_float("ALGPRED_THRESHOLD", 0.3)

    # --- Cleavage MHC-I/II (NetCleave LOCAL, reentrenado con datos propios) ---
    # Instalacion propia (venv dedicado + IEDB/UniProt/UniParc descargados
    # localmente para reentrenar, ver netcleave_src/data/databases/), open
    # source (MIT), 100% local por subprocess. Vive en scipion-chem-netcleave/
    # (repo hermano): ruta absoluta configurable, nunca hardcodeada.
    NETCLEAVE_PYTHON_BIN: str = _env_str(
        "NETCLEAVE_PYTHON_BIN",
        "/home/enzo/DiffSBDD/scipion-chem-netcleave/.venv-netcleave/bin/python",
    )
    NETCLEAVE_SCRIPT_PATH: str = _env_str(
        "NETCLEAVE_SCRIPT_PATH",
        "/home/enzo/DiffSBDD/scipion-chem-netcleave/netcleave_src/NetCleave.py",
    )
    NETCLEAVE_TIMEOUT_SECONDS: int = _env_int("NETCLEAVE_TIMEOUT_SECONDS", 300)

    # --- N-glicosilacion (StackGlyEmbed LOCAL, venv dedicado, subprocess puro) ---
    # Instalacion propia (venv .venv-stackglyembed dentro de StackGlyEmbed/, con
    # torch/xgboost/sklearn/transformers/tensorflow y ProteinBERT instalado via
    # 'pip install git+.../protein_bert.git'). Los 3 embedders que consume
    # (ProteinBERT, ESM-2 650M, ProtT5) cargan 100% offline una vez cacheados
    # (ver docstring de 'StackGlyEmbed/prediction/predict_local.py'): ProtT5
    # se REUSA de los pesos ya descargados para TMbed (mismo encoder,
    # Rostlab/prot_t5_xl_half_uniref50-enc), ESM-2 650M y el dump de
    # ProteinBERT (~/proteinbert_models/default.pkl) se descargaron una
    # unica vez como paso de SETUP.
    STACKGLYEMBED_PYTHON_BIN: str = _env_str(
        "STACKGLYEMBED_PYTHON_BIN",
        "/home/enzo/DiffSBDD/B-Cell-Epitope-Prediction/StackGlyEmbed/.venv-stackglyembed/bin/python",
    )
    # A diferencia de los demas motores, el script NO vive dentro del clon
    # externo ('StackGlyEmbed/', ignorado por git): es codigo propio, ver
    # docstring de 'stackglyembed_predict_local.py' para la razon (repo
    # anidado, git no permite des-ignorar un archivo adentro).
    STACKGLYEMBED_SCRIPT_PATH: str = _env_str(
        "STACKGLYEMBED_SCRIPT_PATH",
        str(Path(__file__).resolve().parent.parent / "engines" / "stackglyembed_predict_local.py"),
    )
    # Carpeta 'prediction/' del clon externo: aqui SI viven los pickles del
    # clasificador ya entrenado (power_transformer_*.sav, base_layer_pickle_files/).
    STACKGLYEMBED_MODELS_DIR: str = _env_str(
        "STACKGLYEMBED_MODELS_DIR",
        "/home/enzo/DiffSBDD/B-Cell-Epitope-Prediction/StackGlyEmbed/prediction",
    )
    STACKGLYEMBED_T5_MODEL_PATH: str = _env_str(
        "STACKGLYEMBED_T5_MODEL_PATH",
        "/home/enzo/DiffSBDD/scipion-chem-tmbed/tmbed_src/tmbed/models/t5",
    )
    STACKGLYEMBED_ESM_MODEL_NAME: str = _env_str("STACKGLYEMBED_ESM_MODEL_NAME", "facebook/esm2_t33_650M_UR50D")
    # Generoso por defecto: carga en frio de 3 modelos (ProteinBERT + ESM-2
    # 650M + ProtT5) sobre CPU antes de procesar el primer sitio.
    STACKGLYEMBED_TIMEOUT_SECONDS: int = _env_int("STACKGLYEMBED_TIMEOUT_SECONDS", 900)

    # --- Cruce con bnAb conocidos (LANL Immunology DB + CATNAP, pandas puro, sin red) ---
    # Reemplaza a bNAber (dominio muerto, ver docstring de lanl_catnap_engine.py).
    # No requiere venv aparte: pandas ya esta en el entorno que corre pipeline.py.
    LANL_AB_ALL_PATH: str = _env_str(
        "LANL_AB_ALL_PATH",
        "/home/enzo/DiffSBDD/B-Cell-Epitope-Prediction/reference_db/lanl_immunology/ab_all.csv",
    )
    CATNAP_ABS_PATH: str = _env_str(
        "CATNAP_ABS_PATH",
        "/home/enzo/DiffSBDD/B-Cell-Epitope-Prediction/reference_db/catnap/abs_2026-07-01.txt",
    )
    LANL_CATNAP_MIN_OVERLAP: int = _env_int("LANL_CATNAP_MIN_OVERLAP", 6)

    @classmethod
    def setup_directories(cls) -> None:
        """Crea los directorios de datos requeridos si aun no existen."""
        cls.BEPIPRED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.EPIDOPE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.FASTA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.FASTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DISCOTOPE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DISCOTOPE_WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.SCANNET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
