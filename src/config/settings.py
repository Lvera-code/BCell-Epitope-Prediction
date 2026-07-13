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

    @classmethod
    def setup_directories(cls) -> None:
        """Crea los directorios de datos requeridos si aun no existen."""
        cls.BEPIPRED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.EPIDOPE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.FASTA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.FASTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
