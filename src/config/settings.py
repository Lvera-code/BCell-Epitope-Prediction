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

    # Nombre del CSV de salida crudo que escribe BepiPred-3.0 localmente
    # (confirmado leyendo bp3/bepipred3.py::create_csvfile en el codigo fuente
    # oficial: columnas 'Accession,Residue,BepiPred-3.0 score,...').
    BEPIPRED_RAW_OUTPUT_FILENAME: str = _env_str("BEPIPRED_RAW_OUTPUT_FILENAME", "raw_output.csv")

    BEPIPRED_OUTPUT_DIR: Path = Path(_env_str("BEPIPRED_OUTPUT_DIR", "produccion_resultados/bepipred3"))

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
    # E-value elevado deliberadamente: para peptidos cortos (9-25 aa) el
    # e-value por defecto de blastp (10) es demasiado estricto y descarta
    # hits reales de alta identidad (guia estandar de NCBI para 'blastp-short').
    BLAST_EVALUE: float = _env_float("BLAST_EVALUE", 200.0)
    # Umbral de longitud (aa) que decide el algoritmo de BLASTp por peptido:
    # < este valor -> '-task blastp-short' (word_size/matriz ajustados para
    # secuencias cortas); >= este valor -> '-task blastp' (algoritmo estandar).
    BLAST_SHORT_PEPTIDE_MAX_LEN: int = _env_int("BLAST_SHORT_PEPTIDE_MAX_LEN", 30)

    # --- Fase 5: Presentacion celular / Inmunogenicidad ---
    DEFAULT_INMUNO_METHOD: str = _env_str("DEFAULT_INMUNO_METHOD", "netmhcpan")
    DEFAULT_ALLELE: str = _env_str("DEFAULT_ALLELE", "HLA-A02:01")
    IC50_THRESHOLD: float = _env_float("IC50_THRESHOLD", 500.0)

    @classmethod
    def setup_directories(cls) -> None:
        """Crea los directorios de datos requeridos si aun no existen."""
        cls.BEPIPRED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.FASTA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.FASTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
