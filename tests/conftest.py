"""Configuracion global de la suite de tests: sys.path y modo offline forzado."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest

from src.config.settings import Settings


@pytest.fixture(autouse=True, scope="session")
def _force_offline_mode():
    """Fuerza modo offline (sin llamadas de red a HuggingFace Hub) para toda la suite.

    El modelo ESM-2 usado por la Fase 2 ya esta cacheado localmente; forzar
    ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` evita que la suite dependa de
    conectividad de red (inestable/restringida en CI) y acelera la carga.
    """
    Settings.apply_offline_mode()
    Settings.apply_thread_limits()
