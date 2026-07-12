"""Configuracion centralizada del sistema de logging del pipeline.

Provee un logger formal (consola + archivo rotativo) para cada modulo, evitando
handlers duplicados en ejecuciones repetidas dentro del mismo proceso.
"""

import logging
import logging.handlers
from pathlib import Path

LOG_DIR: Path = Path("logs")
LOG_FILE: Path = LOG_DIR / "hts_pipeline.log"
MAX_BYTES: int = 5 * 1024 * 1024
BACKUP_COUNT: int = 3


def setup_logger(name: str = "HTS_Pipeline") -> logging.Logger:
    """Crea o recupera un logger formal con salida dual (consola + archivo rotativo).

    Args:
        name: Nombre jerarquico del logger. Se recomienda usar ``__name__`` del
            modulo invocante para que la columna de modulo en el log sea precisa.

    Returns:
        Instancia de ``logging.Logger`` configurada de forma idempotente.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Consola: solo WARNING en adelante. Los mensajes INFO (comandos de
        # subprocess ejecutados, detalle de cada fase) quedan solo en el
        # archivo de log, para no saturar la salida normal del pipeline.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            logger.warning(
                "No se pudo inicializar el archivo de log rotativo en '%s'. "
                "Se continua solo con salida por consola.",
                LOG_FILE,
            )

        logger.propagate = False

    return logger
