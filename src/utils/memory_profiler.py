"""Perfilado de memoria implicito para ejecuciones prolongadas en CPU sin GPU.

Este modulo no depende de librerias externas (``psutil`` incluido): usa
``resource.getrusage`` de la libreria estandar de Python para medir el pico de
memoria residente (RSS) del proceso. Se invoca tras cada lote de inferencia en
los motores de Fase 1 y Fase 2 para detectar tendencias de crecimiento de RAM
antes de que deriven en un OOM (Out-Of-Memory) en ejecuciones de horas de
duracion sobre lotes masivos (HTS).
"""

import gc
import logging
import resource
from typing import Optional


def current_rss_mb() -> float:
    """Devuelve el pico de memoria residente (RSS) del proceso actual en MiB.

    En Linux, ``ru_maxrss`` se reporta en KiB; se convierte a MiB para lectura
    humana en los logs de auditoria de memoria.

    Returns:
        Pico de memoria residente del proceso, en mebibytes (MiB).
    """
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_kb / 1024.0


def log_memory_checkpoint(logger: logging.Logger, label: str, force_gc: bool = True) -> float:
    """Registra un punto de control de memoria y ejecuta recoleccion de basura explicita.

    Se debe invocar inmediatamente despues de liberar (``del``) los tensores
    intermedios de un lote de inferencia, para que ``gc.collect()`` pueda
    reclamar la memoria antes de procesar el siguiente lote.

    Args:
        logger: Logger del modulo invocante, usado para emitir el checkpoint.
        label: Etiqueta descriptiva del punto del pipeline (p. ej. "batch 3/50").
        force_gc: Si es ``True`` (por defecto), invoca ``gc.collect()`` antes de
            medir, garantizando una lectura de RSS post-liberacion representativa.

    Returns:
        El valor de RSS medido en MiB, para permitir alertas de umbral por el
        llamador si asi lo requiere.
    """
    if force_gc:
        gc.collect()

    rss_mb = current_rss_mb()
    logger.debug("Checkpoint de memoria [%s]: RSS=%.1f MiB", label, rss_mb)
    return rss_mb


def warn_if_over_budget(
    logger: logging.Logger, label: str, rss_mb: float, budget_mb: Optional[float] = None
) -> None:
    """Emite una advertencia si el RSS medido supera el presupuesto de memoria.

    Args:
        logger: Logger del modulo invocante.
        label: Etiqueta descriptiva del punto del pipeline.
        rss_mb: Valor de RSS en MiB, tipicamente obtenido de ``log_memory_checkpoint``.
        budget_mb: Presupuesto maximo tolerado en MiB. Si es ``None``, no se
            realiza ninguna comprobacion (util cuando el llamador no tiene un
            presupuesto configurado para ese punto especifico).
    """
    if budget_mb is not None and rss_mb > budget_mb:
        logger.warning(
            "Uso de memoria en '%s' (%.1f MiB) supera el presupuesto configurado "
            "(%.1f MiB). Considere reducir el tamano de lote dinamico.",
            label,
            rss_mb,
            budget_mb,
        )
