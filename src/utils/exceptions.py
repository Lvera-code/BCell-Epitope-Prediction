"""Excepciones específicas del pipeline para control de flujo granular."""


class PipelineError(Exception):
    """Clase base para errores del pipeline."""
    pass


class InvalidSequenceError(PipelineError):
    """Se lanza cuando una secuencia FASTA tiene residuos inválidos o longitud insuficiente."""
    pass


class EngineExecutionError(PipelineError):
    """Error recuperable durante el cálculo de una secuencia individual."""
    pass


class ModelLoadError(PipelineError):
    """Error fatal al cargar los pesos de ESM-2 o recursos de red (detiene el pipeline)."""
    pass