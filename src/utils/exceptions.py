"""Jerarquia de excepciones especificas del pipeline para control de flujo granular."""


class PipelineError(Exception):
    """Clase base para todos los errores controlados del pipeline."""


class InvalidSequenceError(PipelineError):
    """Una secuencia FASTA individual tiene residuos invalidos o longitud insuficiente.

    Este error es recuperable a nivel de registro: el modulo de aduana lo captura
    internamente y descarta unicamente la secuencia afectada, sin detener el lote.
    """


class FastaFormatError(PipelineError):
    """El archivo FASTA de entrada no cumple la sintaxis minima (sin cabeceras '>').

    Es un error fatal: detiene el pipeline antes de iniciar cualquier fase.
    """


class ModelLoadError(PipelineError):
    """Fallo fatal al cargar pesos, tokenizer o arquitectura de un motor de inferencia.

    Cubre tanto la carga de ESM-2 desde HuggingFace Hub/cache local como la carga
    de pesos entrenados de la 1D-CNN de antigenicidad. Detiene el pipeline.
    """


class EngineExecutionError(PipelineError):
    """Error durante el computo de inferencia de un lote (forward pass, subprocess).

    Recuperable a nivel de lote: se loggea y se propaga para que el orquestador
    decida si continuar con los lotes restantes o abortar.
    """


class CLIWrapperError(EngineExecutionError):
    """Fallo especifico de la ejecucion desacoplada por subprocess (CLIWrapperEngine).

    Cubre codigos de salida distintos de cero, timeouts y salidas no parseables
    del binario externo (p. ej. ``bepipred-cli``).
    """


class DatasetPrepError(PipelineError):
    """Fallo durante la curacion del dataset de entrenamiento (IEDB/UniProt).

    Cubre errores de red irrecuperables tras agotar reintentos y respuestas de
    API con un volumen de datos insuficiente para curar un dataset balanceado.
    """
