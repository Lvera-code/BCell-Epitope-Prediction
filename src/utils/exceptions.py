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


class BepiPredExecutionError(EngineExecutionError):
    """Fallo al ejecutar BepiPred-3.0 localmente (Fase 2, via subprocess).

    Cubre tanto la instalacion local ausente (paquete de codigo fuente con
    licencia academica DTU Health Tech no descargado, ver
    ``Settings.BEPIPRED_DOWNLOAD_URL``) como fallos del propio subproceso
    (exit code distinto de cero, timeout, formato de salida inesperado),
    traducidos a un mensaje accionable en vez de un ``FileNotFoundError`` o
    una traza cruda de ``subprocess``.
    """


class DatasetPrepError(PipelineError):
    """Fallo durante la curacion del dataset de entrenamiento (IEDB/UniProt).

    Cubre errores de red irrecuperables tras agotar reintentos y respuestas de
    API con un volumen de datos insuficiente para curar un dataset balanceado.
    """


class BlastExecutionError(EngineExecutionError):
    """Fallo al ejecutar el filtro de tolerancia inmunologica (Fase 4, BLASTp local).

    Cubre binario 'blastp' ausente del PATH, base de datos local no encontrada
    (proteoma humano sin indexar con makeblastdb) y fallos del propio proceso
    (exit code distinto de cero, timeout).
    """


class ImmunogenicityExecutionError(EngineExecutionError):
    """Fallo al ejecutar la prediccion de presentacion T-helper MHC-II (Fase 5).

    Cubre la instalacion local de NetMHCIIpan-4.3 ausente (paquete con
    licencia academica DTU Health Tech no descargado/instalado, ver
    ``Settings.NETMHCIIPAN_DOWNLOAD_URL``), fallos del propio subproceso
    (exit code distinto de cero, timeout) y formato de salida .xls
    inesperado.
    """
