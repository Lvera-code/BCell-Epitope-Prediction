"""Jerarquia de excepciones especificas del pipeline para control de flujo."""


class PipelineError(Exception):
    """Clase base para todos los errores controlados del pipeline."""


class InvalidSequenceError(PipelineError):
    """Una secuencia FASTA individual tiene residuos invalidos o longitud insuficiente.

    Este error es recuperable a nivel de registro: el modulo de saneamiento lo captura
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


class EpidopeExecutionError(EngineExecutionError):
    """Fallo al ejecutar EpiDope localmente (Fase 2, via subprocess/conda run).

    Cubre tanto el entorno conda dedicado ausente (paquete open-source MIT,
    ver ``Settings.EPIDOPE_DOWNLOAD_URL``) como fallos del propio subproceso
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


class DiscoTopeExecutionError(EngineExecutionError):
    """Fallo al ejecutar DiscoTope-3.0 localmente (Fase 2, motor estructural).

    Cubre entorno ``.venv-discotope`` ausente/incompleto, pesos de ESM-IF1 no
    cacheados (ver ``Settings.DISCOTOPE_WEIGHTS_CACHE_DIR``) y fallos del
    propio subproceso (exit code distinto de cero, timeout, formato de salida
    inesperado), traducidos a un mensaje accionable.
    """


class ScanNetExecutionError(EngineExecutionError):
    """Fallo al ejecutar ScanNet localmente (Fase 2, motor estructural).

    Cubre ambos runtimes (``Settings.SCANNET_RUNTIME``): entorno
    ``.venv-scannet`` ausente para el runtime ``venv``, o daemon/imagen Docker
    ausente para el runtime ``docker``. Tambien cubre fallos del propio
    proceso (exit code distinto de cero, timeout, formato de salida
    inesperado).
    """


class StructureParsingError(PipelineError):
    """Fallo al parsear una estructura de entrada (PDB/mmCIF) en Fase 1.5.

    Cubre archivos corruptos o sin sintaxis valida, ausencia de cualquier
    cadena proteica con al menos un residuo de aminoacido valido en la
    estructura, y fallos al resolver la estrategia de seleccion de cadena
    configurada (``Settings.PDB_CHAIN_SELECTION_STRATEGY``). Es fatal: no hay
    manera segura de continuar sin una cadena de referencia valida.
    """


class InputRoutingError(PipelineError):
    """Fallo al determinar el tipo de un archivo de entrada, o combinacion no soportada.

    Cubre tanto un archivo cuyo tipo (FASTA vs estructura) no puede
    determinarse con confianza por extension ni por contenido, como una
    combinacion ``input_type``/``PDB_PROCESSING_MODE`` sin motores definidos
    en ``ENGINE_REGISTRY`` (ver ``src.engines.engine_registry.active_engines_for``).
    Es fatal: detiene el pipeline antes de correr cualquier fase.
    """
