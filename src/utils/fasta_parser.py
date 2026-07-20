"""Fase 1: Saneamiento de FASTA crudo.

Responsabilidad exclusiva: leer un FASTA tal como llega del usuario, separarlo
en registros (cabecera, secuencia) y producir una version saneada -mayusculas,
sin saltos de linea internos- que sea segura para enviar a BepiPred (Fase 2),
a BLASTp (Fase 4) y a los motores de inmunogenicidad (Fase 5). El descarte de
un registro individual genuinamente vacio es recuperable (ver
``InvalidSequenceError``); un FASTA sin ningun '>', un registro con residuos
no canonicos o dos registros con el mismo accession son errores fatales (ver
``FastaFormatError``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.utils.exceptions import FastaFormatError, InvalidSequenceError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Alfabeto canonico de los 20 aminoacidos estandar. Cualquier otro caracter
# (ambiguedades IUPAC como X/B/Z/J/U/O, gaps '-', stops '*', digitos, etc.) se
# rechaza en el saneamiento en vez de eliminarse o sustituirse (ver
# `sanitize_sequence`).
CANONICAL_AMINOACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class FastaRecord:
    """Un registro FASTA ya saneado, listo para las fases siguientes."""

    header: str
    accession: str
    sequence: str


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    """Separa un archivo FASTA crudo en pares ``(cabecera, secuencia_cruda)``.

    Args:
        path: Ruta al archivo FASTA de entrada.

    Returns:
        Lista de tuplas ``(header_sin_'>', secuencia_sin_saltos_de_linea)``
        en el mismo orden en que aparecen en el archivo.

    Raises:
        FileNotFoundError: Si ``path`` no existe.
        FastaFormatError: Si el archivo no contiene ningun registro valido
            (no empieza con '>' o esta vacio). Es un error fatal.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el archivo FASTA de entrada: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not lines or not lines[0].startswith(">"):
        raise FastaFormatError(
            f"'{path.name}' no cumple la sintaxis FASTA minima (debe iniciar con '>')."
        )

    records: List[Tuple[str, str]] = []
    header: str = ""
    seq_chunks: List[str] = []

    for line in lines:
        if line.startswith(">"):
            if header:
                records.append((header, "".join(seq_chunks)))
            header = line[1:].strip()
            seq_chunks = []
        else:
            seq_chunks.append(line)

    if header:
        records.append((header, "".join(seq_chunks)))

    if not records:
        raise FastaFormatError(f"'{path.name}' no contiene ningun registro FASTA valido.")

    return records


def sanitize_sequence(raw_sequence: str) -> Tuple[str, List[str]]:
    """Normaliza a mayusculas y detecta (sin corregir) residuos no canonicos.

    Deliberadamente NO elimina ni sustituye los caracteres no canonicos
    (ambiguedades IUPAC como X/B/Z/J, selenocisteina U, pirrolisina O, gaps
    '-', stops '*', digitos, etc.): borrarlos desplaza la numeracion de
    posicion y fusiona residuos que en la proteina real no son vecinos,
    pudiendo fabricar un "epitopo" quimerico en la costura (confirmado
    empiricamente con GPX1_HUMAN real, selenoproteina con 'U' en su secuencia
    canonica de UniProt: la version anterior de este saneamiento borraba la
    'U' y esa costura fabricada fue marcada como epitopo). Sustituirlos por
    un marcador tampoco es viable: BepiPred-3.0 rechaza en bloque (exit code
    1) cualquier caracter fuera de los 20 aminoacidos estandar, incluida 'X'
    (confirmado empiricamente). La unica opcion robusta es rechazar el
    registro por completo (ver :func:`load_and_sanitize`) y que sea el
    investigador quien decida como resolver ese residuo antes de correr el
    pipeline.

    Args:
        raw_sequence: Secuencia de aminoacidos sin procesar.

    Returns:
        Tupla ``(secuencia_en_mayusculas, caracteres_no_canonicos_encontrados)``,
        con la lista de caracteres invalidos en orden de aparicion (vacia si
        la secuencia es 100% canonica).
    """
    upper = raw_sequence.upper()
    invalid_chars = [c for c in upper if c not in CANONICAL_AMINOACIDS]
    return upper, invalid_chars


def is_bepipred_compatible(sequence: str) -> Tuple[bool, List[str]]:
    """Indica si ``sequence`` puede enviarse a BepiPred-3.0/EpiDope sin ser rechazada.

    Reutiliza el mismo alfabeto canonico que ``load_and_sanitize`` (nunca se
    relaja: confirmado empiricamente -reintentado el 2026-07-20 contra la
    instalacion local real- que BepiPred-3.0 sigue rechazando en bloque,
    exit code 1, cualquier caracter fuera de los 20 aminoacidos estandar,
    incluida 'X'). Pensada para el Camino 3 (PDB -> FASTA derivado, ver
    ``src.utils.structure_parser``): a diferencia de un FASTA subido por el
    usuario (Camino 1, donde un residuo no canonico sigue siendo fatal via
    ``load_and_sanitize``), un residuo no mapeable en la extraccion ATMSEQ de
    una estructura ('X') no debe abortar todo el pipeline -solo excluir esa
    accession de BepiPred/EpiDope, dejando correr igual los motores
    estructurales (DiscoTope-3.0/ScanNet) sobre el PDB original-.

    Returns:
        Tupla ``(compatible, caracteres_no_canonicos_unicos_ordenados)``.
    """
    _, invalid_chars = sanitize_sequence(sequence)
    return not invalid_chars, sorted(set(invalid_chars))


def load_and_sanitize(path: Path) -> List[FastaRecord]:
    """Lee y sanea un FASTA completo, descartando registros vacios y rechazando residuos no canonicos.

    Args:
        path: Ruta al archivo FASTA de entrada (dentro de ``fasta_inputs/``).

    Returns:
        Lista de :class:`FastaRecord` saneados.

    Raises:
        FastaFormatError: Si el archivo no tiene sintaxis FASTA valida (fatal);
            si algun registro contiene residuos no canonicos (fatal: ver
            :func:`sanitize_sequence` para por que no se eliminan ni se
            sustituyen automaticamente); o si dos o mas registros comparten
            el mismo accession (primer token de la cabecera). Este ultimo
            caso es fatal por diseno: las Fases 3 y 4 agrupan todo por
            accession (``groupby``) asumiendo que identifica una unica
            secuencia fisica; si dos proteinas distintas comparten accession,
            esa agrupacion las fusiona en una cadena unica y una ventana de
            epitopo puede caer exactamente sobre la costura entre ambas,
            fabricando un peptido quimerico que no existe en ninguna de las
            dos proteinas reales (confirmado empiricamente). Se detecta y se
            detiene aqui, antes de que el resto del pipeline corra.
        InvalidSequenceError: Si NINGUN registro tiene secuencia (fatal a
            nivel de archivo). El descarte de registros individuales
            genuinamente vacios solo se loggea como warning y no detiene el
            resto del lote.
    """
    raw_records = parse_fasta(path)
    sane_records: List[FastaRecord] = []

    for header, raw_seq in raw_records:
        if not raw_seq:
            logger.warning("Registro '%s' descartado: no tiene ninguna secuencia asociada.", header)
            continue

        upper_seq, invalid_chars = sanitize_sequence(raw_seq)
        if invalid_chars:
            raise FastaFormatError(
                f"Registro '{header}' en '{path.name}' contiene {len(invalid_chars)} residuo(s) no "
                f"canonico(s) ({sorted(set(invalid_chars))}): BepiPred-3.0 rechaza en bloque cualquier "
                "caracter fuera de los 20 aminoacidos estandar (ambiguedades IUPAC X/B/Z/J, "
                "selenocisteina U, pirrolisina O, gaps '-', stops '*', digitos, etc. no estan "
                "soportados). Sustituye manualmente ese residuo por su mejor aproximacion canonica "
                "(o elimina el registro) en el FASTA de entrada y vuelve a intentarlo."
            )

        accession = header.split()[0] if header else "UNKNOWN"
        if "/" in accession or "\\" in accession:
            sane_accession = accession.replace("/", "_").replace("\\", "_")
            logger.warning(
                "Accession '%s' en '%s' contiene un separador de ruta ('/' o '\\'): "
                "renombrado a '%s'. Motivo: BepiPred-3.0 construye el path de sus "
                "encodings ESM-2 concatenando el accession crudo con el operador '/' "
                "de pathlib (bepipred3.py::get_esm2_represention_on_accs_seqs), asi que "
                "cualquier '/' en el accession crea un subdirectorio inesperado que "
                "nunca se crea con mkdir y hace fallar el subproceso con "
                "'Parent directory ... does not exist' (confirmado empiricamente).",
                accession, path.name, sane_accession,
            )
            accession = sane_accession
        sane_records.append(FastaRecord(header=header, accession=accession, sequence=upper_seq))

    if not sane_records:
        raise InvalidSequenceError(f"'{path.name}' no contiene ningun registro con secuencia.")

    accession_counts: dict = {}
    for record in sane_records:
        accession_counts[record.accession] = accession_counts.get(record.accession, 0) + 1
    duplicates = sorted(acc for acc, count in accession_counts.items() if count > 1)
    if duplicates:
        raise FastaFormatError(
            f"'{path.name}' contiene registros con accession duplicado: {duplicates}. "
            "Cada accession debe identificar una unica secuencia fisica (las Fases 3/4 agrupan "
            "por accession); renombra las cabeceras duplicadas en el FASTA de entrada y vuelve a "
            "intentarlo."
        )

    return sane_records


def write_fasta(records: List[FastaRecord], out_path: Path, line_width: int = 60) -> None:
    """Escribe una lista de :class:`FastaRecord` saneados como FASTA valido.

    Escribe unicamente ``record.accession`` (primer token de la cabecera
    original, sin espacios) como cabecera, DESCARTANDO el resto de la
    descripcion libre. Motivo (ver tambien ``src.engines.consensus``):
    BepiPred-3.0 escribe la cabecera CRUDA, sin escapar, en su propio
    ``raw_output.csv`` separado por comas -una descripcion con comas (ej.
    ``'Tetanus toxin, fragment C (Hc domain, residues 865-1315)'``) corrompe
    silenciosamente el parseo de esa columna (pandas absorbe los fragmentos
    de mas como un MultiIndex implicito, dejando un accession irreconocible
    pero igual para todas las filas: los scores por residuo NO se corrompen,
    solo la etiqueta de accession). EpiDope, por su parte, solo conserva el
    primer token separado por espacio como ID de gen (``--idpos 0 --delim
    ' '``). Escribir unicamente el primer token evita ambos problemas de raiz
    y garantiza que BepiPred y EpiDope reporten el MISMO accession para la
    misma secuencia, requisito para el cruce de consenso en Fase 3.

    Args:
        records: Registros a escribir, en orden.
        out_path: Ruta de salida (se sobreescribe si ya existe).
        line_width: Ancho de linea para el envoltorio de la secuencia.
    """
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(f">{record.accession}\n")
            seq = record.sequence
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")
