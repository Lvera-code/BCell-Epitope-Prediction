"""Fase 5: Prediccion de presentacion T-helper (MHC-II) via NetMHCIIpan-4.3 LOCAL.

ADR - pivote metodologico: MHC-I descartado, luego revertido
--------------------------------------------------------------
Toda la logica de prediccion de presentacion MHC-I (celulas T citotoxicas
CD8+, servida anteriormente por MHCflurry/NetMHCpan) fue eliminada de este
pipeline en una version anterior. La Fase 5 evaluaba exclusivamente
presentacion MHC-II (celulas T-helper CD4+), requisito para activar una
respuesta humoral sostenida (T-B cross-talk) en el diseno de vacunas de
subunidad.

Esta decision se REVIRTIO despues: el scope del proyecto crecio (ver
``netmhcpan_engine.py``) para cubrir tambien evaluacion de inmunogenicidad
CD8+ (MHC-I, NetMHCpan-4.2), como parte del set ampliado de chequeos de
construccion (tox/aller/antigenicidad, N-glico, TM/senal, cross-ref bnAb) mas
alla del pipeline original de 5 fases centrado solo en B-cell/T-helper. La
prediccion MHC-I NO se fusiono dentro de esta Fase 5 (que sigue siendo
exclusivamente MHC-II, con su propio modulo/reporte independiente en
``netmhcpan_engine.py``): la razon original (foco en respuesta humoral, no
citotoxica) sigue siendo valida para el veredicto final de esta fase
especifica, asi que MHC-I se anade como una senal adicional en paralelo en
vez de reemplazar o mezclarse con el criterio de promiscuidad T-helper de
aqui.

Este modulo es, igual que ``blast_engine.py`` y ``bepipred_engine.py``, un
wrapper puro de ``subprocess`` sobre un binario local con licencia academica
DTU Health Tech (``Settings.NETMHCIIPAN_HOME``, nunca hardcodeado, resuelto
desde variable de entorno). No se usa ``requests`` ni ninguna llamada de red.

Promiscuidad HLA-II: en vez de evaluar un unico alelo (insuficiente para
cobertura poblacional), cada peptido candidato se evalua contra
``IEDB_REFERENCE_PANEL`` -el panel de referencia de 27 alelos HLA-DR/DQ/DP
mas representativos de la poblacion (IEDB) para estimar cobertura amplia-
pasado tal cual al flag ``-a`` de NetMHCIIpan. Un peptido se reporta como
``'Candidato Valido'`` (T-helper promiscuo) solo si clasifica como
aglutinador fuerte (SB) o debil (WB), segun los umbrales de %Rank POR
DEFECTO del propio NetMHCIIpan-4.3, en al menos
``Settings.NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES`` alelos distintos del panel
CON REGISTRO DE UNION EN ORIENTACION NORMAL.

Fiabilidad para sintesis/validacion experimental - alelos invertidos se
DESCARTAN por completo, no solo del veredicto: NetMHCIIpan marca
``Inverted=1`` en un alelo cuando su
procedimiento de alineacion del core (entrenado por Gibbs sampling sobre
datos de eluido por espectrometria de masas) ajusta mejor leyendo el
peptido en reversa que en su sentido N->C real. No hay evidencia
estructural de que MHC-II presente peptidos "al reves"; se trata como un
artefacto/limitacion del alineador, de menor confianza que un ajuste en
orientacion normal. Como este pipeline alimenta sintesis y validacion
experimental (costosa e irreversible una vez lanzada), un alelo invertido
se trata como si NO existiera para efectos de reporte: no cuenta para la
promiscuidad, y tampoco puede ser el alelo "ganador" que determina
``core_9aa``/``min_rank_el`` (se excluye de ese calculo desde el origen,
no se calcula-y-descarta despues). Asi, todo numero que llega a la tabla
final (consola y CSV) es, por construccion, de un alelo en orientacion
normal; no hace falta exponer un desglose normal/invertido en la salida
porque el invertido nunca entra a la cuenta.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import ImmunogenicityExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

# Panel de referencia de 27 alelos HLA-DR/DQ/DP mas representativos usado
# por el IEDB para estimar cobertura poblacional amplia en el diseno de
# epitopos T-helper (MHC-II). NUNCA se le agregan espacios entre comas:
# NetMHCIIpan lo pasa tal cual a su parser de '-a' y un espacio rompe el
# parseo del alelo siguiente.
IEDB_REFERENCE_PANEL = (
    "DRB1_0101,DRB1_0301,DRB1_0401,DRB1_0405,DRB1_0701,DRB1_0802,DRB1_0901,"
    "DRB1_1101,DRB1_1201,DRB1_1302,DRB1_1501,DRB3_0101,DRB3_0202,DRB4_0101,DRB5_0101,"
    "HLA-DQA10501-DQB10201,HLA-DQA10501-DQB10301,HLA-DQA10301-DQB10302,"
    "HLA-DQA10401-DQB10402,HLA-DQA10101-DQB10501,HLA-DQA10102-DQB10602,"
    "HLA-DPA10201-DPB10101,HLA-DPA10103-DPB10201,HLA-DPA10103-DPB10401,"
    "HLA-DPA10301-DPB10402,HLA-DPA10201-DPB10501,HLA-DPA10201-DPB11401"
)

# Formato de alelo aceptado por el flag '-a' de NetMHCIIpan-4.3 para
# HLA-DR/DQ/DP humano (los 3 patrones que efectivamente aparecen en
# IEDB_REFERENCE_PANEL, verificado: los 27 alelos del panel matchean este
# regex sin excepcion):
#   * DRB1/3/4/5: 'DRB[1345]_' + 4 digitos, ej. 'DRB1_0101'.
#   * DQ:  'HLA-DQA1' + 4 digitos + '-DQB1' + 4 digitos, ej. 'HLA-DQA10501-DQB10201'.
#   * DP:  'HLA-DPA1' + 4 digitos + '-DPB1' + 4 digitos, ej. 'HLA-DPA10201-DPB10101'.
_ALLELE_PATTERN = re.compile(r"(DRB[1345]_\d{4}|HLA-DQA1\d{4}-DQB1\d{4}|HLA-DPA1\d{4}-DPB1\d{4})")


def validate_allele_extra(value: str) -> str:
    """Valida el formato de un string de alelos HLA-DR/DQ/DP adicionales (``--alelo-extra``).

    Pensado para usarse como validacion TEMPRANA (p. ej. como ``type=`` de
    ``argparse`` en ``pipeline.py``), antes de correr ninguna fase del
    pipeline: sin esto, un alelo mal escrito recien se detecta al final de
    la Fase 5 (al parsear el ``.xls`` de NetMHCIIpan, ver ``_parse_xls``),
    despues de haber corrido BepiPred/EpiDope/BLASTp para nada.

    Args:
        value: String de alelos separados por coma SIN espacios (formato
            NetMHCIIpan), ej. ``"DRB1_1602"`` o
            ``"DRB1_1602,HLA-DQA10501-DQB10201"``.

    Returns:
        ``value`` sin modificar, si es valido (permite usarlo directo como
        ``type=`` de ``argparse.add_argument``).

    Raises:
        ValueError: Si ``value`` esta vacio, tiene espacios, tiene tokens
            vacios (comas dobles o al inicio/final), o algun alelo no seria
            aceptado por NetMHCIIpan-4.3 (ver ``_ALLELE_PATTERN``). El
            mensaje incluye el/los token(s) invalidos y el formato esperado.
    """
    if not value or not value.strip():
        raise ValueError("--alelo-extra no puede ser una cadena vacia.")
    if " " in value:
        raise ValueError(
            f"--alelo-extra ('{value}') no puede contener espacios: NetMHCIIpan pasa el "
            "string tal cual a su flag '-a' y un espacio rompe el parseo del alelo "
            "siguiente. Separa los alelos solo con comas, ej. 'DRB1_1602,DRB1_1301'."
        )

    tokens = value.split(",")
    invalid = [t for t in tokens if not _ALLELE_PATTERN.fullmatch(t)]
    if invalid:
        raise ValueError(
            f"--alelo-extra contiene {len(invalid)} alelo(s) con formato invalido: "
            f"{invalid}. Formatos aceptados por NetMHCIIpan-4.3 (HLA-DR/DQ/DP humano): "
            "'DRB1_0101' / 'DRB3_0101' / 'DRB4_0101' / 'DRB5_0101' (4 digitos tras el "
            "guion bajo), 'HLA-DQA10501-DQB10201' o 'HLA-DPA10201-DPB10101' (4 digitos "
            "en cada bloque). Ejemplo completo valido: 'DRB1_1602,HLA-DQA10501-DQB10201'."
        )
    return value

# Footprint minimo del core de union a MHC-II: NetMHCIIpan descarta (o
# calcula sobre un core mas corto que el peptido, degradando la prediccion)
# peptidos mas cortos que esto.
_MIN_PEPTIDE_LENGTH = 9

# Longitud maxima segura para el modo peptido exacto ('-p', -inptype 1). El
# binario NetMHCIIpan-4.3 (Linux_x86_64) revienta con "*** buffer overflow
# detected ***" (SIGABRT, core dump) en ese modo para entradas demasiado
# largas -contra el panel real de 27 alelos de IEDB_REFERENCE_PANEL: 55 aa
# OK, 56 aa crash, reproducible con contenido
# aleatorio, no depende de la secuencia concreta-. El umbral exacto de
# crash puede variar segun el panel de alelos evaluado (el tamano de ``-a``
# influye en el buffer interno del binario), asi que este numero es valido
# especificamente para el panel de 27 alelos que usa este pipeline; si se
# cambia el panel de referencia, conviene re-verificarlo. El wrapper
# 'netMHCIIpan' (tcsh) NO propaga ese crash como exit code distinto de cero
# (ver ``_require_xls_output``), asi que sin este enrutamiento el pipeline
# fallaria con un error generico e indescifrable. Peptidos mas largos que
# esto se evaluan en modo proteina (FASTA, sin '-p'): NetMHCIIpan desliza
# internamente una ventana (``-length``, 15 aa por defecto) y evalua todos
# los nucleos de union candidatos dentro del fragmento -el uso correcto de
# la herramienta para fragmentos largos, en vez de tratarlos como un unico
# peptido exacto-. Se deja un margen de seguridad considerable (40, no 55)
# por si el limite real del binario varia entre builds o paneles de alelos.
_MAX_PEPTIDE_MODE_LENGTH = 40

_OUTPUT_COLUMNS = [
    "sequence", "core_9aa", "n_alelos_evaluados", "n_alelos_promiscuos", "min_rank_el", "veredicto",
]

# Columnas fijas del reporte final enriquecido (Fase 5 + traceback a Fase
# 3/4), ver ``build_traceback_report``. Las columnas '{motor}_score' NO estan
# fijas aqui: se detectan dinamicamente desde ``parent_df`` (ver
# ``_traceback_columns``), porque el subconjunto de motores activos -y por
# tanto de columnas '{motor}_score' presentes en la tabla de consenso de Fase
# 3, ver ``src.engines.consensus.build_annotated_union_table``- varia segun
# el camino de entrada (Camino 1: bepipred/epidope; Camino 2: discotope/
# scannet; Camino 3: los 4). Antes de esa generalizacion esta lista incluia
# 'bepipred_score'/'epidope_score' fijos, lo que rompia con un
# ``AttributeError`` en el Camino 2 (sin esas dos columnas en absoluto).
_TRACEBACK_BASE_COLUMNS = [
    "accession", "sequence_f5", "core_9aa", "start", "end", "origen",
    "n_alelos_promiscuos", "n_alelos_evaluados", "min_rank_el",
]


def _traceback_columns(parent_df: pd.DataFrame) -> List[str]:
    """Columnas fijas mas '{motor}_score' por cada motor presente en ``parent_df``."""
    score_columns = [c for c in parent_df.columns if c.endswith("_score")]
    return _TRACEBACK_BASE_COLUMNS + score_columns


def _resolve_binary() -> Path:
    """Localiza el script local de NetMHCIIpan-4.3 y valida que sea ejecutable.

    Raises:
        ImmunogenicityExecutionError: Con instrucciones de instalacion si el
            script no existe o no tiene permiso de ejecucion.
    """
    binary = Settings.NETMHCIIPAN_HOME / Settings.NETMHCIIPAN_BINARY_NAME
    if not binary.is_file():
        raise ImmunogenicityExecutionError(
            f"No se encontro el script local de NetMHCIIpan-4.3 en '{binary}'. Por "
            "restricciones de licencia academica, DTU Health Tech no permite "
            "redistribuir el paquete: descargalo manualmente desde "
            f"{Settings.NETMHCIIPAN_DOWNLOAD_URL} (seccion 'Downloads', requiere "
            "cuenta academica), descomprimelo en la raiz del proyecto como "
            "'netMHCIIpan-4.3/' (o apunta la variable de entorno NETMHCIIPAN_HOME "
            "a su ubicacion), edita la linea 'NMHOME' del script 'netMHCIIpan' con "
            "la ruta absoluta de instalacion (paso manual obligatorio segun el "
            "propio instructivo de DTU) y vuelve a intentarlo. Ver README.md - "
            "Seccion de Instalacion."
        )
    if not os.access(binary, os.X_OK):
        raise ImmunogenicityExecutionError(
            f"El script '{binary}' no tiene permiso de ejecucion. Corre "
            f"'chmod +x {binary}' y vuelve a intentarlo."
        )
    return binary


def _parse_xls(xls_path: Path, n_alleles: int) -> pd.DataFrame:
    """Parsea el .xls de NetMHCIIpan y evalua la promiscuidad de cada peptido.

    El .xls multi-alelo de NetMHCIIpan-4.3 tiene un formato de 2 filas de
    cabecera: una linea de comentario ('#...'), una fila con el nombre de
    cada alelo (una celda por bloque de 4 columnas: Core/Inverted/Score_EL/
    Rank_EL) y la fila real de nombres de columna. ``pandas`` desambigua
    automaticamente 'Rank_EL', 'Core' e 'Inverted' repetidas como 'Rank_EL',
    'Rank_EL.1', ... / 'Core', 'Core.1', ... / 'Inverted', 'Inverted.1', ...
    (una por alelo, en el mismo orden del panel pasado a '-a'); como los tres
    nombres provienen del mismo bloque de 4 columnas por alelo,
    ``rank_cols[i]``, ``core_cols[i]`` e ``inverted_cols[i]`` siempre
    corresponden al mismo alelo.

    Alelos invertidos: se descartan por completo antes de cualquier calculo,
    no solo del veredicto (ver docstring del modulo, seccion "Fiabilidad
    para sintesis/validacion experimental"). Concretamente, un alelo cuyo
    ``Inverted == 1`` se excluye tanto de la busqueda del alelo "ganador"
    (el de menor %Rank, que determina ``core_9aa``/``min_rank_el``) como del
    conteo de ``n_alelos_promiscuos``: es como si ese alelo no hubiera sido
    evaluado. Esto garantiza que ``core_9aa`` SIEMPRE es una subcadena
    literal del peptido de entrada (nunca hace falta revertirlo) y que todo
    numero devuelto proviene de un registro de union en orientacion normal.
    Si TODOS los alelos de una fila estan invertidos, no queda ningun alelo
    normal candidato: ``min_rank_el`` queda en ``inf`` y
    ``n_alelos_promiscuos`` en 0, por lo que la fila sale ``'Rechazado'`` de
    forma natural (nunca se inventa un core a partir de un alelo invertido).

    Args:
        xls_path: Ruta al .xls crudo generado por NetMHCIIpan.
        n_alleles: Numero de alelos evaluados (debe coincidir con el numero
            de columnas 'Rank_EL*'/'Core*'/'Inverted*' encontradas, si no el
            .xls esta corrupto o el panel no se aplico como se esperaba).

    Returns:
        DataFrame con columnas ``sequence``, ``core_9aa``,
        ``n_alelos_evaluados``, ``n_alelos_promiscuos``, ``min_rank_el`` y
        ``veredicto`` (``'Candidato Valido'`` / ``'Rechazado'``). ``core_9aa``
        es el nucleo de union de 9 aa (columna ``Core`` de NetMHCIIpan) del
        alelo EN ORIENTACION NORMAL con el %Rank mas bajo para ese peptido
        (los alelos invertidos quedan fuera de esta busqueda desde el
        origen, ver arriba); ``n_alelos_promiscuos`` cuenta solo alelos
        normales SB/WB, que es tambien el criterio del ``veredicto``.

    Raises:
        ImmunogenicityExecutionError: Si el .xls no se puede parsear o no
            contiene el numero esperado de columnas 'Rank_EL'/'Core'/'Inverted'.
    """
    try:
        raw = pd.read_csv(xls_path, sep="\t", skiprows=2)
    except Exception as exc:
        raise ImmunogenicityExecutionError(f"No se pudo parsear la salida de NetMHCIIpan en '{xls_path}': {exc}") from exc

    rank_cols = [c for c in raw.columns if c == "Rank_EL" or c.startswith("Rank_EL.")]
    core_cols = [c for c in raw.columns if c == "Core" or c.startswith("Core.")]
    inverted_cols = [c for c in raw.columns if c == "Inverted" or c.startswith("Inverted.")]
    if (
        len(rank_cols) != n_alleles
        or len(core_cols) != n_alleles
        or len(inverted_cols) != n_alleles
        or "Peptide" not in raw.columns
    ):
        raise ImmunogenicityExecutionError(
            f"El formato de salida .xls de NetMHCIIpan no coincide con lo esperado: "
            f"se encontraron {len(rank_cols)} columna(s) 'Rank_EL', {len(core_cols)} "
            f"columna(s) 'Core' y {len(inverted_cols)} columna(s) 'Inverted' para "
            f"{n_alleles} alelo(s) evaluado(s). Columnas encontradas: {list(raw.columns)}."
        )

    rank_matrix = raw[rank_cols].to_numpy()
    core_matrix = raw[core_cols].to_numpy()
    is_inverted = raw[inverted_cols].to_numpy().astype(bool)
    row_idx = np.arange(len(raw))

    # Los alelos invertidos se excluyen ANTES de buscar el "ganador": se les
    # asigna %Rank=+inf para que argmin() nunca los elija, asi que
    # ``best_allele_idx`` (y por tanto ``core_9aa``/``min_rank_el``) sale
    # siempre de un alelo en orientacion normal. Si TODOS los alelos de una
    # fila estan invertidos, la fila entera queda en +inf: ``min_rank_el``
    # sale ``inf`` y ``core_9aa`` corresponde a un alelo sin ningun binder
    # real, lo cual es correcto porque esa fila sera 'Rechazado' de todas
    # formas (``n_alelos_promiscuos`` normal tambien sera 0).
    rank_matrix_normal = np.where(is_inverted, np.inf, rank_matrix)
    best_allele_idx = rank_matrix_normal.argmin(axis=1)
    best_core = core_matrix[row_idx, best_allele_idx]

    # Promiscuidad: solo cuentan alelos normales (no invertidos) con
    # Rank_EL <= NETMHCIIPAN_RANK_WEAK. Ver docstring del modulo, seccion
    # "Fiabilidad para sintesis/validacion experimental".
    is_binder_normal = (rank_matrix_normal <= Settings.NETMHCIIPAN_RANK_WEAK)
    n_alelos_promiscuos = is_binder_normal.sum(axis=1)

    result = pd.DataFrame(
        {
            "sequence": raw["Peptide"],
            "core_9aa": best_core,
            "n_alelos_evaluados": n_alleles,
            "n_alelos_promiscuos": n_alelos_promiscuos,
            "min_rank_el": rank_matrix_normal.min(axis=1),
        }
    )
    result["veredicto"] = result["n_alelos_promiscuos"].apply(
        lambda n: "Candidato Valido" if n >= Settings.NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES else "Rechazado"
    )
    return result[_OUTPUT_COLUMNS]


def _run_netmhciipan(
    binary: Path, mode_args: List[str], allele_panel: str, xls_path: Path, timeout: int
) -> subprocess.CompletedProcess:
    """Invoca el binario local con ``mode_args`` (formato de entrada) + panel + salida .xls."""
    cmd = [str(binary)] + mode_args + ["-a", allele_panel, "-xls", "-xlsfile", str(xls_path)]
    logger.info("Ejecutando NetMHCIIpan-4.3 local: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise ImmunogenicityExecutionError(
            f"NetMHCIIpan-4.3 termino con exit code {exc.returncode}: "
            f"{(exc.stderr or '<sin stderr>')[:2000]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ImmunogenicityExecutionError(f"NetMHCIIpan-4.3 excedio el tiempo limite de {timeout}s.") from exc


def _require_xls_output(xls_path: Path, proc: subprocess.CompletedProcess, mode_desc: str) -> None:
    """Valida que el .xls prometido exista: el wrapper tcsh no propaga fallos internos como exit != 0.

    Causas conocidas: (a) modo peptido exacto con una entrada > 55 aa
    (buffer overflow del binario, ver ``_MAX_PEPTIDE_MODE_LENGTH`` -
    ``predict_netmhciipan`` ya enruta para evitar esto-), (b) la linea
    'NMHOME' dentro del script wrapper apunta a una ruta desactualizada
    (p. ej. tras mover/renombrar la carpeta del proyecto).
    """
    if xls_path.is_file():
        return
    raise ImmunogenicityExecutionError(
        f"NetMHCIIpan-4.3 ({mode_desc}) termino sin error (exit 0) pero no genero el archivo "
        f"de salida esperado en '{xls_path}'. Causas conocidas: un peptido de entrada excede "
        f"el limite del modo usado (revisa Settings._MAX_PEPTIDE_MODE_LENGTH), o la linea "
        f"'NMHOME' dentro de '{Settings.NETMHCIIPAN_HOME / Settings.NETMHCIIPAN_BINARY_NAME}' "
        f"apunta a una ruta desactualizada (p. ej. si moviste la carpeta del proyecto) -en ese "
        f"caso, edita esa linea con la ruta absoluta ACTUAL de "
        f"'{Settings.NETMHCIIPAN_HOME.resolve()}' y vuelve a intentarlo-. "
        f"Salida del proceso: {(proc.stdout or '<vacia>')[:1000]}"
    )


def predict_netmhciipan(
    peptides: List[str],
    output_dir: Path,
    allele_panel: str = IEDB_REFERENCE_PANEL,
    filename_prefix: str = "",
) -> pd.DataFrame:
    """Fase 5: evalua promiscuidad T-helper (MHC-II) via NetMHCIIpan-4.3 local.

    Ejecuta, de forma sincrona (``subprocess.run``), el binario local
    ``./netMHCIIpan-4.3/netMHCIIpan`` sobre los peptidos que superaron el
    filtro de tolerancia inmunologica de la Fase 4 (BLASTp, ``status ==
    'Segura'``), contra el panel completo de alelos HLA-DR/DQ/DP indicado.
    Cada peptido se enruta segun su longitud (ver ``_MAX_PEPTIDE_MODE_LENGTH``):

    * ``<= _MAX_PEPTIDE_MODE_LENGTH`` (40 aa) -> modo peptido exacto (``-p``):
      una fila de salida por peptido de entrada, ``sequence`` = el peptido
      completo tal cual.
    * ``> _MAX_PEPTIDE_MODE_LENGTH`` -> modo proteina (FASTA, sin ``-p``):
      NetMHCIIpan desliza internamente una ventana y evalua todos los
      nucleos de union candidatos dentro del fragmento; puede devolver
      VARIAS filas por peptido de entrada, cada una con ``sequence`` = el
      nucleo candidato especifico (mas corto que el fragmento original), no
      el fragmento completo. Es el uso correcto de la herramienta para
      fragmentos largos (el modo peptido exacto revienta con "buffer
      overflow" en el binario para entradas > 55 aa, ver
      ``_MAX_PEPTIDE_MODE_LENGTH``).

    Args:
        peptides: Peptidos candidatos que superaron la Fase 4. Los mas
            cortos que el footprint minimo del core de MHC-II (9 aa) se
            omiten con un warning.
        output_dir: Carpeta donde persistir el/los .xls crudos devueltos por
            NetMHCIIpan, para trazabilidad.
        allele_panel: Alelos HLA-DR/DQ/DP separados por coma sin espacios
            (formato NetMHCIIpan, ej. ``"DRB1_0101,DRB1_0301"`` o
            ``"HLA-DQA10501-DQB10201"``), pasados tal cual al flag ``-a``.
            Por defecto ``IEDB_REFERENCE_PANEL`` (27 alelos); si se
            necesita cubrir un alelo adicional (ej. especifico de una
            poblacion de interes), se admite sin problema anexandolo al
            string por defecto (ver ``--alelo-extra`` en ``pipeline.py``).
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para los
            .xls crudos persistidos en ``output_dir``. Sin esto, dos
            corridas seguidas con inputs distintos pisan el mismo archivo
            (``netmhciipan_raw_peptide_mode.xls``/``..._protein_mode.xls``,
            sin nombre de accession) -- confirmado como una fuente real de
            confusion probando multiples PDBs/FASTA seguidos en la misma
            ``fasta_outputs/``.

    Returns:
        DataFrame con columnas ``sequence``, ``core_9aa``,
        ``n_alelos_evaluados``, ``n_alelos_promiscuos``, ``min_rank_el`` y
        ``veredicto`` (``'Candidato Valido'`` / ``'Rechazado'``). Los alelos
        invertidos ya estan excluidos de todos estos numeros desde
        ``_parse_xls`` (ver docstring del modulo), asi que no aparecen en
        ningun lado de la salida. Vacio si ningun peptido de entrada alcanza
        la longitud minima.

    Raises:
        ImmunogenicityExecutionError: Si el script local no esta instalado o
            no es ejecutable, el subproceso falla/excede el timeout, o el
            formato del .xls de salida no es el esperado.
    """
    binary = _resolve_binary()

    valid_peptides = [p for p in peptides if len(p) >= _MIN_PEPTIDE_LENGTH]
    skipped = len(peptides) - len(valid_peptides)
    if skipped:
        logger.warning(
            "%d peptido(s) mas cortos que el footprint minimo de MHC-II (%d aa) fueron omitidos.",
            skipped, _MIN_PEPTIDE_LENGTH,
        )
    if not valid_peptides:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    short_peptides = [p for p in valid_peptides if len(p) <= _MAX_PEPTIDE_MODE_LENGTH]
    long_peptides = [p for p in valid_peptides if len(p) > _MAX_PEPTIDE_MODE_LENGTH]
    if long_peptides:
        logger.info(
            "%d peptido(s) > %d aa se evaluaran en modo proteina (ventana deslizante interna "
            "de NetMHCIIpan) para evitar el buffer overflow conocido del modo peptido exacto "
            "con entradas largas.",
            len(long_peptides), _MAX_PEPTIDE_MODE_LENGTH,
        )

    n_alleles = len([a for a in allele_panel.split(",") if a])
    output_dir.mkdir(parents=True, exist_ok=True)

    result_frames = []
    with tempfile.TemporaryDirectory(prefix="netmhciipan_") as tmp:
        tmp_dir = Path(tmp)

        if short_peptides:
            pep_path = tmp_dir / "peptides.pep"
            pep_path.write_text("\n".join(short_peptides) + "\n", encoding="utf-8")
            xls_path = tmp_dir / "peptide_mode_output.xls"
            proc = _run_netmhciipan(
                binary, ["-p", "-f", str(pep_path)], allele_panel, xls_path, Settings.NETMHCIIPAN_TIMEOUT_SECONDS
            )
            _require_xls_output(xls_path, proc, mode_desc="modo peptido exacto")
            result_frames.append(_parse_xls(xls_path, n_alleles))
            shutil.copyfile(xls_path, output_dir / f"{filename_prefix}netmhciipan_raw_peptide_mode.xls")

        if long_peptides:
            fasta_path = tmp_dir / "fragments.fasta"
            with fasta_path.open("w", encoding="utf-8") as fh:
                for i, seq in enumerate(long_peptides):
                    fh.write(f">candidato_{i}\n{seq}\n")
            xls_path = tmp_dir / "protein_mode_output.xls"
            proc = _run_netmhciipan(binary, ["-f", str(fasta_path)], allele_panel, xls_path, Settings.NETMHCIIPAN_TIMEOUT_SECONDS)
            _require_xls_output(xls_path, proc, mode_desc="modo proteina (ventana deslizante)")
            result_frames.append(_parse_xls(xls_path, n_alleles))
            shutil.copyfile(xls_path, output_dir / f"{filename_prefix}netmhciipan_raw_protein_mode.xls")

    if not result_frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.concat(result_frames, ignore_index=True)


def print_th_report(report_df: pd.DataFrame, allele_panel: str = IEDB_REFERENCE_PANEL) -> None:
    """Imprime el informe final de promiscuidad T-helper (MHC-II).

    Solo lista los peptidos/ventanas con ``veredicto == 'Candidato Valido'``
    -los rechazados (incluyendo cualquiera que solo hubiera pasado gracias a
    alelos invertidos) ya fueron descartados por ``_parse_xls`` y no aportan
    nada a un reporte pensado para sintesis/validacion experimental-. El
    resumen final si usa el total evaluado como denominador, para que quede
    claro que proporcion del barrido supero el filtro.
    """
    if report_df.empty:
        print("No hay peptidos candidatos de la Fase 4 para evaluar contra el panel HLA-DR.")
        return

    valid_df = report_df[report_df["veredicto"] == "Candidato Valido"]
    n_alleles = len([a for a in allele_panel.split(",") if a])

    if valid_df.empty:
        print("Ningun peptido/ventana supero el umbral de promiscuidad T-helper (ver Resumen).")
    else:
        seq_width = max(20, valid_df["sequence"].str.len().max() + 2)
        columns = [
            Column("Secuencia", lambda r: r.sequence, seq_width, "<"),
            Column("Alelos promiscuos", lambda r: str(r.n_alelos_promiscuos), 19, ">"),
            Column("/", lambda r: "/", 1, ">"),
            Column("panel", lambda r, n=n_alleles: str(n), 7, "<"),
            Column("Min %Rank", lambda r: f"{r.min_rank_el:.3f}", 12, ">"),
        ]
        print_fixed_width_table(valid_df.itertuples(index=False), columns)

    n_ok = len(valid_df)
    print(f"\nResumen Fase 5: {n_ok}/{len(report_df)} candidato(s) T-helper promiscuo(s) aprobado(s).")


def _deduplicate_protein_mode_windows(traceback_df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa ventanas redundantes del modo proteina que no aportan informacion nueva.

    El modo proteina (``predict_netmhciipan``, fragmentos > 40 aa) desliza una
    ventana de 15 aa un residuo a la vez, asi que un mismo nucleo de union de
    9 aa suele "ganar" (ser el alelo de mejor %Rank) en varias ventanas
    consecutivas -no son epitopos distintos, son la misma prediccion vista
    desde offsets vecinos-. Regla de fusion (acordada explicitamente, NO es
    solapamiento de posiciones sino coincidencia EXACTA de dos valores):

    * Se agrupan las filas por ``(accession, core_9aa, n_alelos_promiscuos)``
      -mismo nucleo de 9 aa LETRA POR LETRA y misma cuenta de promiscuidad-.
      De cada grupo con mas de una fila se conserva unicamente la de menor
      ``min_rank_el`` (el aglutinador mas fuerte); las demas se descartan sin
      dejar rastro (no se reconstruye ningun fragmento nuevo, se queda tal
      cual una de las ventanas originales).
    * Si el core difiere aunque sea en 1 aminoacido, o si la promiscuidad
      difiere (aunque el core sea identico), las filas NO se fusionan: cada
      una es una prediccion distinta y ambas se reportan. Esto es deliberado
      para no perder informacion relevante para sintesis/validacion
      experimental: dos ventanas con el mismo core pero distinta
      promiscuidad reflejan una diferencia real en cuantos alelos HLA
      reconocen ese registro exacto de union.

    Args:
        traceback_df: Tabla ya trazada a la Fase 3/4 (columnas de
            ``_traceback_columns``), antes de deduplicar.

    Returns:
        Mismo esquema que ``traceback_df``, con las filas redundantes
        colapsadas. Vacio si ``traceback_df`` esta vacio.
    """
    if traceback_df.empty:
        return traceback_df

    best_idx = traceback_df.groupby(
        ["accession", "core_9aa", "n_alelos_promiscuos"], sort=False
    )["min_rank_el"].idxmin()
    return traceback_df.loc[best_idx].sort_index().reset_index(drop=True)


def build_traceback_report(report_df: pd.DataFrame, parent_df: pd.DataFrame) -> pd.DataFrame:
    """Cruza los 'Candidato Valido' de la Fase 5 con su region de origen en la Fase 3/4.

    En modo proteina (fragmentos > ``_MAX_PEPTIDE_MODE_LENGTH``, ver
    ``predict_netmhciipan``), ``report_df['sequence']`` es un nucleo mas
    corto que el fragmento de la Fase 3/4 del que proviene (NetMHCIIpan lo
    obtuvo deslizando una ventana internamente), asi que no hay una relacion
    1:1 por posicion entre ambas tablas. Para recuperar accession/origen/
    coordenadas reales se busca cada ``sequence`` de la Fase 5 como
    subcadena literal dentro de ``parent_df['sequence']`` (la region completa
    de la union anotada, Fase 3) y se recalcula la posicion absoluta sumando
    el indice del match al ``start`` original de esa region. En modo peptido
    exacto (fragmentos cortos) esta busqueda tambien funciona: el match cae
    en el indice 0 porque ``sequence`` es identica al peptido padre completo.

    Tras el traceback, las ventanas redundantes del modo proteina se colapsan
    via ``_deduplicate_protein_mode_windows`` (ver su docstring para la regla
    exacta de fusion).

    Args:
        report_df: Salida de ``predict_netmhciipan`` (incluye rechazados;
            aqui se filtra a ``veredicto == 'Candidato Valido'``).
        parent_df: Tabla de la Fase 3/4 (``union_df`` o el ``safe_df`` de
            Fase 4, que conserva todas sus columnas): debe tener
            ``accession``, ``start``, ``sequence``, ``origen`` y al menos una
            columna ``'{motor}_score'`` por region (ver
            ``_traceback_columns``: el subconjunto exacto de motores presentes
            varia segun el camino de entrada, no se asume cuales existen).

    Returns:
        DataFrame con columnas ``_TRACEBACK_BASE_COLUMNS`` mas
        ``'{motor}_score'`` por cada motor presente en ``parent_df`` (una
        fila por match candidato-region no redundante; normalmente 1:1, pero
        si el mismo nucleo aparece en mas de una region de ``parent_df`` se
        reporta una fila por cada una en vez de elegir arbitrariamente). Los
        candidatos que no se logran ubicar en ninguna region padre (no
        deberia ocurrir en condiciones normales) se omiten con un warning en
        el log, para no reportar coordenadas inventadas.
    """
    columns = _traceback_columns(parent_df)

    if report_df.empty or parent_df.empty:
        return pd.DataFrame(columns=columns)

    valid_df = report_df[report_df["veredicto"] == "Candidato Valido"]
    if valid_df.empty:
        return pd.DataFrame(columns=columns)

    score_columns = [c for c in columns if c.endswith("_score")]

    records = []
    for candidate in valid_df.itertuples(index=False):
        matches = parent_df[parent_df["sequence"].str.contains(candidate.sequence, regex=False, na=False)]
        if matches.empty:
            logger.warning(
                "No se pudo trazar el candidato '%s' de vuelta a ninguna region de la Fase 3/4; "
                "se omite del reporte final enriquecido.",
                candidate.sequence,
            )
            continue
        for parent in matches.itertuples(index=False):
            offset = parent.sequence.find(candidate.sequence)
            start_real = parent.start + offset
            end_real = start_real + len(candidate.sequence) - 1
            record = {
                "accession": parent.accession,
                "sequence_f5": candidate.sequence,
                "core_9aa": candidate.core_9aa,
                "start": start_real,
                "end": end_real,
                "origen": parent.origen,
                "n_alelos_promiscuos": candidate.n_alelos_promiscuos,
                "n_alelos_evaluados": candidate.n_alelos_evaluados,
                "min_rank_el": candidate.min_rank_el,
            }
            for score_col in score_columns:
                record[score_col] = getattr(parent, score_col)
            records.append(record)

    traceback_df = pd.DataFrame.from_records(records, columns=columns)
    return _deduplicate_protein_mode_windows(traceback_df)


# Resaltado ANSI (negrita + amarillo) del nucleo de 9 aa dentro de la
# columna de secuencia F5 en ``print_traceback_table``. Solo afecta la
# impresion en terminal, nunca el CSV persistido. Un solo color: no hace
# falta distinguir invertidos porque ``core_9aa`` (ver ``_parse_xls``) nunca
# proviene de un alelo invertido -se excluyen desde el calculo, no se
# resaltan de otro color en la salida-.
_CORE_ANSI_START = "\033[1;33m"
_CORE_ANSI_END = "\033[0m"


def _core_span_in_sequence(sequence: str, core: str) -> Optional[Tuple[int, int]]:
    """Ubica el rango ``[start, end)`` de ``sequence`` que corresponde a ``core``.

    ``core`` no siempre es una subcadena LITERAL de ``sequence`` -- NetMHCpan
    lo construye alineando un nucleo teorico de 9 aa contra la ventana real,
    y esa alineacion puede tener:

    * Una insercion marcada con un guion literal (ventanas <9 aa, ej.
      ``FTP-PHGGL`` para la secuencia de 8 aa ``FTPPHGGL``): el guion nunca
      aparece en ``sequence``, asi que buscarlo como substring literal falla.
    * Una delecion SIN marcar (ventanas >9 aa, ej. core ``FFPDHQLAF`` para la
      secuencia de 11 aa ``FFPDHQLDPAF``: NetMHCpan omite 2 residuos del medio
      sin dejar ningun caracter que indique el hueco): tambien falla como
      substring literal, aunque las letras de ``core`` SI aparecen todas en
      ``sequence``, en el mismo orden, solo que no consecutivas.

    En ambos casos se resuelve con el mismo mecanismo: se quita cualquier
    guion de ``core`` y se buscan sus letras como SUBSECUENCIA (mismo orden,
    saltos permitidos) dentro de ``sequence``, de izquierda a derecha sin
    retroceder. El rango devuelto va de la primera a la ultima letra
    calzada, INCLUYENDO los residuos intermedios que el core se salta -- se
    resalta el tramo completo de interaccion, no solo las letras exactas del
    core (mas legible en terminal que un resaltado discontinuo, y para el
    caso de guion el tramo termina siendo ``sequence`` completa, que es
    ademas la lectura correcta: en una ventana de 8 aa el nucleo ocupa
    practicamente todo el peptido).

    Devuelve ``None`` si no se pudo alinear ninguna letra (no deberia pasar
    en la practica: ``core`` siempre proviene del mismo alelo/ventana que
    ``sequence``, nunca de un alelo invertido -ver docstring de clase-, pero
    se maneja por robustez).
    """
    core_letters = core.replace("-", "")
    if not core_letters:
        return None
    positions = []
    pos = 0
    for letter in core_letters:
        idx = sequence.find(letter, pos)
        if idx == -1:
            return None
        positions.append(idx)
        pos = idx + 1
    return positions[0], positions[-1] + 1


def print_traceback_table(traceback_df: pd.DataFrame, require_exact_core: bool = False) -> None:
    """Imprime la tabla de candidatos validos enriquecida con su traceback a la Fase 3.

    El nucleo de union de 9 aa se resalta en color directamente dentro de la
    columna de secuencia F5, para ubicarlo de un vistazo sin cruzar a la
    columna 'Core (9 aa)'.

    El color se inyecta DESPUES de que ``pandas`` ya formateo y alineo toda
    la tabla como texto plano (``to_string()``), nunca antes: los codigos
    ANSI no ocupan espacio visible en la terminal pero SI cuentan para
    ``len()``, asi que insertarlos en las celdas antes de formatear
    desalinea el ancho de columna que calcula pandas (confirmado
    empiricamente: una celda coloreada mas corta que sus vecinas en texto
    plano se ve como la mas larga para pandas, y descoloca toda la columna
    y las que le siguen a la derecha). En vez de eso, se ubica la posicion
    exacta del core dentro de cada linea ya formateada -buscando primero
    dentro de que rango de esa linea cae la secuencia F5, para no confundir
    el match con la propia columna 'Core (9 aa)' si contiene el mismo texto-
    y se inyecta el color ahi, sin anadir ni quitar ningun caracter visible.

    Args:
        require_exact_core: Controla COMO se busca el core dentro de F5, y la
            diferencia tiene una razon biologica real (no es solo un ajuste
            visual):

            * ``False`` (default, Fase 5/MHC-II): usa ``_core_span_in_sequence``
              (subsecuencia, permite huecos). NetMHCIIpan tiene groove ABIERTO
              en los dos extremos -- el core de 9 aa que encuentra es siempre
              un tramo REAL y contiguo dentro de F5 (nunca necesita insertar
              ni borrar residuos, solo elige el mejor registro de 9 aa dentro
              del peptido dado), asi que el resaltado sigue siendo parcial y
              con significado real: lo que queda sin colorear es flanco
              genuino, no un artefacto.
            * ``True`` (Fase 5b/MHC-I): usa ``sequence.find(core)`` LITERAL,
              sin fallback -- NetMHCpan tiene groove CERRADO en los dos
              extremos, asi que F5 y el core teorico de 9 aa casi nunca
              coinciden para ventanas de 8/10/11 aa (core con guion insertado
              o con 1-2 residuos borrados del medio, ver
              ``_core_span_in_sequence``). Confirmado empiricamente: el
              fallback por subsecuencia terminaba resaltando el 100% de F5 en
              TODOS los casos de MHC-I (los huecos/borrados de NetMHCpan
              nunca tocan las posiciones de anclaje P1/PΩ, que son los dos
              extremos del peptido -- asi que la primera y ultima letra
              siempre calzan en el borde, y el resaltado ya no distinguia
              nada). Con ``True``, solo se colorea cuando F5 es exactamente
              un 9-mero (core = F5 letra por letra); los 8/10/11-meros
              quedan sin colorear -- correcto, ahi no hay "nucleo dentro de
              flanco" que resaltar, todo el peptido corto ES el nucleo.

    Cuando la corrida cubre varias proteinas de entrada (FASTA multi-registro,
    p. ej. ``fasta_inputs/MonkeyPox/mpxv_targets.fasta`` con 6 accessions),
    las filas se ordenan por ``accession``/``start`` -en vez de dejarlas en
    el orden interno de ``predict_netmhciipan`` (peptidos cortos en modo
    exacto primero, largos despues, sin relacion con el orden de las
    proteinas de entrada)- y se imprime una linea separadora cada vez que
    cambia el accession, para que cada proteina se lea como un bloque
    propio en vez de una lista continua.

    Columna ``NetCleave`` condicional: esta misma funcion se reusa para Fase 5
    (MHC-II, nunca corre NetCleave) y Fase 5b (MHC-I, si lo corre) -- la
    columna solo se agrega si ``traceback_df`` trae ``netcleave_c_term_match``
    (la anota ``netcleave_engine.annotate_cterm_cleavage`` ANTES de llamar a
    esta funcion, ver ``pipeline.fase_5b_tc_promiscuidad``), asi que en Fase 5
    la tabla queda igual que antes.
    """
    if traceback_df.empty:
        print("No hay candidatos validos con traceback a la Fase 3 para mostrar.")
        return

    display_df = traceback_df.sort_values(["accession", "start"]).reset_index(drop=True)
    display_df["Promiscuidad"] = (
        display_df["n_alelos_promiscuos"].astype(str) + "/" + display_df["n_alelos_evaluados"].astype(str)
    )
    has_netcleave = "netcleave_c_term_match" in display_df.columns
    if has_netcleave:
        display_df["NetCleave"] = display_df["netcleave_c_term_match"].map({True: "Si", False: "No"})
    display_df = display_df.rename(
        columns={
            "accession": "Accession",
            "sequence_f5": "15-mero/peptido (Secuencia F5)",
            "core_9aa": "Core (9 aa)",
            "start": "Start",
            "end": "End",
            "origen": "Origen",
            "min_rank_el": "Min %Rank",
        }
    )
    seq_col = "15-mero/peptido (Secuencia F5)"
    core_col = "Core (9 aa)"
    columns_order = [
        "Accession", seq_col, core_col, "Start", "End",
        "Origen", "Promiscuidad", "Min %Rank",
    ]
    if has_netcleave:
        columns_order.append("NetCleave")
    display_df = display_df[columns_order].reset_index(drop=True)

    # max_colwidth=None desactiva el truncado con '...' de pandas: con el
    # texto truncado, la busqueda de la secuencia completa dentro de la
    # linea (mas abajo) fallaria y se perderia el resaltado.
    lines = display_df.to_string(index=False, max_colwidth=None).split("\n")
    header_line = lines[0]
    separator = "-" * len(header_line)
    print(header_line)
    print(separator)

    accessions = display_df["Accession"].tolist()
    rows = zip(lines[1:], display_df[seq_col], display_df[core_col], accessions)
    prev_accession = None
    for line, sequence, core, accession in rows:
        if prev_accession is not None and accession != prev_accession:
            print(separator)
        prev_accession = accession

        seq_start = line.find(sequence)
        if seq_start == -1:
            print(line)
            continue
        if require_exact_core:
            core_offset = sequence.find(core)
            span = (core_offset, core_offset + len(core)) if core_offset != -1 else None
        else:
            span = _core_span_in_sequence(sequence, core)
        if span is None:
            print(line)
            continue
        core_start = seq_start + span[0]
        core_end = seq_start + span[1]
        print(f"{line[:core_start]}{_CORE_ANSI_START}{line[core_start:core_end]}{_CORE_ANSI_END}{line[core_end:]}")
