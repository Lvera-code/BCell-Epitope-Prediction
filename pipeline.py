#!/usr/bin/env python3
"""Orquestador CLI del pipeline de descubrimiento de epitopos vacunales.

Tres caminos de entrada, seleccionados automaticamente segun el tipo de
archivo (``src.utils.input_router``) y, para input de estructura, segun
``Settings.PDB_PROCESSING_MODE``/``--pdb-mode`` (ver
``src.engines.engine_registry.active_engines_for``):

    Camino 1 (input FASTA):
        Fase 1 (saneamiento) -> Fase 2 con BepiPred-3.0 + EpiDope.
        Comportamiento identico al pipeline original, sin cambios.
    Camino 2 (input PDB/mmCIF, PDB_PROCESSING_MODE='structure_only'):
        Fase 1.5 (extraccion de estructura) -> Fase 2 con DiscoTope-3.0 +
        ScanNet UNICAMENTE. BepiPred-3.0/EpiDope nunca se invocan.
    Camino 3 (input PDB/mmCIF, PDB_PROCESSING_MODE='structure_and_sequence'):
        Fase 1.5 -> Fase 2 con los 4 motores. El FASTA canonico (ATMSEQ)
        derivado en Fase 1.5 se pasa tambien a BepiPred-3.0/EpiDope, EXCEPTO
        si contiene residuos no canonicos (ver ``is_bepipred_compatible``):
        en ese caso se omiten solo esos dos motores para esta corrida (con
        aviso claro), sin frenar los motores estructurales ni el resto del
        pipeline (confirmado empiricamente que BepiPred-3.0 rechaza en
        bloque, exit code 1, cualquier caracter fuera de los 20 aminoacidos
        estandar -ver ``src.utils.fasta_parser``-).

A partir de Fase 2, el resto del flujo es identico para los 3 caminos:

    2. Prediccion de antigenicidad por residuo via los motores activos para
       este camino, EJECUTADOS EN LOCAL (subprocess). Cada motor tiene
       auto-cache propio en ``fasta_outputs/``.

    Checkpointing (Fases 4/4b/4c/5/5b/6): igual que el auto-cache de Fase 2,
    cada una de estas fases guarda un sidecar ``.inputhash`` junto a su CSV
    final. Si una corrida se interrumpe (p. ej. un OOM en Fase 4c) y se
    relanza con el MISMO input y los mismos parametros, cada fase ya
    completada se detecta por hash y se salta directo a la que fallo, en vez
    de recomputar todo desde el principio (ver ``_load_phase_checkpoint``/
    ``_write_phase_checkpoint``). El pico de memoria residente del proceso se
    loggea despues de las fases mas pesadas (ver ``_log_peak_memory``), para
    diagnosticar en cual ocurrio un OOM sin depender de herramientas externas.
    3. Mapeo local de regiones de epitopo contiguas por encima de un umbral
       para cada motor activo (``src.engines.epitope_mapping``), y UNION
       LOGICA ANOTADA entre todos ellos (``src.engines.consensus``): TODA
       region detectada por CUALQUIER motor activo avanza a la Fase 4; las
       regiones que solapan entre motores se FUSIONAN (start minimo, end
       maximo, sin recortar a la interseccion) y quedan etiquetadas en
       ``origen`` con TODOS los motores contribuyentes. Filtro de longitud
       inquebrantable: se descarta cualquier region final menor a 9 aa antes
       de la Fase 4.
    3b. Enmascarado transmembrana/peptido senal (TMbed LOCAL,
       ``src.engines.tmbed_engine``): corre sobre la secuencia COMPLETA de
       cada accession (no por peptido candidato) y descarta de la union
       anotada de Fase 3 cualquier region que caiga dentro de una
       helice/tira transmembrana o del peptido senal N-terminal, ANTES de
       BLASTp -- esos residuos no son accesibles a anticuerpos en la
       proteina madura/anclada a membrana. Reusa el venv/pesos ya instalados
       para el plugin Scipion ``scipion-chem-tmbed`` (repo hermano), sin
       importar codigo de ese plugin.
    4. Filtro de tolerancia inmunologica: BLASTp local contra el proteoma
       humano, descarta homologos de alta identidad (``src.engines.blast_engine``).
       Los peptidos 'Segura' resultantes alimentan, en paralelo y sin
       depender entre si, TODAS las fases siguientes (4b, 4c, 5, 5b, 6).
    4b. Alergenicidad (AlgPred 2.0 LOCAL, ``src.engines.algpred_engine``):
       señal de seguridad de la secuencia en si, informativa, no filtra
       ninguna fase posterior.
    4c. N-glicosilacion (StackGlyEmbed LOCAL, ``src.engines.stackglyembed_engine``):
       escanea sequones N-X-[S/T] propios (X != Prolina) y evalua cada uno
       con un stack ProteinBERT+ESM-2+ProtT5 ya entrenado; informativa,
       igual que 4b.
    5. Prediccion de presentacion T-helper (MHC-II) via NetMHCIIpan-4.3 LOCAL
       contra un panel de 27 alelos HLA-DR/DQ/DP de referencia
       (IEDB_REFERENCE_PANEL); reporta como candidato final solo los
       peptidos "promiscuos" (SB/WB en >= 3 alelos del panel EN ORIENTACION
       DE UNION NORMAL -los alelos que solo aglutinan via un registro
       invertido, de menor confianza, no cuentan para el veredicto pero se
       reportan aparte, pensado para minimizar falsos positivos que lleguen
       a sintesis/validacion experimental-), enriquecidos con su nucleo de
       union de 9 aa y su traceback de coordenadas/origen a la region de la
       Fase 3 de la que provienen (``src.engines.netmhciipan_engine``).
    5b. Promiscuidad T-citotoxica (MHC-I, NetMHCpan-4.2 LOCAL,
       ``src.engines.netmhcpan_engine``), paralela e independiente de la
       Fase 5 (MHC-II), anotada ademas con evidencia de corte proteasomal
       C-terminal (NetCleave LOCAL, ``src.engines.netcleave_engine``).
    6. Cruce con bnAb conocidos (LANL Immunology DB + CATNAP LOCAL,
       ``src.engines.lanl_catnap_engine``): puramente informativo, solo
       produce matches reales para entradas de la familia HIV Env.
    7. Ensamblaje automatico del constructo multi-epitopo
       (``src.engines.construct_assembly``): selecciona los mejores
       ``Settings.CONSTRUCT_TOP_N_PER_CLASS`` candidatos B-cell/HTL/CTL
       (Fases 4/4b/4c, 5, 5b) y los concatena con los linkers estandar del
       campo (AAY intra-CTL, GPGPG intra-HTL/inter-bloque, KK intra-B-cell)
       en un unico FASTA, con metadata 100% trazable de que peptido/linker
       fue a cada posicion.
    8. Chequeo del constructo ensamblado (alergenicidad -AlgPred2, reusa
       Fase 4b-, toxicidad -ToxinPred2 LOCAL-, antigenicidad intrinseca
       -IApred LOCAL, reemplaza a VaxiJen, descartado por no ser local/open
       source-, peptido senal -SignalP-6.0 LOCAL-): a diferencia de Fase
       4b/4c (por peptido individual), estas 4 corren sobre la secuencia
       COMPLETA del constructo de Fase 7.

Todos los artefactos intermedios y el reporte final se guardan en
``fasta_outputs/``. Requiere, segun los motores que active cada camino:
instalacion local de BepiPred-3.0 en ``bepipred-3.0b.src/`` y de EpiDope en
``.conda-epidope/``, DiscoTope-3.0 en ``DiscoTope-3.0/`` (entorno
``.venv-discotope``) y ScanNet (``.venv-scannet`` o Docker), NCBI BLAST+ con
el proteoma humano indexado en ``reference_db/``, y NetMHCIIpan-4.3 instalado
localmente en ``netMHCIIpan-4.3/`` (descarga manual bajo licencia academica
DTU Health Tech). Ver README.md - Seccion de Instalacion.

Ejemplo:
    python pipeline.py --input fasta_inputs/secuencia.fasta
    python pipeline.py --input fasta_inputs/estructura.pdb --pdb-mode structure_only
"""

import argparse
import hashlib
import resource
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import Settings
from src.engines.bepipred_engine import BepiPredEngine
from src.engines.bepipred_engine import extract_epitopes as extract_bepipred_epitopes
from src.engines.bepipred_engine import ACCESSION_COLUMN as BEPIPRED_ACCESSION_COLUMN
from src.engines.bepipred_engine import RESIDUE_COLUMN_CANDIDATES as BEPIPRED_RESIDUE_CANDIDATES
from src.engines.blast_engine import print_blast_report, run_blastp_filter
from src.engines.algpred_engine import predict_allergenicity, print_allergenicity_report
from src.engines.consensus import build_annotated_union_table, print_union_table
from src.engines.construct_assembly import assemble_construct
from src.engines.iapred_engine import predict_intrinsic_antigenicity, print_iapred_report
from src.engines.discotope_engine import DiscoTopeEngine
from src.engines.discotope_engine import extract_epitopes as extract_discotope_epitopes
from src.engines.discotope_engine import print_epitope_table as print_discotope_epitope_table
from src.engines.engine_registry import ENGINE_REGISTRY, active_engines_for
from src.engines.epidope_engine import EpidopeEngine
from src.engines.epidope_engine import extract_epitopes as extract_epidope_epitopes
from src.engines.epidope_engine import ACCESSION_COLUMN as EPIDOPE_ACCESSION_COLUMN
from src.engines.epidope_engine import RESIDUE_COLUMN as EPIDOPE_RESIDUE_COLUMN
from src.engines.epitope_mapping import build_sequence_lookup, print_epitope_table
from src.engines.lanl_catnap_engine import query_bnab_crossref, print_bnab_crossref_report
from src.engines.netcleave_engine import annotate_cterm_cleavage, predict_cleavage
from src.engines.netmhciipan_engine import (
    IEDB_REFERENCE_PANEL,
    build_traceback_report,
    predict_netmhciipan,
    print_th_report,
    print_traceback_table,
    validate_allele_extra,
)
from src.engines.netmhcpan_engine import (
    NETMHCPAN_REFERENCE_PANEL,
    build_traceback_report as build_traceback_report_mhci,
    predict_netmhcpan,
    print_tc_report,
)
from src.engines.scannet_engine import ScanNetEngine
from src.engines.scannet_engine import extract_epitopes as extract_scannet_epitopes
from src.engines.scannet_engine import print_epitope_table as print_scannet_epitope_table
from src.engines.signalp_engine import predict_signal_peptide, print_signalp_report
from src.engines.stackglyembed_engine import predict_nglycosylation, print_glycosylation_report
from src.engines.tmbed_engine import filter_overlapping_regions, predict_tm_signal_regions, print_tmbed_regions_report
from src.engines.toxinpred_engine import predict_toxicity, print_toxicity_report
from src.utils.exceptions import PipelineError
from src.utils.fasta_parser import FastaRecord, is_bepipred_compatible, load_and_sanitize, write_fasta
from src.utils.input_router import route_input
from src.utils.logger_config import setup_logger
from src.utils.structure_parser import StructureRecord, parse_structure

logger = setup_logger(__name__)

_SEPARATOR = "=" * 70


def _alelo_extra_type(value: str) -> str:
    """Wrapper de ``validate_allele_extra`` para usar como ``type=`` de argparse.

    Validacion TEMPRANA: se ejecuta al parsear los argumentos, antes de que
    corra ninguna fase del pipeline. Asi, un alelo mal escrito en
    ``--alelo-extra`` se rechaza de inmediato con un mensaje accionable, en
    vez de recien fallar al final de la Fase 5 (tras haber corrido
    BepiPred/EpiDope/BLASTp para nada) con un error generico de formato de
    ``.xls``. ``argparse`` descarta el mensaje de un ``ValueError`` plano y
    lo reemplaza por uno generico ("invalid _alelo_extra_type value: ..."),
    asi que aqui se traduce a ``ArgumentTypeError``, que si preserva el
    mensaje detallado de ``validate_allele_extra``.
    """
    try:
        return validate_allele_extra(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos del pipeline."""
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Pipeline de descubrimiento de epitopos vacunales "
        "(BepiPred-3.0 + EpiDope + DiscoTope-3.0 + ScanNet + BLASTp + MHC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Ruta al archivo de entrada (dentro de fasta_inputs/): FASTA, o PDB/mmCIF de "
        "estructura. El tipo se detecta automaticamente (ver src.utils.input_router).",
    )
    parser.add_argument(
        "--pdb-mode", default=None, choices=["structure_only", "structure_and_sequence"],
        help="Override puntual de Settings.PDB_PROCESSING_MODE para esta corrida. Solo aplica "
        "si --input es una estructura (PDB/mmCIF); se ignora para input FASTA. Si no se "
        "especifica, se usa Settings.PDB_PROCESSING_MODE.",
    )
    parser.add_argument(
        "--alelo-extra", default=None, type=_alelo_extra_type,
        help="Alelo(s) HLA-DR/DQ/DP adicionales a anexar al panel por defecto de la "
        "Fase 5 (IEDB_REFERENCE_PANEL, 27 alelos). Formato NetMHCIIpan, separados "
        "por coma sin espacios (ej. 'DRB1_1602' o 'HLA-DQA10501-DQB10201'). No "
        "especificar este flag no requiere ninguna otra accion: el panel por "
        "defecto siempre se evalua. Se valida el formato al momento de parsear "
        "este flag, antes de correr cualquier fase.",
    )
    parser.add_argument(
        "--output-dir", default=str(Settings.FASTA_OUTPUT_DIR),
        help="Carpeta donde se guardan todos los resultados del pipeline.",
    )
    parser.add_argument(
        "--bepipred-threshold", type=float, default=Settings.BEPIPRED_THRESHOLD,
        help="Umbral de score de antigenicidad (BepiPred-3.0) para la ventana deslizante de epitopos (Fase 3).",
    )
    parser.add_argument(
        "--bepipred-min-length", type=int, default=Settings.BEPIPRED_MIN_EPITOPE_LENGTH,
        help="Longitud minima (aa) de una region de epitopo BepiPred-3.0 para no ser descartada (Fase 3).",
    )
    parser.add_argument(
        "--epidope-threshold", type=float, default=Settings.EPIDOPE_THRESHOLD,
        help="Umbral de score de antigenicidad (EpiDope) para la ventana deslizante de epitopos (Fase 3). "
        "No comparable en escala al umbral de BepiPred: cada motor conserva el suyo.",
    )
    parser.add_argument(
        "--epidope-min-length", type=int, default=Settings.EPIDOPE_MIN_EPITOPE_LENGTH,
        help="Longitud minima (aa) de una region de epitopo EpiDope para no ser descartada (Fase 3).",
    )
    parser.add_argument(
        "--discotope-threshold", type=float, default=Settings.DISCOTOPE_THRESHOLD,
        help="Umbral de 'calibrated_score' (DiscoTope-3.0) para la ventana deslizante de epitopos "
        "(Fase 3). El default (0.90, 'moderate') es el nivel oficial publicado por los autores; "
        "otros niveles documentados: 0.40 ('low', ~70%% recall) y 1.51 ('higher', mas precision).",
    )
    parser.add_argument(
        "--discotope-min-length", type=int, default=Settings.DISCOTOPE_MIN_EPITOPE_LENGTH,
        help="Longitud minima (aa) de una region de epitopo DiscoTope-3.0 para no ser descartada (Fase 3).",
    )
    parser.add_argument(
        "--scannet-threshold", type=float, default=None,
        help="Umbral ABSOLUTO fijo de 'Binding site probability' (ScanNet) para la ventana "
        "deslizante de epitopos (Fase 3). ScanNet no publica un umbral oficial (a diferencia de "
        "DiscoTope-3.0): si no se especifica este flag (default), se usa un umbral ADAPTATIVO "
        "calculado por accession como el percentil Settings.SCANNET_THRESHOLD_PERCENTILE (90 por "
        "defecto) de los scores de esa cadena especifica, en vez de un numero fijo universal.",
    )
    parser.add_argument(
        "--scannet-min-length", type=int, default=Settings.SCANNET_MIN_EPITOPE_LENGTH,
        help="Longitud minima (aa) de una region de epitopo ScanNet para no ser descartada (Fase 3).",
    )
    parser.add_argument(
        "--blast-db", default=Settings.BLAST_HUMAN_DB,
        help="Prefijo de la base de datos BLAST local del proteoma humano (Fase 4). "
        "Tambien configurable via la variable de entorno BLAST_HUMAN_DB.",
    )
    parser.add_argument(
        "--identity-threshold", type=float, default=Settings.BLAST_IDENTITY_THRESHOLD,
        help="Porcentaje de identidad (exclusivo) por encima del cual se descarta un peptido (Fase 4).",
    )
    return parser.parse_args(argv)


def fase_1_saneamiento(input_path: Path, output_dir: Path) -> Tuple[List[FastaRecord], Path]:
    """Fase 1 (Camino 1, input FASTA): lee, valida y sanea el FASTA de entrada; escribe una copia limpia."""
    print(f"\n{_SEPARATOR}\nFASE 1 | Saneamiento del FASTA de entrada\n{_SEPARATOR}")

    records = load_and_sanitize(input_path)
    for record in records:
        print(
            f"Archivo: {input_path.name} | Registro: {record.accession} | "
            f"Validacion: OK | Longitud: {len(record.sequence)} aa"
        )

    clean_path = output_dir / f"{input_path.stem}_clean.fasta"
    write_fasta(records, clean_path)
    print(f"-> FASTA saneado escrito en: {clean_path}")
    return records, clean_path


def fase_1_5_estructura(input_path: Path, output_dir: Path) -> StructureRecord:
    """Fase 1.5 (Caminos 2/3, input estructura): extrae ATMSEQ y mapeo de posiciones del PDB/mmCIF.

    Ocurre siempre que el input sea una estructura, sin importar
    ``PDB_PROCESSING_MODE`` (ver docstring del modulo): lo que varia entre
    Camino 2 y Camino 3 es si el FASTA derivado aqui se pasa o no a Fase 2
    para BepiPred-3.0/EpiDope, decidido en ``main()``.
    """
    print(f"\n{_SEPARATOR}\nFASE 1.5 | Extraccion de estructura (PDB/mmCIF)\n{_SEPARATOR}")

    record = parse_structure(input_path, output_dir)
    print(
        f"Archivo: {input_path.name} | Accession: {record.accession} | "
        f"Cadena elegida: {record.chain_id} | Longitud ATMSEQ: {len(record.sequence)} aa"
    )
    print(f"-> FASTA derivado (ATMSEQ) escrito en: {record.fasta_path}")
    print(f"-> PDB de una sola cadena escrito en: {record.chain_pdb_path}")
    print(f"-> Mapeo de posiciones (PDB <-> FASTA derivado) escrito junto al FASTA derivado.")
    return record


def fase_2_antigenicidad(
    active_engines: List[str],
    input_stem: str,
    clean_fasta: Optional[Path],
    structure_record: Optional[StructureRecord],
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Fase 2: obtiene scores crudos de antigenicidad de cada motor activo, con auto-cache en CSV.

    Args:
        active_engines: Claves de ``ENGINE_REGISTRY`` a ejecutar (ver
            ``active_engines_for``), en el mismo orden en que luego se
            etiqueta ``origen`` en Fase 3.
        clean_fasta: FASTA saneado (Camino 1) o FASTA derivado de Fase 1.5
            (Camino 3). Requerido si algun motor activo consume ``'fasta'``.
        structure_record: Resultado de Fase 1.5. Requerido si algun motor
            activo consume ``'pdb'``.

    Returns:
        Diccionario ``nombre_motor -> DataFrame de scores crudos``, solo con
        las claves de ``active_engines`` que efectivamente corrieron.
    """
    print(
        f"\n{_SEPARATOR}\nFASE 2 | Prediccion de antigenicidad "
        f"({' + '.join(active_engines)}, ejecucion local)\n{_SEPARATOR}"
    )

    raw_dfs: Dict[str, pd.DataFrame] = {}

    if "bepipred" in active_engines:
        raw_dfs["bepipred"] = _cached_raw_scores(
            engine_name="BepiPred-3.0",
            cache_path=output_dir / f"{input_stem}_bepipred_raw.csv",
            raw_artifacts_dir=output_dir / "_bepipred_raw",
            clean_fasta=clean_fasta,
            engine=BepiPredEngine(),
        )
    if "epidope" in active_engines:
        raw_dfs["epidope"] = _cached_raw_scores(
            engine_name="EpiDope",
            cache_path=output_dir / f"{input_stem}_epidope_raw.csv",
            raw_artifacts_dir=output_dir / "_epidope_raw",
            clean_fasta=clean_fasta,
            engine=EpidopeEngine(),
        )
    if "discotope" in active_engines:
        raw_dfs["discotope"] = _cached_structural_raw_scores(
            engine_name="DiscoTope-3.0",
            cache_path=output_dir / f"{input_stem}_discotope_raw.csv",
            raw_artifacts_dir=output_dir / "_discotope_raw",
            structure_record=structure_record,
            engine=DiscoTopeEngine(),
        )
    if "scannet" in active_engines:
        raw_dfs["scannet"] = _cached_structural_raw_scores(
            engine_name="ScanNet",
            cache_path=output_dir / f"{input_stem}_scannet_raw.csv",
            raw_artifacts_dir=output_dir / "_scannet_raw",
            structure_record=structure_record,
            engine=ScanNetEngine(),
        )
    return raw_dfs


def _content_hash(path: Path) -> str:
    """Hash corto (16 hex) del contenido de ``path``, usado para invalidar caches obsoletos."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _hash_sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".inputhash")


def _phase_input_hash(*parts) -> str:
    """Hash corto (16 hex) de cualquier combinacion de DataFrames/valores que determinen el resultado de una fase.

    Generalizacion del mecanismo de auto-cache de ``_cached_raw_scores``
    (Fase 2, por contenido de ARCHIVO) a las fases posteriores (4/4b/4c/5/5b/6),
    que reciben DataFrames en memoria en vez de un path -- ``pd.DataFrame`` se
    serializa via ``to_csv`` antes de hashear, cualquier otro tipo (str/float/
    None) se serializa con ``str()``. El orden de ``parts`` importa (dos fases
    con los mismos valores en distinto orden producen hashes distintos), asi
    que cada llamador debe ser consistente entre la escritura y la lectura del
    checkpoint.
    """
    pieces = [part.to_csv(index=False) if isinstance(part, pd.DataFrame) else str(part) for part in parts]
    payload = "\x00".join(pieces).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_phase_checkpoint(engine_name: str, cache_path: Path, input_hash: str) -> Optional[pd.DataFrame]:
    """Carga ``cache_path`` si su sidecar de hash coincide con ``input_hash`` (cache-hit); ``None`` si no.

    Motivado por el OOM que origino ``STATUS.md``: sin esto, un crash a mitad
    de una corrida larga (p. ej. durante Fase 5b/NetMHCpan, tras ya haber
    corrido BLAST/AlgPred2/StackGlyEmbed/NetMHCIIpan) obligaba a repetir TODO
    desde cero. Con este checkpoint por fase, reiniciar la misma corrida
    (mismo input, mismos parametros) salta directo a la fase que fallo.
    """
    hash_path = _hash_sidecar_path(cache_path)
    if cache_path.is_file() and hash_path.is_file() and hash_path.read_text().strip() == input_hash:
        print(f"[{engine_name}] Checkpoint detectado en '{cache_path}' (mismo input que la corrida anterior). Se omite la re-ejecucion.")
        return pd.read_csv(cache_path)
    if cache_path.is_file():
        logger.info(
            "[%s] Checkpoint en '%s' obsoleto (el input de esta fase cambio desde que se genero): se re-ejecuta.",
            engine_name, cache_path,
        )
    return None


def _write_phase_checkpoint(cache_path: Path, input_hash: str) -> None:
    _hash_sidecar_path(cache_path).write_text(input_hash)


def _log_peak_memory(phase_name: str) -> None:
    """Loggea el pico de RSS (memoria residente) del proceso hasta este punto.

    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` es MONOTONO CRECIENTE (pico
    historico del proceso completo, no un snapshot puntual): comparar el valor
    despues de cada fase da una cota inferior de cuanta memoria consumio esa
    fase en particular (la diferencia con el valor anterior), util para
    diagnosticar en que fase ocurre un OOM sin necesitar 'psutil' (no es una
    dependencia del proyecto) ni herramientas externas de profiling. El modulo
    ``resource`` es POSIX-only (no existe en Windows) -- consistente con el
    resto del pipeline, que ya asume Linux/WSL2 (venvs con rutas POSIX,
    subprocess a binarios ELF, etc.), aunque el README no lo declara
    explicitamente. En Linux el valor de ``ru_maxrss`` viene en KB.
    """
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info("[RAM] Pico de memoria residente tras %s: %.1f MB", phase_name, peak_kb / 1024)


def _cached_raw_scores(
    engine_name: str, cache_path: Path, raw_artifacts_dir: Path, clean_fasta: Path, engine
) -> pd.DataFrame:
    """Corre ``engine`` (motor de secuencia) sobre ``clean_fasta`` con auto-cache en ``cache_path`` (CSV).

    El cache se invalida por CONTENIDO, no solo por nombre de archivo:
    ``cache_path`` va acompañado de un sidecar ``{cache_path}.inputhash`` con
    el hash del ``clean_fasta`` que lo genero. CONFIRMADO EMPIRICAMENTE
    (2026-07-20): sin esto, dos corridas del MISMO input_stem con un
    ``clean_fasta``/cadena distinta (p. ej. 6xc2.pdb corrido una vez con
    ``PDB_CHAIN_SELECTION_STRATEGY=longest`` -pica la cadena del Fab- y otra
    con ``explicit``/cadena A -el antigeno real-) reusaba en silencio el CSV
    crudo de la corrida anterior, mezclando datos de una cadena con los de
    otra sin ningun aviso. Si el hash no coincide (input cambio) o falta el
    sidecar (cache de una version anterior sin este mecanismo), se trata como
    cache-miss y se re-ejecuta.
    """
    current_hash = _content_hash(clean_fasta)
    hash_path = _hash_sidecar_path(cache_path)

    if cache_path.is_file() and hash_path.is_file() and hash_path.read_text().strip() == current_hash:
        df = pd.read_csv(cache_path)
        print(f"[{engine_name}] Cache local detectada en '{cache_path}'. Se omite la re-ejecucion.")
        print(f"[{engine_name}] Origen de los datos: CACHE LOCAL | Dimensiones de la matriz: {df.shape}")
        return df

    if cache_path.is_file():
        logger.info(
            "[%s] Cache en '%s' obsoleto (el input cambio de contenido desde que se genero, "
            "p. ej. otra cadena/estrategia de seleccion): se re-ejecuta.",
            engine_name, cache_path,
        )

    df = engine.run([str(clean_fasta)], output_dir=raw_artifacts_dir)[0]

    df.to_csv(cache_path, index=False)
    hash_path.write_text(current_hash)
    print(f"[{engine_name}] Origen de los datos: INFERENCIA LOCAL (subprocess) | Dimensiones de la matriz: {df.shape}")
    print(f"[{engine_name}] -> Resultado crudo cacheado en: {cache_path} (para futuras ejecuciones instantaneas)")
    return df


def _cached_structural_raw_scores(
    engine_name: str, cache_path: Path, raw_artifacts_dir: Path, structure_record: StructureRecord, engine
) -> pd.DataFrame:
    """Corre ``engine`` (motor estructural) sobre ``structure_record.chain_pdb_path``, con auto-cache.

    A diferencia de ``_cached_raw_scores`` (motores de secuencia), reconcilia
    el accession que reporta el motor (derivado del nombre del PDB de una
    sola cadena, p. ej. ``'{accession}_chain_A'``) con el accession real
    (``structure_record.accession``) ANTES de cachear -- ver ADR en
    ``discotope_engine.py``/``scannet_engine.py``: ambos motores reportan
    ``Path(pdb_path).stem`` tal cual, sin ningun ajuste de negocio propio.

    Tambien verifica que el motor haya devuelto una fila por cada residuo de
    ``structure_record.sequence`` (ATMSEQ). CONFIRMADO EMPIRICAMENTE
    (2026-07-20, PDB sintetico con un residuo no mapeable al CCD): a
    diferencia de nuestro propio ``structure_parser`` -que conserva CADA
    residuo del polimero, usando 'X' cuando no puede resolver una letra-,
    DiscoTope-3.0 DESCARTA en silencio los residuos que su propio parser no
    reconoce (3 residuos de entrada -> 2 filas de salida en ese caso). El
    chequeo de limites de ``consensus._warn_if_out_of_bounds`` NO detecta
    esto por si solo: solo compara coordenadas de regiones YA extraidas, y si
    la secuencia es corta o el score no llega al umbral en ningun lado, nunca
    se llega a comparar nada. Este chequeo es mas temprano y mas fuerte: se
    dispara apenas se reciben los scores crudos, exista o no una region
    detectada despues.
    Ver tambien ``_cached_raw_scores``: el cache tambien se invalida por
    CONTENIDO de ``structure_record.chain_pdb_path`` (sidecar
    ``{cache_path}.inputhash``), no solo por nombre de archivo -- mismo bug
    real confirmado con ``chain_pdb_path`` (una corrida con
    ``PDB_CHAIN_SELECTION_STRATEGY`` distinta genera un PDB de una sola
    cadena con contenido distinto bajo el mismo ``input_stem``).
    """
    current_hash = _content_hash(structure_record.chain_pdb_path)
    hash_path = _hash_sidecar_path(cache_path)

    if cache_path.is_file() and hash_path.is_file() and hash_path.read_text().strip() == current_hash:
        df = pd.read_csv(cache_path)
        print(f"[{engine_name}] Cache local detectada en '{cache_path}'. Se omite la re-ejecucion.")
        print(f"[{engine_name}] Origen de los datos: CACHE LOCAL | Dimensiones de la matriz: {df.shape}")
        _warn_if_residue_count_mismatch(engine_name, df, structure_record)
        return df

    if cache_path.is_file():
        logger.info(
            "[%s] Cache en '%s' obsoleto (el PDB de una sola cadena cambio de contenido desde que "
            "se genero, p. ej. otra cadena/estrategia de seleccion): se re-ejecuta.",
            engine_name, cache_path,
        )

    df = engine.run([str(structure_record.chain_pdb_path)], output_dir=raw_artifacts_dir)[0]
    df["Accession"] = structure_record.accession
    _warn_if_residue_count_mismatch(engine_name, df, structure_record)

    df.to_csv(cache_path, index=False)
    hash_path.write_text(current_hash)
    print(f"[{engine_name}] Origen de los datos: INFERENCIA LOCAL (subprocess) | Dimensiones de la matriz: {df.shape}")
    print(f"[{engine_name}] -> Resultado crudo cacheado en: {cache_path} (para futuras ejecuciones instantaneas)")
    return df


def _warn_if_residue_count_mismatch(engine_name: str, raw_df: pd.DataFrame, structure_record: StructureRecord) -> None:
    """Compara el numero de filas crudas de un motor estructural contra len(ATMSEQ).

    Ver docstring de ``_cached_structural_raw_scores``. Solo loggea (no
    detiene el pipeline): un desfase indica que las coordenadas de ese motor
    pueden no alinear 1:1 con ``sequence_lookup``/``position_mapping`` para
    esta accession -- util para diagnosticar resultados sospechosos, pero no
    hay una forma segura y generica de re-alinear automaticamente sin mas
    informacion (ver ADR en ``consensus.py``).
    """
    expected = len(structure_record.sequence)
    actual = len(raw_df)
    if actual != expected:
        logger.warning(
            "Accession '%s': %s devolvio %d fila(s) de score crudo, pero la secuencia ATMSEQ "
            "derivada tiene %d residuo(s). El motor probablemente descarto o agrego algun "
            "residuo que su propio parser interpreta distinto a structure_parser (residuos no "
            "mapeables, backbone incompleto, etc.): las coordenadas de este motor para esta "
            "accession pueden no corresponder 1:1 a la posicion real en la secuencia -- "
            "revisar manualmente antes de confiar en las regiones que reporte.",
            structure_record.accession, engine_name, actual, expected,
        )
        print(
            f"[AVISO] {engine_name} devolvio {actual} fila(s) mientras que la secuencia ATMSEQ "
            f"tiene {expected} residuo(s) (accession '{structure_record.accession}'): posible "
            "desalineacion de coordenadas, revisar manualmente."
        )


def _build_full_sequence_lookup(
    raw_dfs: Dict[str, pd.DataFrame], structure_record: Optional[StructureRecord]
) -> Dict[str, str]:
    """Reconstruye ``accession -> secuencia completa`` desde estructura o motores de secuencia crudos.

    Para input de estructura, la fuente es directamente
    ``StructureRecord.sequence`` (ATMSEQ); para motores de secuencia, se
    reconstruye desde sus propios scores crudos de Fase 2. Usado tanto por
    Fase 3 (union de regiones fusionadas) como por Fase 3b (TMbed necesita
    la secuencia completa de cada accession, no solo el fragmento candidato).
    """
    sequence_lookup: Dict[str, str] = {}
    if structure_record is not None:
        sequence_lookup[structure_record.accession] = structure_record.sequence
    if "epidope" in raw_dfs:
        sequence_lookup.update(
            build_sequence_lookup(raw_dfs["epidope"], accession_col=EPIDOPE_ACCESSION_COLUMN, residue_col_candidates=(EPIDOPE_RESIDUE_COLUMN,))
        )
    if "bepipred" in raw_dfs:
        sequence_lookup.update(
            build_sequence_lookup(raw_dfs["bepipred"], accession_col=BEPIPRED_ACCESSION_COLUMN, residue_col_candidates=BEPIPRED_RESIDUE_CANDIDATES)
        )
    return sequence_lookup


def fase_3_mapeo_y_union(
    raw_dfs: Dict[str, pd.DataFrame],
    structure_record: Optional[StructureRecord],
    bepipred_threshold: float,
    bepipred_min_length: int,
    epidope_threshold: float,
    epidope_min_length: int,
    output_dir: Path,
    input_stem: str,
    discotope_threshold: float = Settings.DISCOTOPE_THRESHOLD,
    discotope_min_length: int = Settings.DISCOTOPE_MIN_EPITOPE_LENGTH,
    scannet_threshold: Optional[float] = None,
    scannet_min_length: int = Settings.SCANNET_MIN_EPITOPE_LENGTH,
) -> pd.DataFrame:
    """Fase 3: mapea regiones de epitopo por motor activo y construye la union logica anotada.

    ``discotope_threshold``/``scannet_threshold`` siguen el mismo patron que
    ``bepipred_threshold``/``epidope_threshold`` (configurables por CLI, ver
    ``parse_args``). Diferencia clave: ``scannet_threshold=None`` (default)
    activa el umbral ADAPTATIVO por accession (percentil de los scores de
    esa cadena especifica, ver ``scannet_engine.extract_epitopes``) en vez de
    un numero fijo -- ScanNet no publica un umbral absoluto oficial, a
    diferencia de DiscoTope-3.0 (``Settings.DISCOTOPE_THRESHOLD`` = 0.90 es
    el nivel "moderate" oficial de los autores).
    """
    print(
        f"\n{_SEPARATOR}\nFASE 3 | Mapeo logico de regiones de epitopo y union anotada "
        f"({' + '.join(raw_dfs.keys())})\n{_SEPARATOR}"
    )

    epitope_dfs: Dict[str, pd.DataFrame] = {}

    if "bepipred" in raw_dfs:
        print(f"-- BepiPred-3.0 (umbral={bepipred_threshold}, min_len={bepipred_min_length}) --")
        df = extract_bepipred_epitopes(raw_dfs["bepipred"], threshold=bepipred_threshold, min_length=bepipred_min_length)
        print_epitope_table(
            df, empty_message=f"No se encontraron regiones >= {bepipred_min_length} aa con score medio >= {bepipred_threshold}."
        )
        df.to_csv(output_dir / f"{input_stem}_bepipred_epitopes.csv", index=False)
        epitope_dfs["bepipred"] = df

    if "epidope" in raw_dfs:
        print(f"\n-- EpiDope (umbral={epidope_threshold}, min_len={epidope_min_length}) --")
        df = extract_epidope_epitopes(raw_dfs["epidope"], threshold=epidope_threshold, min_length=epidope_min_length)
        print_epitope_table(
            df, empty_message=f"No se encontraron regiones >= {epidope_min_length} aa con score medio >= {epidope_threshold}."
        )
        df.to_csv(output_dir / f"{input_stem}_epidope_epitopes.csv", index=False)
        epitope_dfs["epidope"] = df

    if "discotope" in raw_dfs:
        print(f"\n-- DiscoTope-3.0 (umbral={discotope_threshold} 'calibrated_score', min_len={discotope_min_length}) --")
        df = extract_discotope_epitopes(raw_dfs["discotope"], threshold=discotope_threshold, min_length=discotope_min_length)
        print_discotope_epitope_table(df)
        df.to_csv(output_dir / f"{input_stem}_discotope_epitopes.csv", index=False)
        epitope_dfs["discotope"] = df

    if "scannet" in raw_dfs:
        scannet_mode_desc = f"umbral fijo={scannet_threshold}" if scannet_threshold is not None else (
            f"umbral adaptativo, percentil {Settings.SCANNET_THRESHOLD_PERCENTILE} por accession"
        )
        print(f"\n-- ScanNet ({scannet_mode_desc}, min_len={scannet_min_length}) --")
        df = extract_scannet_epitopes(raw_dfs["scannet"], threshold=scannet_threshold, min_length=scannet_min_length)
        print_scannet_epitope_table(df)
        df.to_csv(output_dir / f"{input_stem}_scannet_epitopes.csv", index=False)
        epitope_dfs["scannet"] = df

    print("\n-- Union anotada (fusion de solapes entre motores activos) --")
    # Lookup de secuencia completa por accession: una region fusionada puede
    # exceder el span detectado por cualquiera de los motores por separado,
    # asi que la subsecuencia final se reconstruye desde aqui en vez de
    # recortar las subsecuencias individuales de cada motor. Reusado tal
    # cual por Fase 3b (TMbed necesita la misma secuencia completa).
    sequence_lookup = _build_full_sequence_lookup(raw_dfs, structure_record)

    position_mapping = structure_record.position_mapping if structure_record is not None else None
    union_df = build_annotated_union_table(epitope_dfs, sequence_lookup, position_mapping=position_mapping)
    print_union_table(union_df)

    out_path = output_dir / f"{input_stem}_union_epitopes.csv"
    union_df.to_csv(out_path, index=False)
    print(f"-> Tabla de union anotada guardada en: {out_path}")
    return union_df


def fase_3b_tm_signal_masking(
    raw_dfs: Dict[str, pd.DataFrame],
    structure_record: Optional[StructureRecord],
    union_df: pd.DataFrame,
    output_dir: Path,
    input_stem: str,
) -> pd.DataFrame:
    """Fase 3b: descarta de la union anotada las regiones dentro de una TM helix/strand o peptido senal (TMbed local).

    A diferencia de Fase 4b/4c (evaluan cada peptido candidato ya recortado),
    TMbed corre sobre la secuencia COMPLETA de cada accession (ver
    ``_build_full_sequence_lookup``): necesita el contexto completo de la
    proteina para predecir correctamente topologia de membrana, no solo el
    fragmento de 9+ aa que sobrevivio a Fase 3. El costoso paso (inferencia
    ProtT5 + prediccion) se cachea por CONTENIDO de las secuencias completas
    (``sequence_lookup``), no por ``union_df``: dos corridas del mismo input
    con distintos umbrales de Fase 3 (que cambian que regiones aparecen en
    ``union_df`` pero no las secuencias completas en si) reusan el mismo
    cache de regiones TMbed.

    Args:
        raw_dfs: Scores crudos de Fase 2 (mismo dict que recibe Fase 3),
            usado por ``_build_full_sequence_lookup``.
        structure_record: Resultado de Fase 1.5, o ``None`` para input FASTA puro.
        union_df: Union anotada de Fase 3.
        output_dir: Carpeta donde persistir las regiones TMbed y la union post-enmascarado.
        input_stem: Nombre del archivo de entrada sin extension.

    Returns:
        ``union_df`` sin las filas cuyo rango se solapa con una region
        TM/senal (mismo esquema de columnas, mismo orden relativo de filas
        restantes). Si no hay ninguna secuencia completa disponible o
        ``union_df`` esta vacio, se devuelve ``union_df`` sin cambios.
    """
    print(f"\n{_SEPARATOR}\nFASE 3b | Enmascarado transmembrana/peptido senal (TMbed local)\n{_SEPARATOR}")

    sequence_lookup = _build_full_sequence_lookup(raw_dfs, structure_record)
    regions_path = output_dir / f"{input_stem}_tmbed_regions.csv"
    masked_path = output_dir / f"{input_stem}_union_epitopes_masked.csv"

    if not sequence_lookup or union_df.empty:
        print("No hay secuencia completa disponible o la union anotada esta vacia: se omite Fase 3b.")
        union_df.to_csv(masked_path, index=False)
        return union_df

    input_hash = _phase_input_hash(
        *(f"{accession}:{sequence}" for accession, sequence in sorted(sequence_lookup.items())),
        Settings.TMBED_MIN_REGION_LENGTH,
    )
    cached_regions = _load_phase_checkpoint("Fase 3b", regions_path, input_hash)
    if cached_regions is not None:
        regions_df = cached_regions
    else:
        regions_df = predict_tm_signal_regions(sequence_lookup, output_dir, filename_prefix=f"{input_stem}_")
        regions_df.to_csv(regions_path, index=False)
        _write_phase_checkpoint(regions_path, input_hash)

    print_tmbed_regions_report(regions_df)

    masked_df, n_discarded = filter_overlapping_regions(union_df, regions_df)
    if n_discarded:
        print(
            f"[AVISO] {n_discarded} region(es) de la union anotada descartada(s) por solaparse con una "
            "helice/tira transmembrana o peptido senal (no accesibles a anticuerpos en la proteina "
            "madura/anclada a membrana)."
        )
    else:
        print("Ninguna region de la union anotada se solapa con una region TM/peptido senal detectada.")

    masked_df.to_csv(masked_path, index=False)
    print(f"-> Union anotada (post-enmascarado TM/senal) guardada en: {masked_path}")
    return masked_df


def fase_4_tolerancia(
    union_df: pd.DataFrame,
    blast_db: str,
    identity_threshold: float,
    output_dir: Path,
    input_stem: str,
) -> pd.DataFrame:
    """Fase 4: descarta por BLASTp local los peptidos de la union anotada con alta homologia al proteoma humano."""
    print(f"\n{_SEPARATOR}\nFASE 4 | Filtro de tolerancia inmunologica (BLASTp local, umbral={identity_threshold}%)\n{_SEPARATOR}")

    out_path = output_dir / f"{input_stem}_blast_report.csv"

    if union_df.empty:
        print("No hay peptidos de la union anotada de la Fase 3 para analizar.")
        blast_df = union_df.assign(
            blast_task=pd.Series(dtype=str),
            blast_evalue=pd.Series(dtype=float),
            max_pident=pd.Series(dtype=float),
            status=pd.Series(dtype=str),
        )
        blast_df.to_csv(out_path, index=False)
        return blast_df

    input_hash = _phase_input_hash(union_df, blast_db, identity_threshold)
    cached = _load_phase_checkpoint("Fase 4", out_path, input_hash)
    if cached is not None:
        return cached

    blast_df = run_blastp_filter(union_df, db_path=blast_db, identity_threshold=identity_threshold)
    print_blast_report(blast_df)

    blast_df.to_csv(out_path, index=False)
    _write_phase_checkpoint(out_path, input_hash)
    print(f"-> Informe de tolerancia guardado en: {out_path}")
    return blast_df


def fase_4b_alergenicidad(safe_df: pd.DataFrame, output_dir: Path, input_stem: str) -> pd.DataFrame:
    """Fase 4b: evalua alergenicidad (AlgPred 2.0 local) de los peptidos 'Seguros' de la Fase 4.

    Paso independiente en paralelo a ``fase_5_th_promiscuidad``/``fase_5b_tc_promiscuidad``,
    NO fusionado con ninguna: alergenicidad es una propiedad de seguridad de la
    secuencia en si (potencial de reaccion tipo I mediada por IgE), no de una
    via de presentacion antigenica particular -- se reporta como senal
    independiente, con su propio archivo de salida
    (``<input_stem>_alergenicidad_report.csv``).

    Args:
        safe_df: Mismos peptidos 'Segura' de la Fase 4 usados por Fase 5/5b.
        output_dir: Carpeta donde persistir el reporte final y el CSV crudo de AlgPred2.
        input_stem: Nombre del archivo de entrada sin extension (mismo
            proposito que en Fase 5/5b: evita que corridas sucesivas se pisen).
    """
    print(f"\n{_SEPARATOR}\nFASE 4b | Filtro de alergenicidad (AlgPred 2.0 local)\n{_SEPARATOR}")

    final_path = output_dir / f"{input_stem}_alergenicidad_report.csv"

    if safe_df.empty:
        print("No hay peptidos 'Seguros' provenientes de la Fase 4 para evaluar.")
        empty_df = pd.DataFrame(columns=["sequence", "algpred_score", "algpred_veredicto"])
        empty_df.to_csv(final_path, index=False)
        return empty_df

    input_hash = _phase_input_hash(safe_df)
    cached = _load_phase_checkpoint("Fase 4b", final_path, input_hash)
    if cached is not None:
        return cached

    peptides = safe_df["sequence"].tolist()
    print(f"Peptidos a evaluar: {len(peptides)}")

    report = predict_allergenicity(peptides, output_dir, filename_prefix=f"{input_stem}_")
    print_allergenicity_report(report)

    report.to_csv(final_path, index=False)
    _write_phase_checkpoint(final_path, input_hash)
    print(f"-> Reporte de alergenicidad guardado en: {final_path}")
    return report


def fase_4c_glicosilacion(safe_df: pd.DataFrame, output_dir: Path, input_stem: str) -> pd.DataFrame:
    """Fase 4c: evalua N-glicosilacion (StackGlyEmbed local) de los peptidos 'Seguros' de la Fase 4.

    Mismo tipo de paso que Fase 4b: independiente, en paralelo a Fase 5/5b,
    NO fusionado con ninguna -- N-glicosilacion es una propiedad de la
    secuencia en si (un sequon N-X-[S/T] real puede alterar plegado/
    inmunogenicidad del peptido sintetizado, sin importar la via de
    presentacion antigenica), con su propio archivo de salida
    (``<input_stem>_glicosilacion_report.csv``).

    A diferencia de Fase 4b (que evalua TODOS los peptidos 'Seguros'),
    ``predict_nglycosylation`` ya omite internamente los peptidos sin ningun
    sequon N-X-[S/T] (no producen ninguna fila): el reporte final puede
    tener menos filas que ``len(safe_df)``, o incluso varias filas por
    peptido si tiene mas de un sequon candidato.

    Args:
        safe_df: Mismos peptidos 'Segura' de la Fase 4 usados por Fase 4b/5/5b.
        output_dir: Carpeta donde persistir el reporte final y los CSV crudos de StackGlyEmbed.
        input_stem: Nombre del archivo de entrada sin extension (mismo
            proposito que en Fase 4b: evita que corridas sucesivas se pisen).
    """
    print(f"\n{_SEPARATOR}\nFASE 4c | N-glicosilacion (StackGlyEmbed local)\n{_SEPARATOR}")

    final_path = output_dir / f"{input_stem}_glicosilacion_report.csv"

    if safe_df.empty:
        print("No hay peptidos 'Seguros' provenientes de la Fase 4 para evaluar.")
        empty_df = pd.DataFrame(columns=["sequence", "sequon_position", "stackglyembed_veredicto", "stackglyembed_score"])
        empty_df.to_csv(final_path, index=False)
        return empty_df

    input_hash = _phase_input_hash(safe_df)
    cached = _load_phase_checkpoint("Fase 4c", final_path, input_hash)
    if cached is not None:
        return cached

    peptides = safe_df["sequence"].tolist()
    print(f"Peptidos a evaluar: {len(peptides)}")

    report = predict_nglycosylation(peptides, output_dir, filename_prefix=f"{input_stem}_")

    if not report.empty:
        print(f"Sequones evaluados: {len(report)} (de {report['sequence'].nunique()} peptido(s) con >=1 sequon).")
    print_glycosylation_report(report)

    report.to_csv(final_path, index=False)
    _write_phase_checkpoint(final_path, input_hash)
    print(f"-> Reporte de N-glicosilacion guardado en: {final_path}")
    return report


def fase_5_th_promiscuidad(
    safe_df: pd.DataFrame, output_dir: Path, input_stem: str, allele_extra: str = None
) -> pd.DataFrame:
    """Fase 5: evalua promiscuidad T-helper (MHC-II) de los peptidos 'Seguros' de la Fase 4.

    El reporte final (consola y ``<input_stem>_candidatos_finales.csv``) no
    es la salida cruda de NetMHCIIpan: los 'Candidato Valido' se enriquecen
    con su traceback a la region de origen de la Fase 3/4 (accession,
    coordenadas reales, origen y las columnas ``'{motor}_score'`` de los
    motores que contribuyeron a esa region, detectadas dinamicamente -- ver
    ``build_traceback_report`` en ``netmhciipan_engine.py``) y su nucleo de
    union de 9 aa, via ``build_traceback_report`` -necesario porque en modo
    proteina (fragmentos largos) NetMHCIIpan devuelve nucleos mas cortos que
    el fragmento evaluado, no el fragmento completo-.

    Args:
        safe_df: Peptidos con ``status == 'Segura'`` provenientes de la Fase 4
            (conserva ``accession``/``start``/``sequence``/``origen`` y las
            columnas ``'{motor}_score'`` de la Fase 3, usadas como tabla
            padre del traceback).
        output_dir: Carpeta donde persistir el reporte final y el .xls crudo.
        input_stem: Nombre del archivo de entrada sin extension, usado como
            prefijo de ``candidatos_finales.csv`` y de los .xls crudos de
            NetMHCIIpan. CONFIRMADO (2026-07-20): sin este prefijo, corridas
            sucesivas con inputs distintos pisaban el mismo archivo -- unica
            salida de todo el pipeline que no llevaba el nombre del input.
        allele_extra: Alelo(s) HLA-DR/DQ/DP adicionales (formato NetMHCIIpan,
            separados por coma sin espacios) a anexar a
            ``IEDB_REFERENCE_PANEL``. Se admiten sin romper el panel por
            defecto.
    """
    allele_panel = f"{IEDB_REFERENCE_PANEL},{allele_extra}" if allele_extra else IEDB_REFERENCE_PANEL
    n_alleles = len(allele_panel.split(","))
    print(f"\n{_SEPARATOR}\nFASE 5 | Promiscuidad T-helper (MHC-II, NetMHCIIpan-4.3 local, {n_alleles} alelo(s) HLA-DR/DQ/DP)\n{_SEPARATOR}")

    final_path = output_dir / f"{input_stem}_candidatos_finales.csv"

    if safe_df.empty:
        print("No hay peptidos 'Seguros' provenientes de la Fase 4 para evaluar.")
        traceback_df = build_traceback_report(pd.DataFrame(), safe_df)
        traceback_df.to_csv(final_path, index=False)
        return traceback_df

    input_hash = _phase_input_hash(safe_df, allele_panel)
    cached = _load_phase_checkpoint("Fase 5", final_path, input_hash)
    if cached is not None:
        return cached

    peptides = safe_df["sequence"].tolist()
    print(f"Panel HLA-DR: {allele_panel} | Peptidos a evaluar: {len(peptides)}")

    report = predict_netmhciipan(peptides, output_dir, allele_panel=allele_panel, filename_prefix=f"{input_stem}_")

    if report.empty:
        print("NetMHCIIpan no devolvio resultados evaluables (revisa longitudes minimas: 9 aa).")
    else:
        print_th_report(report, allele_panel=allele_panel)

    traceback_df = build_traceback_report(report, safe_df)
    print_traceback_table(traceback_df)

    traceback_df.to_csv(final_path, index=False)
    _write_phase_checkpoint(final_path, input_hash)
    print(f"-> Reporte final guardado en: {final_path}")
    return traceback_df


def fase_5b_tc_promiscuidad(safe_df: pd.DataFrame, output_dir: Path, input_stem: str) -> pd.DataFrame:
    """Fase 5b: evalua promiscuidad T-citotoxica (MHC-I) de los peptidos 'Seguros' de la Fase 4.

    Paso independiente en paralelo a ``fase_5_th_promiscuidad`` (MHC-II), NO
    fusionado con ella: son vias de presentacion antigenica distintas (ver
    ADR revertido 2026-07-21 en ``src/engines/netmhciipan_engine.py`` y el
    docstring completo de ``src/engines/netmhcpan_engine.py``). El criterio
    de veredicto de Fase 5 (T-helper/CD4+) no se toca; esto es una senal
    adicional, con su propio archivo de salida
    (``<input_stem>_candidatos_finales_mhc1.csv``) para no mezclar ambas
    tablas.

    Ademas anota cada candidato con evidencia de corte proteasomal C-terminal
    (NetCleave local, ver ``src.engines.netcleave_engine.annotate_cterm_cleavage``):
    un peptido puede bindear MHC-I fuerte y aun asi nunca generarse via
    procesamiento antigenico real si el proteasoma no corta exactamente donde
    termina su nucleo de union. Es una columna adicional del mismo reporte, no
    un filtro -- el veredicto de promiscuidad de NetMHCpan sigue siendo el
    unico criterio de 'Candidato Valido'.

    Args:
        safe_df: Mismos peptidos 'Segura' de la Fase 4 usados por Fase 5.
        output_dir: Carpeta donde persistir el reporte final y el .xls crudo.
        input_stem: Nombre del archivo de entrada sin extension (mismo
            proposito que en Fase 5: evita que corridas sucesivas se pisen).
    """
    n_alleles = len(NETMHCPAN_REFERENCE_PANEL.split(","))
    print(
        f"\n{_SEPARATOR}\nFASE 5b | Promiscuidad T-citotoxica (MHC-I, NetMHCpan-4.2 local, "
        f"{n_alleles} alelo(s) HLA-A/B/C)\n{_SEPARATOR}"
    )

    final_path = output_dir / f"{input_stem}_candidatos_finales_mhc1.csv"

    if safe_df.empty:
        print("No hay peptidos 'Seguros' provenientes de la Fase 4 para evaluar.")
        traceback_df = build_traceback_report_mhci(pd.DataFrame(), safe_df)
        traceback_df.to_csv(final_path, index=False)
        return traceback_df

    # Bug real encontrado y corregido (2026-07-22, al ampliar
    # NETMHCPAN_REFERENCE_PANEL de 12 a 23 alelos): a diferencia de Fase 5
    # (MHC-II, que SI incluye 'allele_panel' en su hash), este checkpoint
    # solo hasheaba 'safe_df' -- un cambio al panel de alelos en el CODIGO
    # FUENTE (sin tocar el input) no invalidaba el cache, sirviendo en
    # silencio un reporte calculado contra un panel viejo. NETMHCPAN_REFERENCE_PANEL
    # no es configurable por CLI (a diferencia del panel de MHC-II, que
    # admite '--alelo-extra'), asi que incluir el valor actual de la
    # constante alcanza para detectar cualquier cambio futuro.
    input_hash = _phase_input_hash(safe_df, NETMHCPAN_REFERENCE_PANEL)
    cached = _load_phase_checkpoint("Fase 5b", final_path, input_hash)
    if cached is not None:
        return cached

    peptides = safe_df["sequence"].tolist()
    print(f"Panel HLA-A/B/C: {NETMHCPAN_REFERENCE_PANEL} | Peptidos a evaluar: {len(peptides)}")

    report = predict_netmhcpan(peptides, output_dir, allele_panel=NETMHCPAN_REFERENCE_PANEL, filename_prefix=f"{input_stem}_")

    if report.empty:
        print("NetMHCpan no devolvio resultados evaluables (revisa longitudes minimas: 8 aa).")
    else:
        print_tc_report(report, allele_panel=NETMHCPAN_REFERENCE_PANEL)

    traceback_df = build_traceback_report_mhci(report, safe_df)

    if not traceback_df.empty:
        cleavage_df = predict_cleavage(peptides, output_dir, filename_prefix=f"{input_stem}_")
        traceback_df = annotate_cterm_cleavage(traceback_df, cleavage_df)
        n_cterm_match = int(traceback_df["netcleave_c_term_match"].sum())
        print(f"Evidencia de corte C-terminal (NetCleave): {n_cterm_match}/{len(traceback_df)} candidato(s).")

    print_traceback_table(traceback_df)

    traceback_df.to_csv(final_path, index=False)
    _write_phase_checkpoint(final_path, input_hash)
    print(f"-> Reporte final MHC-I guardado en: {final_path}")
    return traceback_df


def fase_6_bnab_crossref(safe_df: pd.DataFrame, output_dir: Path, input_stem: str) -> pd.DataFrame:
    """Fase 6: cruza los peptidos 'Seguros' de la Fase 4 contra epitopos de bnAb conocidos (LANL + CATNAP).

    Puramente informativa, no filtra ni condiciona ninguna fase anterior o
    posterior (a diferencia de Fase 4/4b/5/5b, no hay ningun 'veredicto' que
    descarte candidatos aqui): reporta que peptidos coinciden, por
    solapamiento de subcadena, con un epitopo lineal de un anticuerpo
    ampliamente neutralizante ya caracterizado (ver
    ``src.engines.lanl_catnap_engine``). Solo produce matches reales para
    proteinas de la familia HIV Env -- para cualquier otra proteina de
    entrada, un reporte vacio es el resultado ESPERADO, no un fallo.

    Args:
        safe_df: Mismos peptidos 'Segura' de la Fase 4 usados por Fase 4b/4c/5/5b.
        output_dir: Carpeta donde persistir el reporte final.
        input_stem: Nombre del archivo de entrada sin extension (mismo
            proposito que en Fase 4b/4c: evita que corridas sucesivas se pisen).
    """
    print(f"\n{_SEPARATOR}\nFASE 6 | Cruce con bnAb conocidos (LANL Immunology DB + CATNAP, local)\n{_SEPARATOR}")

    final_path = output_dir / f"{input_stem}_bnab_crossref.csv"

    if safe_df.empty:
        print("No hay peptidos 'Seguros' provenientes de la Fase 4 para evaluar.")
        empty_df = pd.DataFrame(columns=[
            "sequence", "antibody_name", "epitope_sequence", "match_length", "epitope_name",
            "hxb2_location", "neutralizing", "antibody_type", "binding_region",
            "catnap_mean_ic50", "catnap_n_viruses",
        ])
        empty_df.to_csv(final_path, index=False)
        return empty_df

    input_hash = _phase_input_hash(safe_df, Settings.LANL_CATNAP_MIN_OVERLAP)
    cached = _load_phase_checkpoint("Fase 6", final_path, input_hash)
    if cached is not None:
        return cached

    peptides = safe_df["sequence"].tolist()
    print(f"Peptidos a evaluar: {len(peptides)}")

    report = query_bnab_crossref(
        peptides,
        Path(Settings.LANL_AB_ALL_PATH),
        catnap_abs_path=Path(Settings.CATNAP_ABS_PATH),
        min_overlap=Settings.LANL_CATNAP_MIN_OVERLAP,
    )

    print_bnab_crossref_report(report)

    report.to_csv(final_path, index=False)
    _write_phase_checkpoint(final_path, input_hash)
    print(f"-> Reporte de cruce bnAb guardado en: {final_path}")
    return report


def fase_7_ensamblaje_constructo(
    safe_df: pd.DataFrame,
    algpred_df: pd.DataFrame,
    stackgly_df: pd.DataFrame,
    htl_df: pd.DataFrame,
    ctl_df: pd.DataFrame,
    output_dir: Path,
    input_stem: str,
) -> Tuple[str, pd.DataFrame]:
    """Fase 7: ensambla automaticamente el constructo multi-epitopo a partir de los candidatos finales.

    Ver docstring completo de ``src.engines.construct_assembly`` para las
    reglas de seleccion top-N y los linkers usados. Puramente interno (sin
    subprocess): selecciona, concatena y persiste el FASTA del constructo +
    su metadata de trazabilidad, insumo de la Fase 8.

    Args:
        safe_df: Peptidos B-cell 'Segura' de la Fase 4.
        algpred_df: Reporte de alergenicidad por peptido de la Fase 4b.
        stackgly_df: Reporte de N-glicosilacion por sequon de la Fase 4c.
        htl_df: Candidatos MHC-II validos de la Fase 5 (con traceback).
        ctl_df: Candidatos MHC-I validos de la Fase 5b (con traceback + anotacion NetCleave).
        output_dir: Carpeta donde persistir el FASTA del constructo y su metadata.
        input_stem: Nombre del archivo de entrada sin extension.

    Returns:
        Tupla ``(construct_sequence, metadata_df)``. ``construct_sequence == ""``
        si ninguna clase aporto candidatos (nada que ensamblar).
    """
    print(f"\n{_SEPARATOR}\nFASE 7 | Ensamblaje automatico del constructo multi-epitopo\n{_SEPARATOR}")

    fasta_path = output_dir / f"{input_stem}_constructo.fasta"
    metadata_path = output_dir / f"{input_stem}_constructo_metadata.csv"

    input_hash = _phase_input_hash(safe_df, algpred_df, stackgly_df, htl_df, ctl_df, Settings.CONSTRUCT_TOP_N_PER_CLASS)
    cached_metadata = _load_phase_checkpoint("Fase 7", metadata_path, input_hash)
    if cached_metadata is not None:
        cached_sequence = "".join(cached_metadata["sequence"]) if not cached_metadata.empty else ""
        if cached_sequence:
            print(f"Constructo ensamblado: {len(cached_sequence)} aa.")
            print(f">{input_stem}_constructo\n{cached_sequence}")
        return cached_sequence, cached_metadata

    construct_sequence, metadata_df = assemble_construct(safe_df, algpred_df, stackgly_df, htl_df, ctl_df)

    if not construct_sequence:
        print("Ningun candidato B-cell/HTL/CTL disponible: no hay nada que ensamblar.")
        metadata_df.to_csv(metadata_path, index=False)
        fasta_path.write_text("")
        return construct_sequence, metadata_df

    n_bcell = int((metadata_df["block"] == "B-cell").sum())
    n_htl = int((metadata_df["block"] == "HTL").sum())
    n_ctl = int((metadata_df["block"] == "CTL").sum())
    print(f"Constructo ensamblado: {len(construct_sequence)} aa ({n_bcell} B-cell + {n_htl} HTL + {n_ctl} CTL, "
          f"top-{Settings.CONSTRUCT_TOP_N_PER_CLASS} por clase).")
    print(f">{input_stem}_constructo\n{construct_sequence}")

    fasta_path.write_text(f">{input_stem}_constructo\n{construct_sequence}\n")
    metadata_df.to_csv(metadata_path, index=False)
    _write_phase_checkpoint(metadata_path, input_hash)
    print(f"-> Constructo guardado en: {fasta_path}")
    print(f"-> Metadata de trazabilidad guardada en: {metadata_path}")
    return construct_sequence, metadata_df


def fase_8_chequeo_constructo(construct_sequence: str, output_dir: Path, input_stem: str) -> pd.DataFrame:
    """Fase 8: chequeo de alergenicidad/toxicidad/antigenicidad/peptido senal del constructo ENSAMBLADO.

    A diferencia de Fase 4b/4c (por peptido individual, antes de saber
    siquiera cuales terminan en el constructo), esta fase corre sobre la
    secuencia COMPLETA del constructo de Fase 7 -- el pedido original de
    Carlos que Fase 4b no cubria por si sola (ver STATUS.md).

    4 motores independientes, cada uno informativo (ninguno filtra ni
    aborta el pipeline; el usuario decide que hacer con el resultado):
    AlgPred2 (alergenicidad, REUSA el motor de Fase 4b, sin instalacion
    nueva), ToxinPred2 (toxicidad, recomendado por el propio grupo Raghava
    para proteinas de longitud completa, a diferencia de ToxinPred3.0
    -pensado para peptidos cortos-), IApred (antigenicidad intrinseca de la
    secuencia completa, reemplazo de VaxiJen -descartado: no open-source, sin
    standalone/API local-) y SignalP-6.0 (peptido senal N-terminal: un
    constructo de fusion para expresion recombinante estandar no deberia
    tener uno predicho).

    Args:
        construct_sequence: Secuencia del constructo ensamblado en Fase 7. Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir el reporte final y los CSV/TXT crudos de cada motor.
        input_stem: Nombre del archivo de entrada sin extension.

    Returns:
        DataFrame de una sola fila (el constructo) con las columnas de los
        4 motores combinadas, o vacio si ``construct_sequence`` esta vacio.
    """
    print(f"\n{_SEPARATOR}\nFASE 8 | Chequeo del constructo ensamblado (alergenicidad/toxicidad/antigenicidad/peptido senal)\n{_SEPARATOR}")

    final_path = output_dir / f"{input_stem}_constructo_chequeo.csv"

    if not construct_sequence:
        print("No hay constructo ensamblado (Fase 7) para evaluar.")
        empty_df = pd.DataFrame(columns=[
            "sequence", "algpred_score", "algpred_veredicto",
            "toxinpred_score", "toxinpred_veredicto",
            "iapred_score", "iapred_categoria",
            "signalp_prediction", "signalp_prob_other", "signalp_prob_sp", "signalp_cs_position",
        ])
        empty_df.to_csv(final_path, index=False)
        return empty_df

    input_hash = _phase_input_hash(construct_sequence)
    cached = _load_phase_checkpoint("Fase 8", final_path, input_hash)
    if cached is not None:
        return cached

    print(f"Constructo a evaluar: {len(construct_sequence)} aa")
    sequences = [construct_sequence]

    print("\n-- Alergenicidad (AlgPred2) --")
    algpred_df = predict_allergenicity(sequences, output_dir, filename_prefix=f"{input_stem}_constructo_")
    print_allergenicity_report(algpred_df)

    print("\n-- Toxicidad (ToxinPred2) --")
    toxinpred_df = predict_toxicity(sequences, output_dir, filename_prefix=f"{input_stem}_constructo_")
    print_toxicity_report(toxinpred_df)

    print("\n-- Antigenicidad intrinseca (IApred) --")
    iapred_df = predict_intrinsic_antigenicity(sequences, output_dir, filename_prefix=f"{input_stem}_constructo_")
    print_iapred_report(iapred_df)

    print("\n-- Peptido senal (SignalP-6.0) --")
    signalp_df = predict_signal_peptide(sequences, output_dir, filename_prefix=f"{input_stem}_constructo_")
    print_signalp_report(signalp_df)

    combined = algpred_df.merge(toxinpred_df, on="sequence").merge(iapred_df, on="sequence").merge(signalp_df, on="sequence")

    combined.to_csv(final_path, index=False)
    _write_phase_checkpoint(final_path, input_hash)
    print(f"\n-> Reporte de chequeo del constructo guardado en: {final_path}")
    return combined


def _resolve_active_engines_and_inputs(
    input_path: Path, output_dir: Path, pdb_mode_override: Optional[str]
) -> Tuple[List[str], Optional[Path], Optional[StructureRecord]]:
    """Enruta ``input_path``, corre Fase 1 o Fase 1.5 segun corresponda, y resuelve los motores activos.

    Encapsula la logica de seleccion de camino (1/2/3) descrita en el
    docstring del modulo, incluyendo el gate no-fatal de Camino 3 (residuos
    no canonicos en el FASTA derivado -> se excluyen BepiPred-3.0/EpiDope
    solo para esta corrida, sin frenar el resto del pipeline).

    Returns:
        Tupla ``(active_engines, clean_fasta, structure_record)``. Exactamente
        uno de ``clean_fasta``/``structure_record`` es no-``None`` segun el
        tipo de input detectado (``structure_record`` en Caminos 2 y 3,
        ``clean_fasta`` siempre que algun motor de secuencia este activo).
    """
    routed = route_input(input_path)

    if routed.input_type == "fasta":
        _, clean_fasta = fase_1_saneamiento(input_path, output_dir)
        return active_engines_for("fasta", None), clean_fasta, None

    structure_record = fase_1_5_estructura(input_path, output_dir)
    pdb_mode = pdb_mode_override or Settings.PDB_PROCESSING_MODE
    active_engines = active_engines_for("structure", pdb_mode)

    clean_fasta: Optional[Path] = None
    if pdb_mode == "structure_and_sequence":
        compatible, invalid_chars = is_bepipred_compatible(structure_record.sequence)
        if compatible:
            clean_fasta = structure_record.fasta_path
        else:
            logger.warning(
                "Accession '%s': %d residuo(s) no canonico(s) (%s) en el FASTA derivado de la "
                "estructura -- se excluye a BepiPred-3.0/EpiDope de esta corrida.",
                structure_record.accession, len(invalid_chars), invalid_chars,
            )
            print(
                f"[AVISO] La secuencia ATMSEQ derivada de '{input_path.name}' tiene "
                f"{len(invalid_chars)} residuo(s) no canonico(s) ({invalid_chars}): se omite "
                "BepiPred-3.0/EpiDope para esta corrida (BepiPred-3.0 los rechaza en bloque). "
                "DiscoTope-3.0/ScanNet corren igual, operan directo sobre el PDB."
            )
            active_engines = [
                key for key in active_engines if ENGINE_REGISTRY[key][1] != "fasta"
            ]

    return active_engines, clean_fasta, structure_record


def main(argv: List[str] = None) -> int:
    """Punto de entrada: enruta el input, ejecuta las fases correspondientes y traduce errores."""
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        active_engines, clean_fasta, structure_record = _resolve_active_engines_and_inputs(
            input_path, output_dir, args.pdb_mode
        )

        raw_dfs = fase_2_antigenicidad(active_engines, input_path.stem, clean_fasta, structure_record, output_dir)
        _log_peak_memory("Fase 2 (antigenicidad)")

        union_df = fase_3_mapeo_y_union(
            raw_dfs, structure_record,
            args.bepipred_threshold, args.bepipred_min_length,
            args.epidope_threshold, args.epidope_min_length,
            output_dir, input_path.stem,
            discotope_threshold=args.discotope_threshold, discotope_min_length=args.discotope_min_length,
            scannet_threshold=args.scannet_threshold, scannet_min_length=args.scannet_min_length,
        )
        masked_union_df = fase_3b_tm_signal_masking(
            raw_dfs, structure_record, union_df, output_dir, input_path.stem
        )
        blast_df = fase_4_tolerancia(
            masked_union_df, args.blast_db, args.identity_threshold, output_dir, input_path.stem
        )
        safe_df = blast_df[blast_df["status"] == "Segura"] if not blast_df.empty else blast_df
        algpred_df = fase_4b_alergenicidad(safe_df, output_dir, input_path.stem)
        _log_peak_memory("Fase 4b (alergenicidad)")
        stackgly_df = fase_4c_glicosilacion(safe_df, output_dir, input_path.stem)
        _log_peak_memory("Fase 4c (N-glicosilacion, StackGlyEmbed -- 3 modelos pesados)")
        htl_df = fase_5_th_promiscuidad(safe_df, output_dir, input_path.stem, allele_extra=args.alelo_extra)
        ctl_df = fase_5b_tc_promiscuidad(safe_df, output_dir, input_path.stem)
        _log_peak_memory("Fase 5b (MHC-I + NetCleave)")
        fase_6_bnab_crossref(safe_df, output_dir, input_path.stem)
        construct_sequence, _ = fase_7_ensamblaje_constructo(
            safe_df, algpred_df, stackgly_df, htl_df, ctl_df, output_dir, input_path.stem
        )
        fase_8_chequeo_constructo(construct_sequence, output_dir, input_path.stem)
        _log_peak_memory("Fase 8 (chequeo del constructo -- SignalP-6.0 es el mas pesado)")
    except PipelineError as exc:
        logger.error("Pipeline detenido: %s", exc)
        print(f"\n[ERROR FATAL] {exc}")
        return 1
    except FileNotFoundError as exc:
        logger.error("Archivo no encontrado: %s", exc)
        print(f"\n[ERROR FATAL] {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - ultima barrera: nunca dejar una traza cruda al usuario
        logger.exception("Error inesperado durante la ejecucion del pipeline.")
        print(f"\n[ERROR INESPERADO] {type(exc).__name__}: {exc}")
        return 1

    print(f"\n{_SEPARATOR}\nPIPELINE COMPLETADO\n{_SEPARATOR}")
    print(f"Todos los resultados se guardaron en: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
