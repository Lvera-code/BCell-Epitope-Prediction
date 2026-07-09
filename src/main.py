"""Orquestador CLI del SOTA-B-Epitope-Pipeline.

Encadena el modulo de aduana (saneamiento FASTA), la Fase 1 (cribado de
antigenicidad via 1D-CNN sobre Escalas Z de Hellberg) y la Fase 2 (inferencia
de epitopos via el motor configurado: ESM-2 nativo o wrapper CLI externo),
produciendo un reporte ejecutivo en consola y exportacion CSV trazable.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Sequence

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings
from src.engines.antigenicity_cnn import AntigenicityCNNEngine
from src.engines.epitope_engine import EpitopePredictorFactory
from src.models import AntigenicityResult, EpitopeResult
from src.utils.csv_exporter import CsvExporter
from src.utils.exceptions import CLIWrapperError, EngineExecutionError, ModelLoadError, PipelineError
from src.utils.fasta_parser import FastaParser
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Define y parsea la interfaz de linea de comandos del pipeline.

    Returns:
        Namespace con los argumentos ya parseados y validados por ``argparse``.
    """
    parser = argparse.ArgumentParser(
        description="SOTA-B-Epitope-Pipeline: cribado HTS de epitopos de celulas B (CPU, 1D-CNN + ESM-2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Settings.RAW_FASTA_PATH,
        help="Ruta al archivo FASTA de entrada.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help=(
            "Umbral de antigenicidad global (0.0 a 1.0). Si se omite, se usa el "
            "umbral dinamico F1-optimo persistido junto a la calibracion de Platt "
            "(ver Settings.ANTIGENICITY_CALIBRATION_PATH), o Settings."
            "ANTIGENICITY_THRESHOLD si no hay calibracion. Usa 0.0 para forzar el "
            "mapeo de epitopos en secuencias cortas o fragmentos saltandose el "
            "filtro de la Fase 1."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["esm2", "cli"],
        default=Settings.PREDICTOR_ENGINE if Settings.PREDICTOR_ENGINE in {"esm2", "cli"} else "esm2",
        help="Motor de Fase 2: 'esm2' (nativo, desarrollo local) o 'cli' (subprocess, HPC).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fuerza modo offline: sin llamadas de red a HuggingFace Hub (usa solo cache local).",
    )
    parser.add_argument(
        "--epitope-threshold",
        type=float,
        default=Settings.EPITOPE_THRESHOLD,
        help="Umbral de corte de probabilidad de epitopo para la Fase 2.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Settings.PROCESSED_DIR,
        help="Directorio destino para la exportacion CSV de resultados.",
    )
    return parser.parse_args()


def print_console_report(
    phase1_results: Sequence[AntigenicityResult],
    final_results: List[EpitopeResult],
    is_calibrated: bool = True,
) -> None:
    """Imprime un reporte visual estructurado en forma de tabla ASCII en la terminal.

    Args:
        phase1_results: Resultados completos de la Fase 1 (aprobados y descartados).
        final_results: Resultados de Fase 2 para las secuencias aprobadas.
        is_calibrated: Si ``True``, los scores de Fase 1 provienen de la
            calibracion de Platt; si ``False``, de un sigmoide sin calibrar
            (fallback cuando no existe artefacto de calibracion en disco).
    """
    phase2_map = {r.antigenicity.record.id: r for r in final_results}
    calibration_note = (
        "SCORES CALIBRADOS (Platt Scaling)"
        if is_calibrated
        else "AVISO: scores SIN CALIBRAR (sigmoide crudo) -- ejecute el entrenamiento"
    )

    print("\n┌" + "─" * 85 + "┐")
    print(f"│{'REPORTE FINAL DE CRIBADO - SOTA-B-EPITOPE-PIPELINE':^85}│")
    print(f"│{calibration_note:^85}│")
    print("├" + "─" * 27 + "┬" + "─" * 10 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 20 + "┤")
    print(f"│ {'ID SECUENCIA':<25} │ {'SCORE':<8} │ {'ESTADO':<10} │ {'DENSIDAD B':<10} │ {'REGIONES':<18} │")
    print("├" + "─" * 27 + "┼" + "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 20 + "┤")

    for res1 in phase1_results:
        seq_id = res1.record.id
        display_id = seq_id[:23] + ".." if len(seq_id) > 25 else seq_id
        bypassed_phase1 = res1.method.startswith("Fase1-OMITIDA")
        if bypassed_phase1:
            score_str = "N/A"
            status = "DIRECTA-F2"
        else:
            score_str = f"{res1.score:.4f}"
            status = " APROBADA " if res1.is_antigenic else "DESCARTADA"

        if res1.is_antigenic and seq_id in phase2_map:
            res2 = phase2_map[seq_id]
            epitope_count = sum(1 for residue in res2.residues if residue.is_epitope)
            density = (
                f"{(epitope_count / len(res2.residues)) * 100.0:.2f}%" if res2.residues else "0.00%"
            )
            regions_str = ", ".join(f"{start}-{end}" for start, end in res2.epitope_regions) or "Ninguna"
        else:
            density = "   N/A    "
            regions_str = "-"

        regions_display = regions_str[:16] + ".." if len(regions_str) > 18 else regions_str
        print(
            f"│ {display_id:<25} │ {score_str:<8} │ {status:<10} │ {density:<10} │ "
            f"{regions_display:<18} │"
        )

    print("└" + "─" * 27 + "┴" + "─" * 10 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 20 + "┘\n")


def main() -> int:
    """Ejecuta el pipeline completo de extremo a extremo.

    Returns:
        Codigo de salida del proceso: ``0`` en exito, ``1`` en fallo fatal.
    """
    args = parse_arguments()

    # Offline forzado incondicionalmente: bajo datos moviles/red publica compartida
    # una verificacion de actualizacion contra HuggingFace Hub puede colgarse o
    # fallar a mitad de una demo. El cache local ya contiene todo lo necesario.
    Settings.apply_offline_mode()
    logger.info("Modo offline activado: sin llamadas de red a HuggingFace Hub.")

    Settings.apply_thread_limits()
    Settings.setup_directories()
    Settings.EPITOPE_THRESHOLD = args.epitope_threshold

    # 1. Modulo de Aduana y Saneamiento
    try:
        records = FastaParser.parse(args.input, min_length=Settings.MIN_SEQUENCE_LENGTH)
        logger.info("Se cargaron %d secuencias validas.", len(records))
    except PipelineError as exc:
        logger.critical("Error fatal de formato en el FASTA de entrada: %s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.critical("Archivo de entrada no encontrado: %s", exc)
        return 1

    if not records:
        logger.warning("Ninguna secuencia sobrevivio al saneamiento. Pipeline finalizado sin resultados.")
        return 0

    # 2. Enrutamiento dinamico + Fase 1: Cribado de Antigenicidad (1D-CNN)
    # Secuencias <= Settings.SHORT_PEPTIDE_DIRECT_ROUTING_MAX_LEN omiten la
    # CNN (no calibrada de forma fiable en ese rango, ver Settings) y van
    # directo a Fase 2. El resto sigue el cribado normal.
    routing_max_len = Settings.SHORT_PEPTIDE_DIRECT_ROUTING_MAX_LEN
    short_records = [r for r in records if len(r.sequence) <= routing_max_len]
    long_records = [r for r in records if len(r.sequence) > routing_max_len]

    try:
        antigenicity_engine = AntigenicityCNNEngine(threshold=args.threshold)
        logger.info("FASE 1: Cribado de Antigenicidad (umbral >= %.4f)", antigenicity_engine.threshold)
        is_calibrated = antigenicity_engine.is_calibrated
        logger.info(
            "Fase 1: scores %s.",
            "calibrados via Platt Scaling" if is_calibrated else "SIN CALIBRAR (sigmoide crudo)",
        )
        cnn_results = antigenicity_engine.run(long_records)
    except ModelLoadError as exc:
        logger.critical("Fallo critico cargando el motor de antigenicidad: %s", exc)
        return 1

    bypassed_results: List[AntigenicityResult] = []
    for record in short_records:
        logger.info(
            "[%s] Secuencia <= 25aa detectada. Redireccionando directamente a la "
            "Fase 2 (ESM-2) para mapeo molecular.",
            record.id,
        )
        bypassed_results.append(
            AntigenicityResult(
                record=record,
                score=float("nan"),
                is_antigenic=True,
                method="Fase1-OMITIDA(ruteo-directo<=25aa)",
            )
        )

    results_by_id = {res.record.id: res for res in cnn_results + bypassed_results}
    phase1_results = [results_by_id[record.id] for record in records]

    accepted_candidates = [res for res in phase1_results if res.is_antigenic]
    for res in phase1_results:
        if res.method.startswith("Fase1-OMITIDA"):
            logger.info("[%s] Fase 1 omitida (ruteo directo) -> APROBADA", res.record.id)
        else:
            status = "APROBADA" if res.is_antigenic else "DESCARTADA"
            logger.info("[%s] Score: %.4f -> %s", res.record.id, res.score, status)

    final_results: List[EpitopeResult] = []

    if not accepted_candidates:
        logger.warning("Ninguna secuencia supero el umbral en Fase 1.")
        print_console_report(phase1_results, final_results, is_calibrated)
        return 0

    logger.info(
        "Pasan a Fase 2 (%s): %d de %d secuencias.",
        args.engine,
        len(accepted_candidates),
        len(records),
    )

    # 3. Fase 2: Inferencia de Epitopos (Patron Adaptador)
    logger.info("FASE 2: Deteccion de epitopos de celulas B via motor '%s'.", args.engine)
    predictor = None
    try:
        predictor = EpitopePredictorFactory.create(args.engine)
        final_results = predictor.predict(accepted_candidates)
    except ModelLoadError as exc:
        logger.critical("Fallo critico cargando el motor de epitopos: %s", exc)
        print_console_report(phase1_results, final_results, is_calibrated)
        return 1
    except (EngineExecutionError, CLIWrapperError) as exc:
        logger.error("Fallo durante la inferencia de Fase 2: %s", exc)
        print_console_report(phase1_results, final_results, is_calibrated)
        return 1
    except ValueError as exc:
        logger.critical("Configuracion de motor invalida: %s", exc)
        return 1
    finally:
        if predictor is not None:
            predictor.close()

    # 4. Exportacion y Trazabilidad
    CsvExporter.export(final_results, output_dir=args.output_dir)

    # 5. Reporte Visual en Consola
    print_console_report(phase1_results, final_results, is_calibrated)

    logger.info("Pipeline finalizado sin errores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
