import argparse
import sys
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings
from src.engines.vaxijen_engine import VaxiJenEngine
from src.engines.bepipred_engine import BepiPredEngine
from src.utils.csv_exporter import CsvExporter
from src.utils.exceptions import ModelLoadError, EngineExecutionError
from src.utils.fasta_parser import FastaParser
from src.utils.logger_config import setup_logger
from src.models import AntigenicityResult, EpitopeResult

logger = setup_logger()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline HTS de Cribado de Epítopos B (CNB-CSIC) | Arquitectura Funnel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Settings.RAW_FASTA_PATH,
        help="Ruta al archivo FASTA de entrada"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=Settings.VAXIJEN_THRESHOLD,
        help="Umbral de corte de antigenicidad para la Fase 1 (VaxiJen ACC)"
    )
    return parser.parse_args()

def imprimir_tabla_consola(phase1_results: Sequence[AntigenicityResult], final_results: List[EpitopeResult]) -> None:
    """Imprime un reporte visual estructurado en forma de tabla ASCII en la terminal."""
    # Crear un mapa para buscar rápido los resultados detallados de la Fase 2
    fase2_map = {r.antigenicity.record.id: r for r in final_results}
    
    print("\n┌" + "─"*85 + "┐")
    print(f"│{'REPORTE FINAL DE CRIBADO':^85}│")
    print("├" + "─"*27 + "┬" + "─"*10 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*20 + "┤")
    print(f"│ {'ID SECUENCIA':<25} │ {'VAXIJEN':<8} │ {'ESTADO':<10} │ {'DENSIDAD B':<10} │ {'REGIONES':<18} │")
    print("├" + "─"*27 + "┼" + "─"*10 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*20 + "┤")
    
    for res1 in phase1_results:
        seq_id = res1.record.id
        # Truncar visualmente IDs muy largos para no romper la alineación de la tabla
        display_id = seq_id[:23] + ".." if len(seq_id) > 25 else seq_id
        vaxi_score = f"{res1.score:.4f}"
        status = " APROBADA " if res1.is_antigenic else "DESCARTADA"
        
        if res1.is_antigenic and seq_id in fase2_map:
            res2 = fase2_map[seq_id]
            epitope_count = sum(1 for res in res2.residues if res.is_epitope)
            density = f"{(epitope_count / len(res2.residues)) * 100.0:.2f}%" if res2.residues else "0.00%"
            regions_str = ", ".join([f"{start}-{end}" for start, end in res2.epitope_regions]) or "Ninguna"
        else:
            density = "   N/A    "
            regions_str = "-"
            
        regions_display = regions_str[:16] + ".." if len(regions_str) > 18 else regions_str
        print(f"│ {display_id:<25} │ {vaxi_score:<8} │ {status:<10} │ {density:<10} │ {regions_display:<18} │")
        
    print("└" + "─"*27 + "┴" + "─"*10 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*20 + "┘\n")


def main() -> int:
    args = parse_arguments()
    Settings.setup_directories()

    input_fasta = args.input
    vaxijen_thresh = args.threshold

    # 1. Parseo y Validación FASTA
    try:
        logger.info(f"Cargando secuencias desde: {input_fasta}")
        records = FastaParser.parse(input_fasta, min_length=Settings.VAXIJEN_LAG + 1)
        logger.info(f"Se cargaron {len(records)} secuencias válidas.")
    except Exception as e:
        logger.critical(f"Error fatal al leer FASTA de entrada: {e}")
        return 1

    # 2. Fase 1: Cribado Grueso de Antigenicidad (VaxiJen)
    logger.info(f"FASE 1: Cribado de Antigenicidad (Umbral >= {vaxijen_thresh})")
    vaxijen = VaxiJenEngine(threshold=vaxijen_thresh)
    phase1_results = vaxijen.run(records)

    accepted_candidates = []
    for res in phase1_results:
        status = "APROBADA" if res.is_antigenic else "DESCARTADA"
        logger.info(f"[{res.record.id}] ACC Score: {res.score:.4f} -> {status}")
        if res.is_antigenic:
            accepted_candidates.append(res)

    final_results: List[EpitopeResult] = []

    if not accepted_candidates:
        logger.warning("Ninguna secuencia superó el umbral en Fase 1.")
        imprimir_tabla_consola(phase1_results, final_results)
        return 0

    logger.info(f"Pasan a Fase 2 (Deep Learning ESM-2): {len(accepted_candidates)} de {len(records)} secuencias.")

    # 3. Fase 2: Inferencia Conformacional ESM-2 (BepiPred 3.0)
    logger.info("FASE 2: Detección de Epítopos de Células B vía ESM-2")
    try:
        bepipred = BepiPredEngine()
        final_results = bepipred.run(accepted_candidates)
    except ModelLoadError as e:
        logger.critical(f"Fallo crítico cargando motor ESM-2: {e}")
        imprimir_tabla_consola(phase1_results, final_results)
        return 1
    except EngineExecutionError as e:
        logger.error(f"Fallo durante la inferencia en lote: {e}")
        imprimir_tabla_consola(phase1_results, final_results)
        return 1

    # 4. Exportación y Trazabilidad
    CsvExporter.export(final_results)
    
    # 5. Reporte Visual en Consola (La adición solicitada)
    imprimir_tabla_consola(phase1_results, final_results)
    
    logger.info("Pipeline finalizado sin errores")
    return 0


if __name__ == "__main__":
    sys.exit(main())