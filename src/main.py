"""Orquestador Principal del Pipeline de Cribado HTS (CNB-CSIC)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.settings import Settings
from src.engines.vaxijen_engine import VaxiJenEngine
from src.engines.bepipred_engine import BepiPredEngine
from src.utils.csv_exporter import CsvExporter
from src.utils.exceptions import InvalidSequenceError, EngineExecutionError, ModelLoadError
from src.utils.fasta_parser import FastaParser
from src.utils.logger_config import setup_logger

logger = setup_logger()


def main() -> int:
    Settings.setup_directories()

    # 1. Parseo y Validación FASTA
    try:
        logger.info(f"Cargando secuencias desde: {Settings.RAW_FASTA_PATH}")
        records = FastaParser.parse(Settings.RAW_FASTA_PATH, min_length=Settings.VAXIJEN_LAG + 1)
        logger.info(f"Se cargaron {len(records)} secuencias válidas.")
    except Exception as e:
        logger.critical(f"Error fatal al leer FASTA de entrada: {e}")
        return 1

    # 2. Fase 1: Cribado Grueso de Antigenicidad (VaxiJen)
    logger.info(f"Ejecutando FASE 1: Cribado de Antigenicidad (Umbral >= {Settings.VAXIJEN_THRESHOLD})")
    vaxijen = VaxiJenEngine()
    phase1_results = vaxijen.run(records)

    accepted_candidates = []
    for res in phase1_results:
        status = "APROBADA" if res.is_antigenic else "DESCARTADA"
        logger.info(f"[{res.record.id}] ACC Score: {res.score:.4f} -> {status}")
        if res.is_antigenic:
            accepted_candidates.append(res)

    if not accepted_candidates:
        logger.warning("Ninguna secuencia superó el umbral de VaxiJen en Fase 1. Finalizando pipeline.")
        return 0

    logger.info(f"Pasan a Fase 2 (Deep Learning ESM-2): {len(accepted_candidates)} de {len(records)} secuencias.")

    # 3. Fase 2: Inferencia Conformacional ESM-2 (BepiPred 3.0)
    logger.info("Ejecutando FASE 2: Detección de Epítopos de Células B vía ESM-2")
    try:
        bepipred = BepiPredEngine()
        final_results = bepipred.run(accepted_candidates)
    except ModelLoadError as e:
        logger.critical(f"Fallo crítico cargando motor ESM-2: {e}")
        return 1
    except EngineExecutionError as e:
        logger.error(f"Fallo durante la inferencia en lote: {e}")
        return 1

    # 4. Exportación y Trazabilidad
    CsvExporter.export(final_results)
    logger.info("Pipeline finalizado sin errores")
    return 0


if __name__ == "__main__":
    sys.exit(main())