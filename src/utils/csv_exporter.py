"""Exportacion de metricas de cribado y detalle por residuo a CSV."""

from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import Settings
from src.models import EpitopeResult
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)


class CsvExporter:
    """Serializa los resultados finales del pipeline a disco en formato CSV."""

    @staticmethod
    def export(results: List[EpitopeResult], output_dir: Path = Settings.PROCESSED_DIR) -> None:
        """Exporta un resumen por proteina y un detalle por residuo.

        Args:
            results: Resultados consolidados de Fase 2 a exportar.
            output_dir: Directorio destino de los archivos CSV. Se crea si no
                existe.
        """
        if not results:
            logger.warning("No hay resultados para exportar al CSV.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for result in results:
            regions_str = ", ".join(f"{start}-{end}" for start, end in result.epitope_regions)
            epitope_count = sum(1 for residue in result.residues if residue.is_epitope)
            total_residues = len(result.residues)
            density = (epitope_count / total_residues) * 100.0 if total_residues else 0.0

            summary_rows.append(
                {
                    "sequence_id": result.antigenicity.record.id,
                    "length": len(result.antigenicity.record.sequence),
                    "antigenicity_score": round(result.antigenicity.score, 4),
                    "passed_phase1": result.antigenicity.is_antigenic,
                    "epitope_density_pct": round(density, 2),
                    "epitope_regions": regions_str or "Ninguna",
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "ranking_resumen.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Exportado resumen general a: %s", summary_path)

        detail_rows = []
        for result in results:
            for residue in result.residues:
                detail_rows.append(
                    {
                        "sequence_id": result.antigenicity.record.id,
                        "position": residue.position,
                        "residue": residue.residue,
                        "epitope_probability": round(residue.epitope_probability, 4),
                        "is_epitope": residue.is_epitope,
                    }
                )

        detail_df = pd.DataFrame(detail_rows)
        detail_path = output_dir / "residuos_detalle.csv"
        detail_df.to_csv(detail_path, index=False)
        logger.info("Exportado detalle por residuo a: %s", detail_path)
