"""Exportación de métricas de cribado y detalle por residuo."""

import pandas as pd
from typing import List
from src.config.settings import Settings
from src.models import EpitopeResult
from src.utils.logger_config import setup_logger

logger = setup_logger()


class CsvExporter:
    @staticmethod
    def export(results: List[EpitopeResult]) -> None:
        if not results:
            logger.warning("No hay resultados para exportar al CSV.")
            return

        # 1. Archivo de Resumen por Proteína
        summary_rows = []
        for r in results:
            regions_str = ", ".join([f"{start}-{end}" for start, end in r.epitope_regions])
            epitope_count = sum(1 for res in r.residues if res.is_epitope)
            density = (epitope_count / len(r.residues)) * 100.0 if r.residues else 0.0

            summary_rows.append({
                "sequence_id": r.antigenicity.record.id,
                "length": len(r.antigenicity.record.sequence),
                "vaxijen_acc_score": round(r.antigenicity.score, 4),
                "passed_phase1": r.antigenicity.is_antigenic,
                "organism_class": r.antigenicity.organism_class.value,
                "epitope_density_pct": round(density, 2),
                "epitope_regions": regions_str or "Ninguna",
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = Settings.PROCESSED_DIR / "ranking_resumen.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Exportado resumen general a: {summary_path}")

        # 2. Archivo de Detalle por Residuo
        detail_rows = []
        for r in results:
            for res in r.residues:
                detail_rows.append({
                    "sequence_id": r.antigenicity.record.id,
                    "position": res.position,
                    "residue": res.residue,
                    "epitope_probability": round(res.epitope_probability, 4),
                    "is_epitope": res.is_epitope,
                })

        detail_df = pd.DataFrame(detail_rows)
        detail_path = Settings.PROCESSED_DIR / "residuos_detalle.csv"
        detail_df.to_csv(detail_path, index=False)
        logger.info(f"Exportado detalle por residuo a: {detail_path}")