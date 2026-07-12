#!/usr/bin/env python3
"""Orquestador CLI del pipeline de descubrimiento de epitopos vacunales.

Flujo estricto de 5 fases, cada una consumiendo la salida de la anterior:

    1. Saneamiento del FASTA de entrada (``src.utils.fasta_parser``).
    2. Prediccion de antigenicidad por residuo via BepiPred-3.0 EJECUTADO EN
       LOCAL (subprocess sobre el codigo fuente oficial), con auto-cache en
       ``fasta_outputs/`` (``src.engines.bepipred_engine``).
    3. Mapeo local de regiones de epitopo contiguas por encima de un umbral
       (misma fuente que Fase 2).
    4. Filtro de tolerancia inmunologica: BLASTp local contra el proteoma
       humano, descarta homologos de alta identidad (``src.engines.blast_engine``).
    5. Prediccion de presentacion T-helper (MHC-II) via NetMHCIIpan-4.3 LOCAL
       contra un panel de 15 alelos HLA-DR de referencia (IEDB_DR_PANEL);
       reporta como candidato final solo los peptidos "promiscuos" (SB/WB en
       >= 3 alelos del panel) (``src.engines.netmhciipan_engine``).

Todos los artefactos intermedios y el reporte final se guardan en
``fasta_outputs/``. Requiere: instalacion local de BepiPred-3.0 en
``bepipred-3.0b.src/`` (descarga manual, ver README.md), NCBI BLAST+ con el
proteoma humano indexado en ``reference_db/``, y NetMHCIIpan-4.3 instalado
localmente en ``netMHCIIpan-4.3/`` (descarga manual bajo licencia academica
DTU Health Tech, ver README.md).

Ejemplo:
    python pipeline.py --input fasta_inputs/secuencia.fasta
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config.settings import Settings
from src.engines.bepipred_engine import BepiPredEngine, extract_epitopes
from src.engines.blast_engine import print_blast_report, run_blastp_filter
from src.engines.netmhciipan_engine import IEDB_DR_PANEL, predict_netmhciipan, print_th_report
from src.utils.exceptions import PipelineError
from src.utils.fasta_parser import FastaRecord, load_and_sanitize, write_fasta
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

_SEPARATOR = "=" * 70


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos del pipeline."""
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Pipeline de descubrimiento de epitopos vacunales (BepiPred-3.0 + BLASTp + MHC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Ruta al FASTA de entrada (dentro de fasta_inputs/).",
    )
    parser.add_argument(
        "--alelo-extra", default=None,
        help="Alelo(s) HLA-DR adicionales a anexar al panel por defecto de la Fase 5 "
        "(IEDB_DR_PANEL, 15 alelos). Formato NetMHCIIpan, separados por coma sin "
        "espacios (ej. 'DRB1_1602'). No especificar este flag no requiere ninguna "
        "otra accion: el panel por defecto siempre se evalua.",
    )
    parser.add_argument(
        "--output-dir", default=str(Settings.FASTA_OUTPUT_DIR),
        help="Carpeta donde se guardan todos los resultados del pipeline.",
    )
    parser.add_argument(
        "--threshold", type=float, default=Settings.BEPIPRED_THRESHOLD,
        help="Umbral de score de antigenicidad para la ventana deslizante de epitopos (Fase 3).",
    )
    parser.add_argument(
        "--min-length", type=int, default=Settings.BEPIPRED_MIN_EPITOPE_LENGTH,
        help="Longitud minima (aa) de una region de epitopo para no ser descartada (Fase 3).",
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
    """Fase 1: lee, valida y sanea el FASTA de entrada; escribe una copia limpia."""
    print(f"\n{_SEPARATOR}\nFASE 1 | Saneamiento del FASTA de entrada\n{_SEPARATOR}")

    records = load_and_sanitize(input_path)
    for record in records:
        estado = (
            "OK"
            if record.removed_chars == 0
            else f"OK (se eliminaron {record.removed_chars} caracter(es) no canonico(s))"
        )
        print(
            f"Archivo: {input_path.name} | Registro: {record.accession} | "
            f"Validacion: {estado} | Longitud final: {len(record.sequence)} aa"
        )

    clean_path = output_dir / f"{input_path.stem}_clean.fasta"
    write_fasta(records, clean_path)
    print(f"-> FASTA saneado escrito en: {clean_path}")
    return records, clean_path


def fase_2_antigenicidad(input_stem: str, clean_fasta: Path, output_dir: Path) -> pd.DataFrame:
    """Fase 2: obtiene scores crudos de BepiPred-3.0 (ejecucion local), con auto-cache en CSV."""
    print(f"\n{_SEPARATOR}\nFASE 2 | Prediccion de antigenicidad (BepiPred-3.0, ejecucion local)\n{_SEPARATOR}")

    cache_path = output_dir / f"{input_stem}_bepipred_raw.csv"
    if cache_path.is_file():
        df = pd.read_csv(cache_path)
        print(f"Cache local detectada en '{cache_path}'. Se omite la re-ejecucion de BepiPred-3.0.")
        print(f"Origen de los datos: CACHE LOCAL | Dimensiones de la matriz: {df.shape}")
        return df

    engine = BepiPredEngine()

    raw_artifacts_dir = output_dir / "_bepipred_raw"
    df = engine.run([str(clean_fasta)], output_dir=raw_artifacts_dir)[0]

    df.to_csv(cache_path, index=False)
    print(f"Origen de los datos: INFERENCIA LOCAL (subprocess) | Dimensiones de la matriz: {df.shape}")
    print(f"-> Resultado crudo cacheado en: {cache_path} (para futuras ejecuciones instantaneas)")
    return df


def fase_3_mapeo(
    raw_df: pd.DataFrame, threshold: float, min_length: int, output_dir: Path, input_stem: str
) -> pd.DataFrame:
    """Fase 3: agrupa residuos contiguos por encima del umbral en regiones de epitopo."""
    print(f"\n{_SEPARATOR}\nFASE 3 | Mapeo logico de regiones de epitopo (umbral={threshold}, min_len={min_length})\n{_SEPARATOR}")

    epitopes_df = extract_epitopes(raw_df, threshold=threshold, min_length=min_length)

    if epitopes_df.empty:
        print(f"No se encontraron regiones >= {min_length} aa con score medio >= {threshold}.")
    else:
        header = f"{'N.Region':<10}{'Coordenadas':<16}{'Score Medio':>13}  Secuencia"
        print(header)
        print("-" * len(header))
        for i, row in enumerate(epitopes_df.itertuples(index=False), start=1):
            coords = f"{row.start}-{row.end}"
            print(f"{i:<10}{coords:<16}{row.mean_score:>13.4f}  {row.sequence}")

    out_path = output_dir / f"{input_stem}_epitopes.csv"
    epitopes_df.to_csv(out_path, index=False)
    print(f"-> Tabla de regiones guardada en: {out_path}")
    return epitopes_df


def fase_4_tolerancia(
    epitopes_df: pd.DataFrame,
    blast_db: str,
    identity_threshold: float,
    output_dir: Path,
    input_stem: str,
) -> pd.DataFrame:
    """Fase 4: descarta por BLASTp local los peptidos con alta homologia al proteoma humano."""
    print(f"\n{_SEPARATOR}\nFASE 4 | Filtro de tolerancia inmunologica (BLASTp local, umbral={identity_threshold}%)\n{_SEPARATOR}")

    if epitopes_df.empty:
        print("No hay peptidos candidatos de la Fase 3 para analizar.")
        blast_df = epitopes_df.assign(
            blast_task=pd.Series(dtype=str),
            blast_evalue=pd.Series(dtype=float),
            max_pident=pd.Series(dtype=float),
            status=pd.Series(dtype=str),
        )
    else:
        blast_df = run_blastp_filter(epitopes_df, db_path=blast_db, identity_threshold=identity_threshold)
        print_blast_report(blast_df)

    out_path = output_dir / f"{input_stem}_blast_report.csv"
    blast_df.to_csv(out_path, index=False)
    print(f"-> Informe de tolerancia guardado en: {out_path}")
    return blast_df


def fase_5_th_promiscuidad(
    safe_df: pd.DataFrame, output_dir: Path, allele_extra: str = None
) -> pd.DataFrame:
    """Fase 5: evalua promiscuidad T-helper (MHC-II) de los peptidos 'Seguros' de la Fase 4.

    Args:
        safe_df: Peptidos con ``status == 'Segura'`` provenientes de la Fase 4.
        output_dir: Carpeta donde persistir el reporte final y el .xls crudo.
        allele_extra: Alelo(s) HLA-DR adicionales (formato NetMHCIIpan,
            separados por coma sin espacios) a anexar a ``IEDB_DR_PANEL``. Se
            admiten sin romper el panel por defecto.
    """
    allele_panel = f"{IEDB_DR_PANEL},{allele_extra}" if allele_extra else IEDB_DR_PANEL
    n_alleles = len(allele_panel.split(","))
    print(f"\n{_SEPARATOR}\nFASE 5 | Promiscuidad T-helper (MHC-II, NetMHCIIpan-4.3 local, {n_alleles} alelo(s) HLA-DR)\n{_SEPARATOR}")

    final_path = output_dir / "candidatos_finales.csv"

    if safe_df.empty:
        print("No hay peptidos 'Seguros' provenientes de la Fase 4 para evaluar.")
        report = pd.DataFrame(columns=["sequence", "n_alelos_evaluados", "n_alelos_promiscuos", "min_rank_el", "veredicto"])
        report.to_csv(final_path, index=False)
        return report

    peptides = safe_df["sequence"].tolist()
    print(f"Panel HLA-DR: {allele_panel} | Peptidos a evaluar: {len(peptides)}")

    report = predict_netmhciipan(peptides, output_dir, allele_panel=allele_panel)

    if report.empty:
        print("NetMHCIIpan no devolvio resultados evaluables (revisa longitudes minimas: 9 aa).")
    else:
        print_th_report(report, allele_panel=allele_panel)

    report.to_csv(final_path, index=False)
    print(f"-> Reporte final guardado en: {final_path}")
    return report


def main(argv: List[str] = None) -> int:
    """Punto de entrada: ejecuta las 5 fases en orden y traduce errores a mensajes accionables."""
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, clean_fasta = fase_1_saneamiento(input_path, output_dir)
        raw_df = fase_2_antigenicidad(input_path.stem, clean_fasta, output_dir)
        epitopes_df = fase_3_mapeo(raw_df, args.threshold, args.min_length, output_dir, input_path.stem)
        blast_df = fase_4_tolerancia(
            epitopes_df, args.blast_db, args.identity_threshold, output_dir, input_path.stem
        )
        safe_df = blast_df[blast_df["status"] == "Segura"] if not blast_df.empty else blast_df
        fase_5_th_promiscuidad(safe_df, output_dir, allele_extra=args.alelo_extra)
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
