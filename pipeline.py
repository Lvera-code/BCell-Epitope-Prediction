#!/usr/bin/env python3
"""Orquestador CLI del pipeline de descubrimiento de epitopos vacunales.

Flujo estricto de 5 fases, cada una consumiendo la salida de la anterior:

    1. Saneamiento del FASTA de entrada (``src.utils.fasta_parser``).
    2. Prediccion de antigenicidad por residuo via DOS motores independientes
       EJECUTADOS EN LOCAL: BepiPred-3.0 (subprocess sobre el codigo fuente
       oficial, ``src.engines.bepipred_engine``) y EpiDope (subprocess via
       conda, codigo abierto, ``src.engines.epidope_engine``). Cada motor
       tiene auto-cache propio en ``fasta_outputs/``.
    3. Mapeo local de regiones de epitopo contiguas por encima de un umbral
       para cada motor (misma fuente que Fase 2, ``src.engines.epitope_mapping``),
       y UNION LOGICA ANOTADA entre ambos (``src.engines.consensus``): TODA
       region detectada por BepiPred y/o por EpiDope avanza a la Fase 4;
       las regiones que solapan entre motores se FUSIONAN (start minimo, end
       maximo, sin recortar a la interseccion) y quedan etiquetadas como
       ``'Consenso'`` en la columna ``origen`` (``'BepiPred'``/``'EpiDope'``
       si solo un motor la detecto). Filtro de longitud inquebrantable: se
       descarta cualquier region final menor a 9 aa antes de la Fase 4.
    4. Filtro de tolerancia inmunologica: BLASTp local contra el proteoma
       humano, descarta homologos de alta identidad (``src.engines.blast_engine``).
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

Todos los artefactos intermedios y el reporte final se guardan en
``fasta_outputs/``. Requiere: instalacion local de BepiPred-3.0 en
``bepipred-3.0b.src/`` y de EpiDope en ``.conda-epidope/`` (descarga/instalacion
manual, ver README.md), NCBI BLAST+ con el proteoma humano indexado en
``reference_db/``, y NetMHCIIpan-4.3 instalado localmente en
``netMHCIIpan-4.3/`` (descarga manual bajo licencia academica DTU Health
Tech, ver README.md).

Ejemplo:
    python pipeline.py --input fasta_inputs/secuencia.fasta
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config.settings import Settings
from src.engines.bepipred_engine import BepiPredEngine
from src.engines.bepipred_engine import extract_epitopes as extract_bepipred_epitopes
from src.engines.blast_engine import print_blast_report, run_blastp_filter
from src.engines.consensus import build_annotated_union_table, print_union_table
from src.engines.epidope_engine import EpidopeEngine
from src.engines.epidope_engine import extract_epitopes as extract_epidope_epitopes
from src.engines.epidope_engine import ACCESSION_COLUMN as EPIDOPE_ACCESSION_COLUMN
from src.engines.epidope_engine import RESIDUE_COLUMN as EPIDOPE_RESIDUE_COLUMN
from src.engines.bepipred_engine import ACCESSION_COLUMN as BEPIPRED_ACCESSION_COLUMN
from src.engines.bepipred_engine import RESIDUE_COLUMN_CANDIDATES as BEPIPRED_RESIDUE_CANDIDATES
from src.engines.epitope_mapping import build_sequence_lookup, print_epitope_table
from src.engines.netmhciipan_engine import (
    IEDB_REFERENCE_PANEL,
    build_traceback_report,
    predict_netmhciipan,
    print_th_report,
    print_traceback_table,
    validate_allele_extra,
)
from src.utils.exceptions import PipelineError
from src.utils.fasta_parser import FastaRecord, load_and_sanitize, write_fasta
from src.utils.logger_config import setup_logger

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
        description="Pipeline de descubrimiento de epitopos vacunales (BepiPred-3.0 + EpiDope + BLASTp + MHC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Ruta al FASTA de entrada (dentro de fasta_inputs/).",
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
        print(
            f"Archivo: {input_path.name} | Registro: {record.accession} | "
            f"Validacion: OK | Longitud: {len(record.sequence)} aa"
        )

    clean_path = output_dir / f"{input_path.stem}_clean.fasta"
    write_fasta(records, clean_path)
    print(f"-> FASTA saneado escrito en: {clean_path}")
    return records, clean_path


def fase_2_antigenicidad(input_stem: str, clean_fasta: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fase 2: obtiene scores crudos de antigenicidad de BepiPred-3.0 y EpiDope (ambos locales), con auto-cache en CSV."""
    print(f"\n{_SEPARATOR}\nFASE 2 | Prediccion de antigenicidad (BepiPred-3.0 + EpiDope, ejecucion local)\n{_SEPARATOR}")

    bepipred_df = _cached_raw_scores(
        engine_name="BepiPred-3.0",
        cache_path=output_dir / f"{input_stem}_bepipred_raw.csv",
        raw_artifacts_dir=output_dir / "_bepipred_raw",
        clean_fasta=clean_fasta,
        engine=BepiPredEngine(),
    )
    epidope_df = _cached_raw_scores(
        engine_name="EpiDope",
        cache_path=output_dir / f"{input_stem}_epidope_raw.csv",
        raw_artifacts_dir=output_dir / "_epidope_raw",
        clean_fasta=clean_fasta,
        engine=EpidopeEngine(),
    )
    return bepipred_df, epidope_df


def _cached_raw_scores(
    engine_name: str, cache_path: Path, raw_artifacts_dir: Path, clean_fasta: Path, engine
) -> pd.DataFrame:
    """Corre ``engine`` sobre ``clean_fasta`` con auto-cache en ``cache_path`` (CSV)."""
    if cache_path.is_file():
        df = pd.read_csv(cache_path)
        print(f"[{engine_name}] Cache local detectada en '{cache_path}'. Se omite la re-ejecucion.")
        print(f"[{engine_name}] Origen de los datos: CACHE LOCAL | Dimensiones de la matriz: {df.shape}")
        return df

    df = engine.run([str(clean_fasta)], output_dir=raw_artifacts_dir)[0]

    df.to_csv(cache_path, index=False)
    print(f"[{engine_name}] Origen de los datos: INFERENCIA LOCAL (subprocess) | Dimensiones de la matriz: {df.shape}")
    print(f"[{engine_name}] -> Resultado crudo cacheado en: {cache_path} (para futuras ejecuciones instantaneas)")
    return df


def fase_3_mapeo_y_union(
    bepipred_raw_df: pd.DataFrame,
    epidope_raw_df: pd.DataFrame,
    bepipred_threshold: float,
    bepipred_min_length: int,
    epidope_threshold: float,
    epidope_min_length: int,
    output_dir: Path,
    input_stem: str,
) -> pd.DataFrame:
    """Fase 3: mapea regiones de epitopo por motor y construye la union logica anotada entre ambos."""
    print(f"\n{_SEPARATOR}\nFASE 3 | Mapeo logico de regiones de epitopo y union anotada BepiPred U EpiDope\n{_SEPARATOR}")

    print(f"-- BepiPred-3.0 (umbral={bepipred_threshold}, min_len={bepipred_min_length}) --")
    bepipred_epitopes_df = extract_bepipred_epitopes(
        bepipred_raw_df, threshold=bepipred_threshold, min_length=bepipred_min_length
    )
    print_epitope_table(
        bepipred_epitopes_df,
        empty_message=f"No se encontraron regiones >= {bepipred_min_length} aa con score medio >= {bepipred_threshold}.",
    )
    bepipred_epitopes_df.to_csv(output_dir / f"{input_stem}_bepipred_epitopes.csv", index=False)

    print(f"\n-- EpiDope (umbral={epidope_threshold}, min_len={epidope_min_length}) --")
    epidope_epitopes_df = extract_epidope_epitopes(
        epidope_raw_df, threshold=epidope_threshold, min_length=epidope_min_length
    )
    print_epitope_table(
        epidope_epitopes_df,
        empty_message=f"No se encontraron regiones >= {epidope_min_length} aa con score medio >= {epidope_threshold}.",
    )
    epidope_epitopes_df.to_csv(output_dir / f"{input_stem}_epidope_epitopes.csv", index=False)

    print("\n-- Union anotada (fusion de solapes, origen BepiPred/EpiDope/Consenso) --")
    # Lookup de secuencia completa por accession: una region fusionada puede
    # exceder el span detectado por cualquiera de los dos motores por
    # separado, asi que la subsecuencia final se reconstruye desde aqui en
    # vez de recortar las subsecuencias individuales de cada motor. Se
    # combinan ambos motores (BepiPred como fuente preferente) porque, tras
    # el saneamiento de Fase 1, ambos reciben el mismo FASTA y deberian
    # coincidir residuo a residuo para la misma accession.
    sequence_lookup = build_sequence_lookup(
        epidope_raw_df, accession_col=EPIDOPE_ACCESSION_COLUMN, residue_col_candidates=(EPIDOPE_RESIDUE_COLUMN,)
    )
    sequence_lookup.update(
        build_sequence_lookup(
            bepipred_raw_df, accession_col=BEPIPRED_ACCESSION_COLUMN, residue_col_candidates=BEPIPRED_RESIDUE_CANDIDATES
        )
    )

    union_df = build_annotated_union_table(bepipred_epitopes_df, epidope_epitopes_df, sequence_lookup)
    print_union_table(union_df)

    out_path = output_dir / f"{input_stem}_union_epitopes.csv"
    union_df.to_csv(out_path, index=False)
    print(f"-> Tabla de union anotada guardada en: {out_path}")
    return union_df


def fase_4_tolerancia(
    union_df: pd.DataFrame,
    blast_db: str,
    identity_threshold: float,
    output_dir: Path,
    input_stem: str,
) -> pd.DataFrame:
    """Fase 4: descarta por BLASTp local los peptidos de la union anotada con alta homologia al proteoma humano."""
    print(f"\n{_SEPARATOR}\nFASE 4 | Filtro de tolerancia inmunologica (BLASTp local, umbral={identity_threshold}%)\n{_SEPARATOR}")

    if union_df.empty:
        print("No hay peptidos de la union anotada de la Fase 3 para analizar.")
        blast_df = union_df.assign(
            blast_task=pd.Series(dtype=str),
            blast_evalue=pd.Series(dtype=float),
            max_pident=pd.Series(dtype=float),
            status=pd.Series(dtype=str),
        )
    else:
        blast_df = run_blastp_filter(union_df, db_path=blast_db, identity_threshold=identity_threshold)
        print_blast_report(blast_df)

    out_path = output_dir / f"{input_stem}_blast_report.csv"
    blast_df.to_csv(out_path, index=False)
    print(f"-> Informe de tolerancia guardado en: {out_path}")
    return blast_df


def fase_5_th_promiscuidad(
    safe_df: pd.DataFrame, output_dir: Path, allele_extra: str = None
) -> pd.DataFrame:
    """Fase 5: evalua promiscuidad T-helper (MHC-II) de los peptidos 'Seguros' de la Fase 4.

    El reporte final (consola y ``candidatos_finales.csv``) no es la salida
    cruda de NetMHCIIpan: los 'Candidato Valido' se enriquecen con su
    traceback a la region de origen de la Fase 3/4 (accession, coordenadas
    reales, origen BepiPred/EpiDope/Consenso, bepipred_score/epidope_score) y
    su nucleo de union de 9 aa, via ``build_traceback_report`` -necesario
    porque en modo proteina (fragmentos largos) NetMHCIIpan devuelve nucleos
    mas cortos que el fragmento evaluado, no el fragmento completo-.

    Args:
        safe_df: Peptidos con ``status == 'Segura'`` provenientes de la Fase 4
            (conserva ``accession``/``start``/``sequence``/``origen``/
            ``bepipred_score``/``epidope_score`` de la Fase 3, usadas como
            tabla padre del traceback).
        output_dir: Carpeta donde persistir el reporte final y el .xls crudo.
        allele_extra: Alelo(s) HLA-DR/DQ/DP adicionales (formato NetMHCIIpan,
            separados por coma sin espacios) a anexar a
            ``IEDB_REFERENCE_PANEL``. Se admiten sin romper el panel por
            defecto.
    """
    allele_panel = f"{IEDB_REFERENCE_PANEL},{allele_extra}" if allele_extra else IEDB_REFERENCE_PANEL
    n_alleles = len(allele_panel.split(","))
    print(f"\n{_SEPARATOR}\nFASE 5 | Promiscuidad T-helper (MHC-II, NetMHCIIpan-4.3 local, {n_alleles} alelo(s) HLA-DR/DQ/DP)\n{_SEPARATOR}")

    final_path = output_dir / "candidatos_finales.csv"

    if safe_df.empty:
        print("No hay peptidos 'Seguros' provenientes de la Fase 4 para evaluar.")
        traceback_df = build_traceback_report(pd.DataFrame(), safe_df)
        traceback_df.to_csv(final_path, index=False)
        return traceback_df

    peptides = safe_df["sequence"].tolist()
    print(f"Panel HLA-DR: {allele_panel} | Peptidos a evaluar: {len(peptides)}")

    report = predict_netmhciipan(peptides, output_dir, allele_panel=allele_panel)

    if report.empty:
        print("NetMHCIIpan no devolvio resultados evaluables (revisa longitudes minimas: 9 aa).")
    else:
        print_th_report(report, allele_panel=allele_panel)

    traceback_df = build_traceback_report(report, safe_df)
    print_traceback_table(traceback_df)

    traceback_df.to_csv(final_path, index=False)
    print(f"-> Reporte final guardado en: {final_path}")
    return traceback_df


def main(argv: List[str] = None) -> int:
    """Punto de entrada: ejecuta las 5 fases en orden y traduce errores a mensajes accionables."""
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, clean_fasta = fase_1_saneamiento(input_path, output_dir)
        bepipred_raw_df, epidope_raw_df = fase_2_antigenicidad(input_path.stem, clean_fasta, output_dir)
        union_df = fase_3_mapeo_y_union(
            bepipred_raw_df, epidope_raw_df,
            args.bepipred_threshold, args.bepipred_min_length,
            args.epidope_threshold, args.epidope_min_length,
            output_dir, input_path.stem,
        )
        blast_df = fase_4_tolerancia(
            union_df, args.blast_db, args.identity_threshold, output_dir, input_path.stem
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
