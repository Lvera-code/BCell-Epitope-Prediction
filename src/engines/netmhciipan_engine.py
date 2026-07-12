"""Fase 5: Prediccion de presentacion T-helper (MHC-II) via NetMHCIIpan-4.3 LOCAL.

ADR - pivote metodologico: se descarta MHC-I (2026-07-12)
------------------------------------------------------------
Toda la logica de prediccion de presentacion MHC-I (celulas T citotoxicas
CD8+, servida anteriormente por MHCflurry/NetMHCpan) fue eliminada de este
pipeline. La Fase 5 evalua exclusivamente presentacion MHC-II (celulas
T-helper CD4+), requisito para activar una respuesta humoral sostenida (T-B
cross-talk) en el diseno de vacunas de subunidad.

Este modulo es, igual que ``blast_engine.py`` y ``bepipred_engine.py``, un
wrapper puro de ``subprocess`` sobre un binario local con licencia academica
DTU Health Tech (``Settings.NETMHCIIPAN_HOME``, nunca hardcodeado, resuelto
desde variable de entorno). No se usa ``requests`` ni ninguna llamada de red.

Promiscuidad HLA-DR: en vez de evaluar un unico alelo (insuficiente para
cobertura poblacional), cada peptido candidato se evalua contra
``IEDB_DR_PANEL`` -un panel de referencia de 15 alelos HLA-DR/DRB3/DRB4/DRB5
usado por el IEDB para estimar cobertura poblacional amplia- pasado tal cual
al flag ``-a`` de NetMHCIIpan. Un peptido se reporta como ``'Candidato
Valido'`` (T-helper promiscuo) solo si clasifica como aglutinador fuerte (SB)
o debil (WB), segun los umbrales de %Rank POR DEFECTO del propio
NetMHCIIpan-4.3, en al menos ``Settings.NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES``
alelos distintos del panel.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import ImmunogenicityExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Panel de referencia HLA-DR/DRB3/DRB4/DRB5 usado por el IEDB para estimar
# cobertura poblacional amplia en el diseno de epitopos T-helper (MHC-II).
# NUNCA se le agregan espacios entre comas: NetMHCIIpan lo pasa tal cual a
# su parser de '-a' y un espacio rompe el parseo del alelo siguiente.
IEDB_DR_PANEL = (
    "DRB1_0101,DRB1_0301,DRB1_0401,DRB1_0405,DRB1_0701,DRB1_0802,DRB1_0901,"
    "DRB1_1101,DRB1_1201,DRB1_1302,DRB1_1501,DRB3_0101,DRB3_0202,DRB4_0101,DRB5_0101"
)

# Footprint minimo del core de union a MHC-II: NetMHCIIpan descarta (o
# calcula sobre un core mas corto que el peptido, degradando la prediccion)
# peptidos mas cortos que esto.
_MIN_PEPTIDE_LENGTH = 9

_OUTPUT_COLUMNS = ["sequence", "n_alelos_evaluados", "n_alelos_promiscuos", "min_rank_el", "veredicto"]


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
    automaticamente las columnas 'Rank_EL' repetidas como 'Rank_EL',
    'Rank_EL.1', 'Rank_EL.2', ... (una por alelo, en el mismo orden del
    panel pasado a '-a').

    Args:
        xls_path: Ruta al .xls crudo generado por NetMHCIIpan.
        n_alleles: Numero de alelos evaluados (debe coincidir con el numero
            de columnas 'Rank_EL*' encontradas, si no el .xls esta corrupto
            o el panel no se aplico como se esperaba).

    Returns:
        DataFrame con columnas ``sequence``, ``n_alelos_evaluados``,
        ``n_alelos_promiscuos``, ``min_rank_el`` y ``veredicto``
        (``'Candidato Valido'`` / ``'Rechazado'``).

    Raises:
        ImmunogenicityExecutionError: Si el .xls no se puede parsear o no
            contiene el numero esperado de columnas 'Rank_EL'.
    """
    try:
        raw = pd.read_csv(xls_path, sep="\t", skiprows=2)
    except Exception as exc:
        raise ImmunogenicityExecutionError(f"No se pudo parsear la salida de NetMHCIIpan en '{xls_path}': {exc}") from exc

    rank_cols = [c for c in raw.columns if c == "Rank_EL" or c.startswith("Rank_EL.")]
    if len(rank_cols) != n_alleles or "Peptide" not in raw.columns:
        raise ImmunogenicityExecutionError(
            f"El formato de salida .xls de NetMHCIIpan no coincide con lo esperado: "
            f"se encontraron {len(rank_cols)} columna(s) 'Rank_EL' para {n_alleles} "
            f"alelo(s) evaluado(s). Columnas encontradas: {list(raw.columns)}."
        )

    is_binder = raw[rank_cols] <= Settings.NETMHCIIPAN_RANK_WEAK
    n_alelos_promiscuos = is_binder.sum(axis=1)

    result = pd.DataFrame(
        {
            "sequence": raw["Peptide"],
            "n_alelos_evaluados": n_alleles,
            "n_alelos_promiscuos": n_alelos_promiscuos,
            "min_rank_el": raw[rank_cols].min(axis=1),
        }
    )
    result["veredicto"] = result["n_alelos_promiscuos"].apply(
        lambda n: "Candidato Valido" if n >= Settings.NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES else "Rechazado"
    )
    return result[_OUTPUT_COLUMNS]


def predict_netmhciipan(
    peptides: List[str],
    output_dir: Path,
    allele_panel: str = IEDB_DR_PANEL,
) -> pd.DataFrame:
    """Fase 5: evalua promiscuidad T-helper (MHC-II) via NetMHCIIpan-4.3 local.

    Ejecuta, de forma sincrona (``subprocess.run``), el binario local
    ``./netMHCIIpan-4.3/netMHCIIpan`` sobre los peptidos que superaron el
    filtro de tolerancia inmunologica de la Fase 4 (BLASTp, ``status ==
    'Segura'``), evaluandolos en modo peptido exacto (``-p``, sin digestion
    de proteina) contra el panel completo de alelos HLA-DR indicado.

    Args:
        peptides: Peptidos candidatos que superaron la Fase 4. Los mas
            cortos que el footprint minimo del core de MHC-II (9 aa) se
            omiten con un warning.
        output_dir: Carpeta donde persistir el .xls crudo devuelto por
            NetMHCIIpan, para trazabilidad.
        allele_panel: Alelos HLA-DR separados por coma sin espacios (formato
            NetMHCIIpan, ej. ``"DRB1_0101,DRB1_0301"``), pasados tal cual al
            flag ``-a``. Por defecto ``IEDB_DR_PANEL`` (15 alelos); si se
            necesita cubrir un alelo adicional (ej. especifico de una
            poblacion de interes), se admite sin problema anexandolo al
            string por defecto (ver ``--alelo-extra`` en ``pipeline.py``).

    Returns:
        DataFrame con columnas ``sequence``, ``n_alelos_evaluados``,
        ``n_alelos_promiscuos``, ``min_rank_el`` y ``veredicto``
        (``'Candidato Valido'`` / ``'Rechazado'``). Vacio si ningun peptido
        de entrada alcanza la longitud minima.

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

    n_alleles = len([a for a in allele_panel.split(",") if a])
    output_dir.mkdir(parents=True, exist_ok=True)
    persisted_xls = output_dir / "netmhciipan_raw.xls"

    with tempfile.TemporaryDirectory(prefix="netmhciipan_") as tmp:
        tmp_dir = Path(tmp)
        pep_path = tmp_dir / "peptides.pep"
        xls_path = tmp_dir / "netmhciipan_output.xls"

        pep_path.write_text("\n".join(valid_peptides) + "\n", encoding="utf-8")

        cmd = [
            str(binary), "-p", "-f", str(pep_path),
            "-a", allele_panel,
            "-xls", "-xlsfile", str(xls_path),
        ]
        logger.info("Ejecutando NetMHCIIpan-4.3 local sobre %d peptido(s), %d alelo(s): %s", len(valid_peptides), n_alleles, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=Settings.NETMHCIIPAN_TIMEOUT_SECONDS)
        except subprocess.CalledProcessError as exc:
            raise ImmunogenicityExecutionError(
                f"NetMHCIIpan-4.3 termino con exit code {exc.returncode}: "
                f"{(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ImmunogenicityExecutionError(
                f"NetMHCIIpan-4.3 excedio el tiempo limite de {Settings.NETMHCIIPAN_TIMEOUT_SECONDS}s."
            ) from exc

        result = _parse_xls(xls_path, n_alleles)
        shutil.copyfile(xls_path, persisted_xls)

    return result


def print_th_report(report_df: pd.DataFrame, allele_panel: str = IEDB_DR_PANEL) -> None:
    """Imprime el informe final de promiscuidad T-helper (MHC-II)."""
    if report_df.empty:
        print("No hay peptidos candidatos de la Fase 4 para evaluar contra el panel HLA-DR.")
        return

    n_alleles = len([a for a in allele_panel.split(",") if a])
    seq_width = max(20, report_df["sequence"].str.len().max() + 2)
    header = (
        f"{'Secuencia':<{seq_width}}{'Alelos promiscuos':>19}{'/':>1}{'panel':<7}"
        f"{'Min %Rank':>12}{'Veredicto':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in report_df.itertuples(index=False):
        print(
            f"{row.sequence:<{seq_width}}{row.n_alelos_promiscuos:>19}{'/':>1}{n_alleles:<7}"
            f"{row.min_rank_el:>12.3f}{row.veredicto:>18}"
        )

    n_ok = int((report_df["veredicto"] == "Candidato Valido").sum())
    print(f"\nResumen Fase 5: {n_ok}/{len(report_df)} candidato(s) T-helper promiscuo(s) aprobado(s).")
