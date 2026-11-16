"""Cytotoxic T-cell immunogenicity (MHC-I) via LOCAL NetMHCpan-4.2.

ADR: MHC-I discarded, then reverted
----------------------------------------
See the docstring of ``netmhciipan_engine.py`` for the full history. This
module is the reintroduction of MHC-I, as part of the extended set of
construct checks (tox/aller/antigenicity, N-glyco, TM/signal,
bnAb cross-ref) beyond the original 5-phase pipeline. It is deliberately
NOT merged with Phase 5 (NetMHCIIpan, T-helper/MHC-II): they are biologically
distinct antigen presentation pathways (professional presenting
cell vs. any nucleated cell; cytotoxic CD8+ vs. helper CD4+;
different peptide length rules -8-11 aa vs. a 9 aa core inside
a flexible window-), so they are reported as independent verdicts in
parallel, never mixed into a single "promiscuity" figure.

Just like ``netmhciipan_engine.py``, ``blast_engine.py`` and
``bepipred_engine.py``: a pure ``subprocess`` wrapper over a local binary
with a DTU Health Tech academic licence (``Settings.NETMHCPAN_HOME``, never
hardcoded). Neither ``requests`` nor any network call is used.

HLA-I promiscuity: each candidate peptide is evaluated against
``Settings.NETMHCPAN_REFERENCE_PANEL`` -a reference panel of representative
HLA-A/B/C alleles, passed as-is to the ``-a`` flag. A peptide is
reported as ``'Candidato Valido'`` (promiscuous cytotoxic T-cell) only if it
classifies as a strong (SB) or weak (WB) binder, according to
``Settings.NETMHCPAN_RANK_STRONG``/``NETMHCPAN_RANK_WEAK`` (0.5/2.0 by
default, DIFFERENT from those of NetMHCIIpan, 1.0/5.0: MHC-I has its own
%Rank scale, not comparable 1:1), in at least
``Settings.NETMHCPAN_MIN_PROMISCUOUS_ALLELES`` distinct alleles of the panel.

**Panel composition:**
- **HLA-A/B (12 alleles, unchanged since the original design):**
  representative of the 12 HLA class I supertypes of Sidney et al. 2008
  ("HLA class I supertypes: a revised and updated classification",
  Immunome Res 4:2). That paper classifies ~700 HLA-A/-B alleles into 12
  supertypes, but does NOT cover HLA-C at all (confirmed by reading the
  paper). The IEDB panel of 27 alleles for MHC-II
  (``netmhciipan_engine.IEDB_REFERENCE_PANEL``) is not the precedent
  here either: the most cited HOMOLOGOUS MHC-I panel for population
  coverage (Weiskopf et al. 2013, PNAS, "Comprehensive analysis of dengue
  virus-specific responses...", the "27 allele panel" that IEDB
  popularized for class I) is ALSO exclusively HLA-A/B (16 A + 11 B),
  meaning that the historical absence of HLA-C in this module was not a
  deliberate decision of this project, but reproduced a real and
  well documented omission in the two most cited references of the field
  for MHC-I panels.
- **HLA-C (11 alleles):** `HLA-C01:02`, `HLA-C03:04`,
  `HLA-C04:01`, `HLA-C05:01`, `HLA-C06:02`, `HLA-C07:01`, `HLA-C07:02`,
  `HLA-C08:02`, `HLA-C12:03`, `HLA-C15:02`, `HLA-C16:01`. HLA-C IS a
  real CD8+ restriction element (only historically less
  characterized and expressed at the cell surface at a lower level than
  HLA-A/B). There is no classification of HLA-C "supertypes" as
  clean/unique as that of Sidney 2008 for A/B (the C1/C2 division that IS
  well established is about residues 77/80 relevant for ligands
  of NK cell KIR receptors, a classification axis different from that of
  T cell peptide binding specificity). These 11 alleles were
  chosen for being the ones that appear consistently in two
  independent sources: (a) Rasmussen et al. 2014, J Immunol 193(10):4790-4801,
  "Uncovering the Peptide-Binding Specificities of HLA-C...", 17 of the
  most globally common HLA-C alleles, chosen by population
  frequency and characterized experimentally (real binding motifs,
  not merely inferred); and (b) the criterion of a >=1% global population
  frequency threshold that IEDB recommends for population coverage
  panels, as applied by a 2020 SARS-CoV-2 vaccine design
  study that evaluated HLA-A/B/C together to maximize coverage
  (PMC7754929). The 11 chosen appear in both sources. Confirmed as
  supported by the local binary (``netMHCpan-4.2/data/allelenames`` and
  ``MHC_pseudo.dat``, verified before adding them) and validated with a
  real run against known reference epitopes (NLVPMVATV/CMV
  pp65, GILGFVFTL/Flu M1, both HLA-A*02:01) without errors.
- **``NETMHCPAN_MIN_PROMISCUOUS_ALLELES`` (3) was left UNCHANGED** when
  extending the panel from 12 to 23 alleles. This is a deliberate decision, not an
  oversight: it follows the same convention already established by
  ``netmhciipan_engine.IEDB_REFERENCE_PANEL`` (27 alleles, the same fixed
  threshold of 3, never scaled as a fraction of the panel size). That said, 3
  of 23 is a relatively more lax bar than 3 of 12. If in the future
  it is decided that "promiscuity" should scale with the panel size,
  this is the place to review; redefining that criterion is out of the scope
  of adding HLA-C to the panel.

Unlike NetMHCIIpan, the NetMHCpan-4.2 .xls does NOT have an
``Inverted`` column: the MHC-I binding core does not suffer the "backwards"
alignment artifact of the MHC-II Gibbs sampling training (a groove closed at
both ends, with no room to invert the register), so no
equivalent filter is needed here, confirmed by reading the real columns of the
.xls (see ``_parse_xls``), not assumed by analogy with NetMHCIIpan.

Buffer overflow of the binary in exact peptide mode: just like NetMHCIIpan,
NetMHCpan-4.2 (Linux_x86_64) fails with "*** buffer overflow detected ***"
(SIGABRT, core dump, exit code 0, the tcsh wrapper does NOT propagate the failure) for
peptides that are too long in ``-p`` mode: 55 aa OK, 57 aa crash, verified
both with the panel of 12 and with that of 23 alleles. The limit does NOT change with
the panel size, it is a property of the internal parsing of the peptide
length, not
of how many alleles are passed via '-a' (the same binary/buffer architecture
as NetMHCIIpan, whose limit also falls in that range). Peptides longer
than ``_MAX_PEPTIDE_MODE_LENGTH`` are routed to FASTA/protein mode (``-l``,
NetMHCpan's internal sliding window over the lengths of
``Settings.NETMHCPAN_PEPTIDE_LENGTHS``), just as ``netmhciipan_engine.py`` does.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import ImmunogenicityExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

# Panel de referencia de 23 alelos HLA-A/B/C. HLA-A/B (12): representativos
# de los 12 supertipos de Sidney et al. 2008 (no cubre HLA-C). HLA-C (11):
# comunes globalmente segun Rasmussen et al. 2014 + criterio de frecuencia
# poblacional >=1% de IEDB -- ver docstring del modulo para el detalle
# completo y las fuentes.
# NUNCA se le agregan espacios entre comas: NetMHCpan lo pasa tal cual a su
# parser de '-a' y un espacio rompe el parseo del alelo siguiente.
NETMHCPAN_REFERENCE_PANEL = (
    "HLA-A01:01,HLA-A02:01,HLA-A03:01,HLA-A24:02,HLA-A26:01,"
    "HLA-B07:02,HLA-B08:01,HLA-B27:05,HLA-B39:01,HLA-B40:01,HLA-B58:01,HLA-B15:01,"
    "HLA-C01:02,HLA-C03:04,HLA-C04:01,HLA-C05:01,HLA-C06:02,HLA-C07:01,"
    "HLA-C07:02,HLA-C08:02,HLA-C12:03,HLA-C15:02,HLA-C16:01"
)

# Footprint minimo de un peptido MHC-I: por debajo de 8 aa no hay nucleo de
# union viable en ningun alelo HLA-I conocido.
_MIN_PEPTIDE_LENGTH = 8

# Longitud maxima segura para el modo peptido exacto ('-p'). El binario
# NetMHCpan-4.2 (Linux_x86_64) revienta con "*** buffer overflow detected
# ***" (SIGABRT, core dump) en ese modo para entradas demasiado largas --
# verificado contra NETMHCPAN_REFERENCE_PANEL (23 alelos): 55 aa OK, 57 aa
# crash. El wrapper 'netMHCpan' (tcsh) NO propaga ese crash como
# exit code distinto de cero (ver ``_require_xls_output``). Se deja un margen
# de seguridad considerable (40, no 55) por si el limite real del binario
# varia entre builds o paneles de alelos -- mismo valor y misma logica que
# ``netmhciipan_engine._MAX_PEPTIDE_MODE_LENGTH``, coincidencia porque ambos
# binarios comparten arquitectura de buffer, no porque se haya copiado sin
# verificar.
_MAX_PEPTIDE_MODE_LENGTH = 40

_OUTPUT_COLUMNS = [
    "sequence", "core_9aa", "n_alelos_evaluados", "n_alelos_promiscuos", "promiscuous_alleles", "min_rank_el", "veredicto",
]

_TRACEBACK_BASE_COLUMNS = [
    "accession", "sequence_f5", "core_9aa", "start", "end", "origen",
    "n_alelos_promiscuos", "promiscuous_alleles", "n_alelos_evaluados", "min_rank_el",
]


def _traceback_columns(parent_df: pd.DataFrame) -> List[str]:
    """Columnas fijas mas '{motor}_score' por cada motor presente en ``parent_df``."""
    score_columns = [c for c in parent_df.columns if c.endswith("_score")]
    return _TRACEBACK_BASE_COLUMNS + score_columns


def _resolve_binary() -> Path:
    """Localiza el script local de NetMHCpan-4.2 y valida que sea ejecutable.

    Raises:
        ImmunogenicityExecutionError: Con instrucciones de instalacion si el
            script no existe o no tiene permiso de ejecucion.
    """
    binary = Settings.NETMHCPAN_HOME / Settings.NETMHCPAN_BINARY_NAME
    if not binary.is_file():
        raise ImmunogenicityExecutionError(
            f"No se encontro el script local de NetMHCpan-4.2 en '{binary}'. Por "
            "restricciones de licencia academica, DTU Health Tech no permite "
            "redistribuir el paquete: descargalo manualmente desde "
            f"{Settings.NETMHCPAN_DOWNLOAD_URL} (seccion 'Downloads', requiere "
            "cuenta academica), descomprimelo en la raiz del proyecto como "
            "'netMHCpan-4.2/' (o apunta la variable de entorno NETMHCPAN_HOME "
            "a su ubicacion), edita la linea 'NMHOME' del script 'netMHCpan' con "
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


def _parse_xls(xls_path: Path, n_alleles: int, allele_names: List[str]) -> pd.DataFrame:
    """Parsea el .xls de NetMHCpan-4.2 y evalua la promiscuidad de cada peptido.

    El .xls multi-alelo de NetMHCpan-4.2 tiene el mismo formato de 2 filas de
    cabecera que NetMHCIIpan (comentario + fila de nombres de alelo), pero
    con columnas por alelo DISTINTAS: ``core``/``icore``/``EL_score``/
    ``EL_rank`` (minuscula), sin columna ``Inverted`` -- MHC-I no sufre el
    artefacto de orientacion invertida de MHC-II (ver docstring del modulo),
    asi que no hace falta ningun filtro equivalente al de
    ``netmhciipan_engine._parse_xls``. ``pandas`` desambigua las columnas
    repetidas como ``EL_rank``, ``EL_rank.1``, ... (una por alelo, mismo
    orden del panel pasado a '-a').

    Args:
        xls_path: Ruta al .xls crudo devuelto por NetMHCpan.
        n_alleles: Numero de alelos evaluados (debe coincidir con el numero
            de columnas 'EL_rank*'/'core*' encontradas).
        allele_names: Alelos del panel EN EL MISMO ORDEN pasado a '-a', usado
            para nombrar ``promiscuous_alleles`` (ver docstring de
            ``netmhciipan_engine._parse_xls``, misma logica).

    Returns:
        DataFrame con columnas ``sequence``, ``core_9aa``,
        ``n_alelos_evaluados``, ``n_alelos_promiscuos``, ``promiscuous_alleles``,
        ``min_rank_el`` y ``veredicto`` (``'Candidato Valido'`` / ``'Rechazado'``).
        ``core_9aa`` es el nucleo de union (columna ``core`` de NetMHCpan) del
        alelo con el %Rank mas bajo para ese peptido. ``promiscuous_alleles``:
        ver docstring de ``netmhciipan_engine._parse_xls`` (misma logica,
        necesaria para cobertura poblacional real, ``src.engines.population_coverage``).

    Raises:
        ImmunogenicityExecutionError: Si el .xls no se puede parsear o no
            contiene el numero esperado de columnas 'EL_rank'/'core'.
    """
    try:
        raw = pd.read_csv(xls_path, sep="\t", skiprows=2)
    except Exception as exc:
        raise ImmunogenicityExecutionError(f"No se pudo parsear la salida de NetMHCpan en '{xls_path}': {exc}") from exc

    rank_cols = [c for c in raw.columns if c == "EL_rank" or c.startswith("EL_rank.")]
    core_cols = [c for c in raw.columns if c == "core" or c.startswith("core.")]
    if len(rank_cols) != n_alleles or len(core_cols) != n_alleles or "Peptide" not in raw.columns:
        raise ImmunogenicityExecutionError(
            f"El formato de salida .xls de NetMHCpan no coincide con lo esperado: "
            f"se encontraron {len(rank_cols)} columna(s) 'EL_rank' y {len(core_cols)} "
            f"columna(s) 'core' para {n_alleles} alelo(s) evaluado(s). "
            f"Columnas encontradas: {list(raw.columns)}."
        )

    rank_matrix = raw[rank_cols].to_numpy()
    core_matrix = raw[core_cols].to_numpy()
    row_idx = np.arange(len(raw))

    best_allele_idx = rank_matrix.argmin(axis=1)
    best_core = core_matrix[row_idx, best_allele_idx]

    is_binder = rank_matrix <= Settings.NETMHCPAN_RANK_WEAK
    n_alelos_promiscuos = is_binder.sum(axis=1)
    promiscuous_alleles = [
        ",".join(allele_names[j] for j in range(n_alleles) if is_binder[i, j])
        for i in range(len(raw))
    ]

    result = pd.DataFrame(
        {
            "sequence": raw["Peptide"],
            "core_9aa": best_core,
            "n_alelos_evaluados": n_alleles,
            "n_alelos_promiscuos": n_alelos_promiscuos,
            "promiscuous_alleles": promiscuous_alleles,
            "min_rank_el": rank_matrix.min(axis=1),
        }
    )
    result["veredicto"] = result["n_alelos_promiscuos"].apply(
        lambda n: "Candidato Valido" if n >= Settings.NETMHCPAN_MIN_PROMISCUOUS_ALLELES else "Rechazado"
    )
    return result[_OUTPUT_COLUMNS]


def _run_netmhcpan(
    binary: Path, mode_args: List[str], allele_panel: str, xls_path: Path, timeout: int
) -> subprocess.CompletedProcess:
    """Invoca el binario local con ``mode_args`` (formato de entrada) + panel + salida .xls."""
    cmd = [str(binary)] + mode_args + ["-a", allele_panel, "-xls", "-xlsfile", str(xls_path)]
    logger.info("Ejecutando NetMHCpan-4.2 local: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise ImmunogenicityExecutionError(
            f"NetMHCpan-4.2 termino con exit code {exc.returncode}: "
            f"{(exc.stderr or '<sin stderr>')[:2000]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ImmunogenicityExecutionError(f"NetMHCpan-4.2 excedio el tiempo limite de {timeout}s.") from exc


def _require_xls_output(xls_path: Path, proc: subprocess.CompletedProcess, mode_desc: str) -> None:
    """Valida que el .xls prometido exista: el wrapper tcsh no propaga fallos internos como exit != 0.

    Causas conocidas: (a) modo peptido exacto con una entrada > ~55 aa
    (buffer overflow del binario, ver ``_MAX_PEPTIDE_MODE_LENGTH`` --
    ``predict_netmhcpan`` ya enruta para evitar esto), (b) la linea 'NMHOME'
    dentro del script wrapper apunta a una ruta desactualizada.
    """
    if xls_path.is_file():
        return
    raise ImmunogenicityExecutionError(
        f"NetMHCpan-4.2 ({mode_desc}) termino sin error (exit 0) pero no genero el archivo "
        f"de salida esperado en '{xls_path}'. Causas conocidas: un peptido de entrada excede "
        f"el limite del modo usado (revisa Settings._MAX_PEPTIDE_MODE_LENGTH), o la linea "
        f"'NMHOME' dentro de '{Settings.NETMHCPAN_HOME / Settings.NETMHCPAN_BINARY_NAME}' "
        f"apunta a una ruta desactualizada (p. ej. si moviste la carpeta del proyecto) -en ese "
        f"caso, edita esa linea con la ruta absoluta ACTUAL de "
        f"'{Settings.NETMHCPAN_HOME.resolve()}' y vuelve a intentarlo-. "
        f"Salida del proceso: {(proc.stdout or '<vacia>')[:1000]}"
    )


def predict_netmhcpan(
    peptides: List[str],
    output_dir: Path,
    allele_panel: str = NETMHCPAN_REFERENCE_PANEL,
    filename_prefix: str = "",
) -> pd.DataFrame:
    """Evalua promiscuidad T-citotoxica (MHC-I) via NetMHCpan-4.2 local.

    Mismo patron que ``netmhciipan_engine.predict_netmhciipan``, adaptado a
    las diferencias de NetMHCpan-4.2 (ver docstring del modulo): sin filtro
    de alelos invertidos, longitudes de peptido MHC-I (8-11 aa canonico) en
    vez de la ventana fija de 15 aa de MHC-II.

    Args:
        peptides: Peptidos candidatos a evaluar (tipicamente los mismos que
            superaron la Fase 4, igual que NetMHCIIpan). Los mas cortos que
            el footprint minimo de MHC-I (8 aa) se omiten con un warning.
        output_dir: Carpeta donde persistir el/los .xls crudos, para trazabilidad.
        allele_panel: Alelos HLA-A/B/C separados por coma sin espacios (formato
            NetMHCpan), pasados tal cual al flag ``-a``. Por defecto
            ``NETMHCPAN_REFERENCE_PANEL`` (23 alelos: 12 HLA-A/B de los
            supertipos Sidney 2008 + 11 HLA-C, ver docstring del modulo).
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para los
            .xls crudos persistidos en ``output_dir``.

    Returns:
        DataFrame con columnas ``sequence``, ``core_9aa``,
        ``n_alelos_evaluados``, ``n_alelos_promiscuos``, ``min_rank_el`` y
        ``veredicto``. Vacio si ningun peptido de entrada alcanza la longitud
        minima.

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
            "%d peptido(s) mas cortos que el footprint minimo de MHC-I (%d aa) fueron omitidos.",
            skipped, _MIN_PEPTIDE_LENGTH,
        )
    if not valid_peptides:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    short_peptides = [p for p in valid_peptides if len(p) <= _MAX_PEPTIDE_MODE_LENGTH]
    long_peptides = [p for p in valid_peptides if len(p) > _MAX_PEPTIDE_MODE_LENGTH]
    if long_peptides:
        logger.info(
            "%d peptido(s) > %d aa se evaluaran en modo proteina (ventana deslizante de "
            "NetMHCpan sobre longitudes %s) para evitar el buffer overflow conocido del modo "
            "peptido exacto con entradas largas.",
            len(long_peptides), _MAX_PEPTIDE_MODE_LENGTH, Settings.NETMHCPAN_PEPTIDE_LENGTHS,
        )

    allele_names = [a for a in allele_panel.split(",") if a]
    n_alleles = len(allele_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_frames = []
    with tempfile.TemporaryDirectory(prefix="netmhcpan_") as tmp:
        tmp_dir = Path(tmp)

        if short_peptides:
            pep_path = tmp_dir / "peptides.pep"
            pep_path.write_text("\n".join(short_peptides) + "\n", encoding="utf-8")
            xls_path = tmp_dir / "peptide_mode_output.xls"
            proc = _run_netmhcpan(
                binary, ["-p", "-f", str(pep_path)], allele_panel, xls_path, Settings.NETMHCPAN_TIMEOUT_SECONDS
            )
            _require_xls_output(xls_path, proc, mode_desc="modo peptido exacto")
            result_frames.append(_parse_xls(xls_path, n_alleles, allele_names))
            shutil.copyfile(xls_path, output_dir / f"{filename_prefix}netmhcpan_raw_peptide_mode.xls")

        if long_peptides:
            fasta_path = tmp_dir / "fragments.fasta"
            with fasta_path.open("w", encoding="utf-8") as fh:
                for i, seq in enumerate(long_peptides):
                    fh.write(f">candidato_{i}\n{seq}\n")
            xls_path = tmp_dir / "protein_mode_output.xls"
            proc = _run_netmhcpan(
                binary, ["-f", str(fasta_path), "-l", Settings.NETMHCPAN_PEPTIDE_LENGTHS],
                allele_panel, xls_path, Settings.NETMHCPAN_TIMEOUT_SECONDS,
            )
            _require_xls_output(xls_path, proc, mode_desc="modo proteina (ventana deslizante)")
            result_frames.append(_parse_xls(xls_path, n_alleles, allele_names))
            shutil.copyfile(xls_path, output_dir / f"{filename_prefix}netmhcpan_raw_protein_mode.xls")

    if not result_frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.concat(result_frames, ignore_index=True)


def print_tc_report(report_df: pd.DataFrame, allele_panel: str = NETMHCPAN_REFERENCE_PANEL) -> None:
    """Imprime el informe final de promiscuidad T-citotoxica (MHC-I).

    Analogo a ``netmhciipan_engine.print_th_report``: solo lista candidatos
    validos, resumen final usa el total evaluado como denominador.
    """
    if report_df.empty:
        print("No hay peptidos candidatos para evaluar contra el panel HLA-A/B/C.")
        return

    valid_df = report_df[report_df["veredicto"] == "Candidato Valido"]
    n_alleles = len([a for a in allele_panel.split(",") if a])

    if valid_df.empty:
        print("Ningun peptido/ventana supero el umbral de promiscuidad T-citotoxica (ver Resumen).")
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
    print(f"\nResumen T-citotoxico (MHC-I): {n_ok}/{len(report_df)} candidato(s) promiscuo(s) aprobado(s).")


def _deduplicate_protein_mode_windows(traceback_df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa ventanas redundantes del modo proteina (ver docstring analogo en netmhciipan_engine)."""
    if traceback_df.empty:
        return traceback_df

    best_idx = traceback_df.groupby(
        ["accession", "core_9aa", "n_alelos_promiscuos"], sort=False
    )["min_rank_el"].idxmin()
    return traceback_df.loc[best_idx].sort_index().reset_index(drop=True)


def build_traceback_report(report_df: pd.DataFrame, parent_df: pd.DataFrame) -> pd.DataFrame:
    """Cruza los 'Candidato Valido' de MHC-I con su region de origen en la Fase 3/4.

    Analogo exacto a ``netmhciipan_engine.build_traceback_report`` (ver su
    docstring para la logica completa de traceback por subcadena literal).

    Args:
        report_df: Salida de ``predict_netmhcpan``.
        parent_df: Tabla de la Fase 3/4 (``union_df`` o el ``safe_df`` de
            Fase 4), con ``accession``, ``start``, ``sequence``, ``origen`` y
            columnas ``'{motor}_score'``.

    Returns:
        DataFrame con columnas ``_TRACEBACK_BASE_COLUMNS`` mas
        ``'{motor}_score'`` por cada motor presente en ``parent_df``.
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
                "No se pudo trazar el candidato MHC-I '%s' de vuelta a ninguna region de la "
                "Fase 3/4; se omite del reporte final enriquecido.",
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
                "promiscuous_alleles": candidate.promiscuous_alleles,
                "n_alelos_evaluados": candidate.n_alelos_evaluados,
                "min_rank_el": candidate.min_rank_el,
            }
            for score_col in score_columns:
                record[score_col] = getattr(parent, score_col)
            records.append(record)

    traceback_df = pd.DataFrame.from_records(records, columns=columns)
    return _deduplicate_protein_mode_windows(traceback_df)
