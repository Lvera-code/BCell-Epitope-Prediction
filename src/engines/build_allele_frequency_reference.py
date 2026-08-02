"""Paso de SETUP (correr UNA SOLA VEZ, no en runtime): genera
``reference_db/allele_frequencies/world_pooled_afnd.csv`` a partir de la
Allele Frequency Net Database (AFND, allelefrequencies.net), via el mirror
tab-delimited MIT-licenciado ``github.com/slowkow/allelefrequencies``
(``afnd.tsv``, no requiere scraping en vivo del sitio de AFND).

Mismo patron que ``reference_db/iedb/bcell_protective_epitopes.csv`` (ver
``src/engines/iedb_engine.py``, seccion "paso de SETUP" del docstring del
modulo): ``reference_db/`` esta gitignored (binario/regenerable, no codigo
fuente), asi que este script reconstruye el CSV pequeno y curado que SI usa
el pipeline (``src.engines.population_coverage``), a partir de la fuente
publica cruda.

Uso:
    curl -sL -o /tmp/afnd.tsv https://raw.githubusercontent.com/slowkow/allelefrequencies/master/afnd.tsv
    python -m src.engines.build_allele_frequency_reference /tmp/afnd.tsv

Metodologia (ver docstring completo de ``population_coverage.py`` para el
porque de cada simplificacion):
* Frecuencia por alelo = promedio ponderado por tamano de muestra (``n``)
  entre TODAS las poblaciones/estudios de AFND que reportan ese alelo --
  promedio mundial agrupado, no estratificado por region/etnia.
* Alelos DQ/DP del panel (combos alfa-beta, ej. 'HLA-DQA10501-DQB10201'):
  AFND reporta frecuencias de gen por separado (DQA1, DQB1), no del
  heterodimero conjunto -- se aproxima como el PRODUCTO de ambas frecuencias
  de gen (asuncion de independencia, no corregida por desequilibrio de
  ligamiento real entre DQA1/DQB1).
* DRB3_0101/DRB3_0202/DRB4_0101/DRB5_0101: sin datos en este dataset (genes
  DR secundarios, ligados a haplotipos DRB1 especificos, reportados con
  mucha menos consistencia entre estudios) -- quedan con ``frequency`` vacio,
  EXCLUIDOS del calculo de cobertura en vez de asumir frecuencia 0.
"""

import re
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUT_PATH = _REPO_ROOT / "reference_db" / "allele_frequencies" / "world_pooled_afnd.csv"

# Paneles actuales (ver netmhcpan_engine.NETMHCPAN_REFERENCE_PANEL /
# netmhciipan_engine.IEDB_REFERENCE_PANEL) -- si el panel cambia, correr este
# script de nuevo para regenerar la referencia con los alelos nuevos.
_MHC1_PANEL = (
    "HLA-A01:01,HLA-A02:01,HLA-A03:01,HLA-A24:02,HLA-A26:01,"
    "HLA-B07:02,HLA-B08:01,HLA-B27:05,HLA-B39:01,HLA-B40:01,HLA-B58:01,HLA-B15:01,"
    "HLA-C01:02,HLA-C03:04,HLA-C04:01,HLA-C05:01,HLA-C06:02,HLA-C07:01,"
    "HLA-C07:02,HLA-C08:02,HLA-C12:03,HLA-C15:02,HLA-C16:01"
).split(",")
_MHC2_PANEL = (
    "DRB1_0101,DRB1_0301,DRB1_0401,DRB1_0405,DRB1_0701,DRB1_0802,DRB1_0901,"
    "DRB1_1101,DRB1_1201,DRB1_1302,DRB1_1501,DRB3_0101,DRB3_0202,DRB4_0101,DRB5_0101,"
    "HLA-DQA10501-DQB10201,HLA-DQA10501-DQB10301,HLA-DQA10301-DQB10302,"
    "HLA-DQA10401-DQB10402,HLA-DQA10101-DQB10501,HLA-DQA10102-DQB10602,"
    "HLA-DPA10201-DPB10101,HLA-DPA10103-DPB10201,HLA-DPA10103-DPB10401,"
    "HLA-DPA10301-DPB10402,HLA-DPA10201-DPB10501,HLA-DPA10201-DPB11401"
).split(",")


def _weighted_freq(afnd: pd.DataFrame, gene: str, allele_code: str):
    subset = afnd[(afnd["gene"] == gene) & (afnd["allele"] == allele_code)]
    if subset.empty:
        return None, 0, 0
    total_n = subset["n"].sum()
    weighted = (subset["alleles_over_2n"] * subset["n"]).sum() / total_n
    return float(weighted), int(total_n), len(subset)


def build_reference(afnd_tsv_path: str) -> pd.DataFrame:
    """Construye el CSV de referencia a partir de ``afnd.tsv`` (mirror de AFND)."""
    afnd = pd.read_csv(afnd_tsv_path, sep="\t")
    afnd = afnd[afnd["group"] == "hla"].copy()
    afnd["n"] = pd.to_numeric(afnd["n"], errors="coerce")
    afnd["alleles_over_2n"] = pd.to_numeric(afnd["alleles_over_2n"], errors="coerce")
    afnd = afnd.dropna(subset=["n", "alleles_over_2n"])
    afnd = afnd[afnd["n"] > 0]

    rows = []
    for panel_allele in _MHC1_PANEL:
        gene, digits = re.match(r"HLA-([ABC])(\d{2}:\d{2})", panel_allele).groups()
        freq, n_total, n_studies = _weighted_freq(afnd, gene, f"{gene}*{digits}")
        rows.append({"panel_allele": panel_allele, "locus": gene, "frequency": freq,
                     "n_total": n_total, "n_studies": n_studies, "method": "afnd_weighted_pooled"})

    for panel_allele in _MHC2_PANEL:
        drb_match = re.match(r"(DRB[1345])_(\d{2})(\d{2})", panel_allele)
        if drb_match:
            gene, d1, d2 = drb_match.groups()
            freq, n_total, n_studies = _weighted_freq(afnd, gene, f"{gene}*{d1}:{d2}")
            method = "afnd_weighted_pooled" if freq is not None else "unavailable_no_afnd_data"
            rows.append({"panel_allele": panel_allele, "locus": "DR", "frequency": freq,
                         "n_total": n_total, "n_studies": n_studies, "method": method})
            continue
        agene, ad1, ad2, bgene, bd1, bd2 = re.match(
            r"HLA-(DQA1|DPA1)(\d{2})(\d{2})-(DQB1|DPB1)(\d{2})(\d{2})", panel_allele
        ).groups()
        a_freq, a_n, a_studies = _weighted_freq(afnd, agene, f"{agene}*{ad1}:{ad2}")
        b_freq, b_n, b_studies = _weighted_freq(afnd, bgene, f"{bgene}*{bd1}:{bd2}")
        joint_freq = (a_freq * b_freq) if (a_freq is not None and b_freq is not None) else None
        locus = "DQ" if agene == "DQA1" else "DP"
        rows.append({"panel_allele": panel_allele, "locus": locus, "frequency": joint_freq,
                     "n_total": min(a_n, b_n), "n_studies": min(a_studies, b_studies),
                     "method": "afnd_weighted_pooled_product_independence"})

    out = pd.DataFrame(rows)
    out["frequency"] = out["frequency"].round(6)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Uso: python -m src.engines.build_allele_frequency_reference <ruta_a_afnd.tsv>", file=sys.stderr)
        sys.exit(1)

    reference = build_reference(sys.argv[1])
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(_OUTPUT_PATH, index=False)
    n_missing = reference["frequency"].isna().sum()
    print(f"Referencia guardada en: {_OUTPUT_PATH} ({len(reference)} alelos, {n_missing} sin dato disponible).")
