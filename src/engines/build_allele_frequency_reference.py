"""SETUP step (to be run ONCE ONLY, not at runtime): it generates
``reference_db/allele_frequencies/world_pooled_afnd.csv`` from the
Allele Frequency Net Database (AFND, allelefrequencies.net), via the
MIT-licensed tab-delimited mirror ``github.com/slowkow/allelefrequencies``
(``afnd.tsv``, which requires no live scraping of the AFND site).

The same pattern as ``reference_db/iedb/bcell_protective_epitopes.csv`` (see
``src/engines/iedb_engine.py``, the "SETUP step" section of the module's
docstring): ``reference_db/`` is gitignored (binary/regenerable, not source
code), so this script reconstructs the small, curated CSV that the
pipeline does use (``src.engines.population_coverage``) from the raw
public source.

Usage:
    curl -sL -o /tmp/afnd.tsv https://raw.githubusercontent.com/slowkow/allelefrequencies/master/afnd.tsv
    python -m src.engines.build_allele_frequency_reference /tmp/afnd.tsv

Methodology (see the full docstring of ``population_coverage.py`` for the
reason behind each simplification):
* Frequency per allele = the sample-size-weighted average (``n``)
  across ALL the AFND populations/studies that report that allele, a
  pooled worldwide average, not stratified by region/ethnicity.
* The panel's DQ/DP alleles (alpha-beta combos, e.g. 'HLA-DQA10501-DQB10201'):
  AFND reports gene frequencies separately (DQA1, DQB1), not those of the
  joint heterodimer, so it is approximated as the PRODUCT of both gene
  frequencies (an independence assumption, not corrected for the real
  linkage disequilibrium between DQA1/DQB1).
* DRB3_0101/DRB3_0202/DRB4_0101/DRB5_0101: no data in this dataset (secondary
  DR genes, linked to specific DRB1 haplotypes, reported with
  far less consistency between studies), so they are left with an empty ``frequency``
  and EXCLUDED from the coverage calculation rather than assuming frequency 0.
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
