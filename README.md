# Vaccine Epitope Discovery Pipeline

Terminal orchestrator (`pipeline.py`) that processes a sequence FASTA or a
structure (PDB/mmCIF) through 15 phases: from antigenicity (annotated
union of up to 4 independent engines: 2 sequence-based + 2
structure-based), masking of transmembrane/signal-peptide/
intracellular regions (not accessible to antibodies), absence of homology with the human proteome
(autoimmunity risk), allergenicity, N-glycosylation, T-helper
immunogenicity (MHC-II/HLA-DR/DQ/DP, CD4+) and cytotoxic T-cell
immunogenicity (MHC-I/HLA-A/B/C, CD8+, with proteasomal cleavage evidence)
and cross-referencing against known broadly neutralizing antibody (bnAb)
epitopes, through to the
**automatic assembly of a multi-epitope construct** with the best
candidates and an **allergenicity/toxicity/antigenicity/signal-peptide
check on that assembled construct** (not per individual peptide).
All phases run **100% locally** (subprocess over binaries/packages
installed on your machine): the pipeline never makes a network call during
inference. Each heavy phase (3b/4/4b/4c/5/5b/6/6b/7/8) is auto-cached by content
hash of its input. Restarting an interrupted run with the same
input/parameters jumps straight to the phase that failed, see "Checkpointing"
below.

## Three input paths

The file type passed to `--input` is detected automatically
(`src/utils/input_router.py`) and determines which Phase 2 engines run:

| Path | Input | Active engines | Notes |
|---|---|---|---|
| **1** | FASTA | BepiPred-3.0 + EpiDope | Original behaviour, unchanged |
| **2** | PDB/mmCIF, `--pdb-mode structure_only` | DiscoTope-3.0 + ScanNet | BepiPred/EpiDope are never invoked |
| **3** | PDB/mmCIF, `--pdb-mode structure_and_sequence` (default for structure input) | All 4 engines | A canonical FASTA (ATMSEQ) is derived from the structure for BepiPred/EpiDope |

For structure input, Phase 1 (FASTA sanitation) is replaced by
**Phase 1.5** (`src/utils/structure_parser.py`, via `gemmi`): it picks a
reference chain (`PDB_CHAIN_SELECTION_STRATEGY`, `'longest'` by default),
extracts its **ATMSEQ** sequence (the one actually resolved in the atoms, not
`SEQRES`) resolving modified residues via the CCD (MSE→M, SEP→S, etc.), and
builds a PDB↔FASTA position mapping. If that derived sequence contains
any non-canonical residue without a mapping (`X`), the pipeline **does not abort**:
it excludes only BepiPred-3.0/EpiDope from that run (with a clear warning) because BepiPred
rejects them wholesale, but the structural engines run on the
PDB regardless.

## Workflow (15 phases)

1. **FASTA sanitation / structure extraction.** Path 1: validates and
   cleans the input sequence(s) (`inputs/`, accepts multi-record
   FASTA). Paths 2/3: Phase 1.5, see above.
2. **Antigenicity.** Up to 4 independent engines executed locally
   according to the input path (see table above), each with its own
   auto-cache in CSV: BepiPred-3.0 and EpiDope (sequence engines, FASTA →
   per-residue score) and DiscoTope-3.0 and ScanNet (structural engines, single-chain
   PDB → per-residue score).
3. **Epitope mapping and annotated logical union.** For each active engine, a
   local sliding window (9 aa, tolerating up to 2 residues below
   the threshold per window, merging overlapping/adjacent windows;
   independent threshold per engine, the scales are not comparable with each other)
   over its antigenicity scores. Then, EVERY region detected by
   any active engine advances to Phase 4 (`src/engines/consensus.py`):
   regions that overlap between engines are **merged** (minimum `start`,
   maximum `end`, without trimming to the intersection, including transitive merging
   of chains of regions), and are marked in the `origen` column with
   2-letter abbreviations per contributing engine: `Bp` (BepiPred), `Ed`
   (EpiDope), `Dt` (DiscoTope-3.0), `Sn` (ScanNet). A single engine is
   reported on its own (e.g. `'Bp'`); two or three are joined with `'+'` (e.g.
   `'Bp+Ed'`, `'Dt+Sn'`, `'Bp+Dt'`, any combination, not just the
   "natural" pairs, so that all of them can be distinguished unambiguously).
   The only exception: when all 4 engines contribute at once, the label
   is `'Consenso total'` instead of `'Bp+Ed+Dt+Sn'`. Unbreakable length
   filter: any final region shorter than 9 aa is discarded
   before Phase 4.

   *Biological note:* DiscoTope-3.0/ScanNet score conformational epitopes
   (3D patches potentially discontinuous in the linear sequence);
   collapsing them into contiguous regions via a sliding window is a
   deliberate simplification so that Phases 4/5 keep operating on
   synthesizable linear peptides.
3b. **Transmembrane/signal-peptide/intracellular masking.** TMbed executed
   locally (`src/engines/tmbed_engine.py`), on the COMPLETE sequence of
   each accession (not per candidate peptide, unlike Phases 4b/4c: TMbed
   needs the full protein context to predict membrane
   topology). It discards from the Phase 3 annotated union any region that
   falls inside a transmembrane helix/strand, the N-terminal
   signal peptide or an intracellular (cytoplasmic) stretch. Those residues are
   not accessible to antibodies in the mature/membrane-anchored protein, so
   proposing them as a B-cell epitope makes no biological sense. Complete
   per-residue topology comes directly from TMbed (`--out-format 1`, classes
   `i`/`o` for inside/outside the membrane in addition to `B`/`H`/`S`), without adding
   any separate subcellular localization tool: an accession with no
   TM and no signal peptide but entirely cytoplasmic (e.g. PSMD7,
   a 26S proteasome subunit) ends up covered end to end by a
   single `intracellular` region and loses all its candidates, a
   complete exclusion by the same overlap mechanism, without a separate flag at
   the protein level. It reuses the same venv/weights already installed for the
   Scipion plugin `scipion-chem-tmbed` (sibling repo, same ProtT5-XL-U50 encoder
   as StackGlyEmbed), without importing code from that plugin (it depends on
   `pwchem`). Its own report: `<nombre>_tmbed_regions.csv` (detected
   regions) and `<nombre>_union_epitopes_masked.csv` (post-masking
   union, the real input of Phase 4).
4. **Tolerance filter.** Local BLASTp against the human proteome, with an
   E-value selected dynamically by peptide length (lax for
   short peptides, strict for complete domains/proteins), discarding
   peptides with high homology (`status = 'Autoinmunidad'`, threshold
   `BLAST_IDENTITY_THRESHOLD`, 75% by default). The resulting `'Segura'`
   peptides feed, in parallel and without depending on each other, **all** the
   following phases (4b, 4c, 5, 5b, 6). None of them discards candidates
   from another.
4b. **Allergenicity.** AlgPred 2.0 executed locally
   (`src/engines/algpred_engine.py`) on each `'Segura'` peptide.
   Purely informative: a safety signal of the sequence itself
   (potential IgE-mediated type I reaction), it does not condition any other
   phase. Its own report: `<nombre>_alergenicidad_report.csv`.
4c. **N-glycosylation.** StackGlyEmbed executed locally
   (`src/engines/stackglyembed_engine.py`), an already trained ProteinBERT + ESM-2 650M +
   ProtT5 stack. The original repo does not ship a scanner of candidate
   sites: this pipeline internally scans each `'Segura'` peptide
   looking for the canonical N-X-[S/T] sequon (X ≠ Proline, including overlapping
   sequons) and evaluates each site found; peptides without any sequon are
   skipped (they produce no row). Just as informative as 4b. Its own report:
   `<nombre>_glicosilacion_report.csv`.
5. **T-helper promiscuity (MHC-II).** NetMHCIIpan-4.3 executed locally
   against a panel of 27 IEDB reference HLA-DR/DQ/DP alleles
   (`IEDB_REFERENCE_PANEL`). Each peptide is routed according to its length:
   `<= 40 aa` in exact peptide mode (one output row per peptide);
   `> 40 aa` in protein mode (NetMHCIIpan internally slides a 15 aa
   window and evaluates all candidate binding cores inside the
   fragment; exact peptide mode blows up with a *buffer overflow* of the
   binary for inputs > 55 aa with the panel of 27 alleles that this
   pipeline uses, hence the safety margin of 40).

   **Reliability for synthesis/experimental validation.** NetMHCIIpan marks
   an allele as `Inverted=1` when its binding-core alignment procedure
   fits better reading the peptide in reverse than in its real
   direction. There is no structural evidence that MHC-II presents
   peptides "backwards"; it is treated as an artifact/limitation of the
   aligner, of lower confidence than a fit in normal orientation. This
   pipeline **discards inverted alleles completely** before
   computing anything: they do not count towards promiscuity nor can they be the
   "winning" allele that determines the reported 9 aa core. A peptide is
   approved (`'Candidato Valido'`) only if it classifies as a strong binder
   or weak binder (SB/WB, NetMHCIIpan default %Rank thresholds) in at
   least `NETMHCIIPAN_MIN_PROMISCUOUS_ALLELES` (3 by default) alleles
   **in normal orientation** of the panel.

   **Deduplication of protein-mode windows.** When sliding the 15 aa
   window one residue at a time, the same 9 aa core usually "wins" in
   several consecutive windows. They are not distinct epitopes, they are the same
   prediction seen from neighbouring offsets. Windows are grouped by
   **exact** `(accession, 9 aa core, promiscuity)`: within each
   group with more than one row only the one with the best (lowest)
   %Rank is kept. If the core differs by even 1 amino acid, or if
   promiscuity differs (with the same core), the rows are **not**
   merged: they are distinct predictions and both are reported, so as not to lose
   any variant with real binding support.
5b. **Cytotoxic T-cell promiscuity (MHC-I).** NetMHCpan-4.2 executed locally
   (`src/engines/netmhcpan_engine.py`), parallel to and independent of Phase
   5 (MHC-II): they are biologically distinct antigen presentation pathways
   (professional presenting cell vs. any nucleated cell, cytotoxic CD8+
   vs. helper CD4+), they are never merged into a single verdict.
   Same pattern as Phase 5 (reference panel, `NETMHCPAN_REFERENCE_PANEL`,
   23 HLA-A/B/C alleles: 12 HLA-A/B from the supertypes of Sidney et al. 2008
   (that paper does not cover HLA-C) + 11 globally common HLA-C (Rasmussen et
   al. 2014 + IEDB population frequency criterion ≥1%, see the docstring
   of `netmhcpan_engine.py` for the full detail)), routing
   by length, known buffer overflow of the binary (re-verified with the
   extended panel: the limit does not change), with its own
   %Rank thresholds (0.5/2.0, different from those of MHC-II: they are non-comparable
   scales). **Additional annotation of C-terminal proteasomal cleavage**: each
   accepted candidate is cross-referenced with NetCleave (`src/engines/netcleave_engine.py`)
   to check whether there is evidence of an EXACT cleavage at the residue
   immediately after the binding core. A peptide can bind
   MHC-I strongly and still never be generated via real antigen processing
   if the proteasome does not cleave there. It is an informative column
   (`netcleave_c_term_match`/`netcleave_c_term_score`), not a filter: the
   NetMHCpan verdict remains the only acceptance criterion.
   Its own report: `<nombre>_candidatos_finales_mhc1.csv`.
6. **Cross-reference with known bnAbs.** LANL HIV Molecular Immunology Database +
   CATNAP, executed locally on already downloaded CSVs
   (`src/engines/lanl_catnap_engine.py`), without any subprocess (pure
   pandas). It replaces bNAber, whose domain is dead/parked. It cross-references each
   `'Segura'` peptide against the known linear epitopes of broadly
   neutralizing antibodies (substring matching, configurable minimum
   threshold, `LANL_CATNAP_MIN_OVERLAP`, 6 aa by default) and appends
   neutralization potency (IC50, number of viruses in the panel) from CATNAP
   when the antibody name matches. Purely informative, it does not
   filter anything. It only produces real matches for inputs from the HIV
   Env family. An empty report is the expected result for any other
   protein, not a failure. It does not align to HXB2 coordinates nor capture
   conformational epitopes (out of scope for a sequence-based cross-reference).
   Its own report: `<nombre>_bnab_crossref.csv`.
6b. **Conservation against a reference panel (OPTIONAL).**
   (`src/engines/conservation_engine.py`). It only runs if
   `--panel-conservacion <fasta>` is passed (an unindexed multi-FASTA with other
   strains/clades/variants of the same pathogen; it is indexed once with
   `makeblastdb` and cached locally by content hash). Unlike
   Phase 6, if the flag is not passed this phase is **skipped
   completely** (it does not even invoke `blastp`/`makeblastdb`). It measures
   **breadth**, not best-hit: how many DISTINCT sequences of the panel
   match each candidate (`≥ CONSERVATION_IDENTITY_THRESHOLD`, 90% by
   default, more lax than the 75% of Phase 4 because here variants of the
   SAME pathogen are compared, not homology with the human proteome).
   A candidate identical to a single rare strain is not "conserved" in the
   sense that matters (protecting against many circulating variants).
   It reuses the Phase 4 BLASTp execution (same dynamic selection of
   `-task`/E-value by length) without duplicating it. Purely informative,
   like Phase 6: it does not discard any candidate. Its own report:
   `<nombre>_conservacion_report.csv`.
6c. **Regions with documented protection/neutralization (IEDB).**
   (`src/engines/iedb_engine.py`). A generalization of Phase 6 (LANL/CATNAP,
   specific to HIV Env) to **any studied pathogen**: it cross-references each
   `'Segura'` peptide against a local subset of IEDB (B-cell bulk
   export, 3429 records / 561 distinct organisms, already filtered by
   `Response measured` in {neutralization, protection} and `Qualitative
   Measure` = Positive*) by the same substring overlap mechanism
   as Phase 6 (`IEDB_MIN_OVERLAP`, 6 aa by default), **without filtering by
   organism first**. The sequence match already works as an implicit
   filter. Unlike Phase 6b, it **always runs** (it does not require any
   flag): the filtered CSV already lives in `reference_db/iedb/`. It only applies to
   B-cell candidates (not HTL/CTL; IEDB here is specifically antibody
   assays, a different mechanism from MHC presentation). Purely
   informative. Its own report: `<nombre>_iedb_crossref.csv`.
7. **Automatic assembly of the multi-epitope construct.**
   (`src/engines/construct_assembly.py`, pure logic, without subprocess;
   Phase 6b has already run before and delivers the computed result).
   It selects the best `CONSTRUCT_TOP_N_PER_CLASS` candidates (3 by
   default) of each class, **B-cell** (`'Segura'` peptides that are also
   `Non-Allergen` in Phase 4b, ranked by the best available `{motor}_score`),
   **HTL** (`'Candidato Válido'` from Phase 5, collapsed by
   `core_9aa`) and **CTL** (`'Candidato Válido'` from Phase 5b, collapsed by
   `core_9aa`, prioritizing `netcleave_c_term_match == True`), and
   concatenates them with the standard linkers of the multi-epitope vaccine
   design field: `AAY` intra-CTL (proteasome cleavage site),
   `GPGPG` intra-HTL and inter-block (universal spacer, Livingston et
   al. 2002), `KK` intra-B-cell. **N-glycosylation is NO longer an exclusion
   criterion** (Phase 4c) in any class: `glycosylated` is annotated per candidate
   (antibodies against glycosylated regions have been described), without
   discarding. If `--panel-conservacion` was passed, `conservation_pct` is also
   annotated (Phase 6b). B-cell candidates are additionally annotated with
   `documented_region` (Phase 6c, IEDB). None of the three annotations
   changes the top-N selection yet.
   **Block order: B-cell → HTL → CTL** (there is no strong consensus in the
   literature on the optimal order; the linkers already guarantee
   correct release by antigen processing independently of position; it is
   anchored on B-cell because that was the project's original humoral focus). **Without
   adjuvant** by active decision (it requires biological/strategic criteria
   specific to the pathogen/host, out of scope for this pipeline); the
   optional `adjuvant_sequence` parameter of `assemble_construct` allows
   adding one later without redesigning anything (rigid linker `EAAAK`, Arai
   et al. 2001). The "epitope" inserted in the HTL/CTL blocks is
   `sequence_f5` (the complete evaluated window, with flanks; the flanking
   residues also contribute to recognition), not just `core_9aa`
   (which is still used to deduplicate overlapping candidates).
   Overlapping epitopes of the SAME class are already merged in Phase 3 (annotated
   union); between different classes they are deliberately NOT merged (it would break
   the semantics of the linkers). It generates `<nombre>_constructo.fasta` and
   `<nombre>_constructo_metadata.csv` (100% traceability: one row per
   segment (epitope or linker), with its position in the construct, source accession/
   position and the score that drove the selection).
8. **Check of the assembled construct.** 4 independent engines
   running on the COMPLETE sequence of the construct (not per individual
   peptide, unlike Phases 4b/4c):
   - **Allergenicity**: AlgPred 2.0, reused as-is from Phase 4b
     (no new installation).
   - **Toxicity**: ToxinPred2 (`src/engines/toxinpred_engine.py`). The
     Raghava group itself recommends it for full-length
     proteins (unlike ToxinPred3.0, intended for short peptides).
   - **Intrinsic antigenicity**: IApred (`src/engines/iapred_engine.py`),
     a replacement for VaxiJen (discarded: it is not open-source nor does it have
     a local standalone/API), the only published open-source/local predictor
     specifically for full-sequence antigenicity (Miles
     et al. 2025).
   - **Signal peptide**: SignalP-6.0 (`src/engines/signalp_engine.py`),
     confirms that the construct does NOT have a predicted N-terminal signal
     peptide (expected for a synthetic fusion construct).
   All 4 are informative, none filters or aborts the pipeline. Combined
   report: `<nombre>_constructo_chequeo.csv`.

All intermediate results and the final report are saved in
`outputs/`.

### Checkpointing

Phases 3b/4/4b/4c/5/5b/6/6b/6c/7/8 are auto-cached by content hash of their input
(the same mechanism Phase 2 already used for its raw scores): each one
saves a `<archivo>.inputhash` sidecar next to its final CSV. If you relaunch
`pipeline.py` with the **same** `--input` and the **same** parameters (identity
threshold, extra alleles, etc.), each already completed phase is detected and
skipped ("`[Fase X] Checkpoint detectado...`" in the console) instead of
recomputing everything from the beginning. This is useful if a long run is interrupted
halfway (e.g. by an OOM in a heavy phase such as 4c/StackGlyEmbed,
which loads 3 large models). Changing any parameter that affects the
result of a phase invalidates its checkpoint automatically (and, in cascade,
that of the phases that depend on it). The process's peak resident memory
is logged after the heaviest phases (`INFO` level), to
diagnose in which one an OOM occurred without depending on external tools.

## Installation

Follow these steps **in order** before the first run.

### 1. Python environment

```bash
pip install -r requirements.txt
```

### 2. Local BepiPred-3.0 (required for Phase 2)

Phase 2 runs BepiPred-3.0 entirely on your machine via `subprocess`,
without any network call.

**a) Mandatory manual download.** Because of academic licence restrictions,
DTU Health Tech does not allow redistributing the BepiPred-3.0 source code: it is
not included in this repository (see `.gitignore`). You must request it
yourself from:

[https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=bepipred&version=3.0&packageversion=3.0b&platform=src](https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=bepipred&version=3.0&packageversion=3.0b&platform=src)

**b) Place it at the project root** so that it ends up like this:

```
<raiz-del-proyecto>/
└── bepipred-3.0b.src/
    └── BepiPred3_src/
        ├── bepipred3_CLI.py
        ├── bp3/
        └── requirements.txt
```

If you prefer another location, define the environment variable `BEPIPRED_HOME`
pointing to the folder that contains `bepipred3_CLI.py`. The pipeline never
assumes a fixed path.

**c) Install its dependencies in a SEPARATE venv.** BepiPred-3.0 pins
old versions (`torch==1.12.0`, `numpy==1.20.2`) that can clash with
those of this project:

```bash
python -m venv .venv-bepipred
source .venv-bepipred/bin/activate
pip install -r bepipred-3.0b.src/BepiPred3_src/requirements.txt
deactivate
```

> **Note (confirmed on Python 3.10+):** the exact pins of BepiPred's
> `requirements.txt` (from 2022) have no precompiled wheels
> for Python 3.10+ and fail to compile `numpy==1.20.2` from source
> (`Broken toolchain: cannot link a simple C program`) if you do not have a
> C compiler installed. If this happens to you, install modern versions
> instead. The code of `bp3/bepipred3.py` works the same with them:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install fair-esm plotly pandas
> ```

Then point the pipeline at that interpreter:

```bash
export BEPIPRED_PYTHON_BIN=$(pwd)/.venv-bepipred/bin/python
```

(If you skip this step, the same interpreter that runs `pipeline.py` is used.)

> The first run will download the ESM-2 models (~2.5 GB) to the
> `torch hub` cache; it can take several minutes depending on your connection and hardware
> (CPU vs GPU). If the `bepipred3_CLI.py` script is not found at the
> configured path, the pipeline stops with a clear message indicating this
> same download link, instead of failing with an opaque traceback.

### 3. Local EpiDope (required for Phase 2/3, second antigenicity engine)

Unlike BepiPred-3.0 and NetMHCIIpan-4.3, **EpiDope is open source**
(MIT licence, [github.com/rnajena/EpiDope](https://github.com/rnajena/EpiDope),
an actively maintained fork, successor of github.com/flomock/EpiDope) and does not
require an academic request. It does, however, pin an old dependency stack
that is very sensitive to exact versions (Python 3.6, TensorFlow 1.13,
Keras 2.3, PyTorch 0.4, AllenNLP 0.7.2 for ELMo embeddings), incompatible with
the rest of the pipeline: it requires a dedicated conda environment, the same pattern as
`.venv-bepipred`. The model weights and the ELMo embeddings come
packaged in EpiDope's own repo. Inference is 100% local, without
any network call or external API.

**a) Create the dedicated conda environment** from the repo's official
`epidope.yml` (do not install the packages by hand: that dependency resolution is
fragile and version by version ends in an inconsistent environment):

```bash
git clone https://github.com/rnajena/EpiDope.git /tmp/EpiDope
conda env create -f /tmp/EpiDope/epidope.yml -p .conda-epidope
```

(The repo weighs several hundred MB because of the packaged ELMo embeddings; creating
the environment can take several minutes because of the number of packages
pinned exactly.)

**b) Configurable path, without hardcoding.** If you installed EpiDope in another
location, point the pipeline there with one of these environment variables (in
order of precedence):

```bash
export EPIDOPE_BIN=/ruta/al/ejecutable/epidope       # full conda bypass
# or
export EPIDOPE_CONDA_ENV=mi-entorno-epidope           # by environment name
# or
export EPIDOPE_CONDA_PREFIX=/ruta/a/.conda-epidope     # by path prefix (default)
```

If the environment is not installed, Phase 2 stops with a clear message
indicating exactly these commands, instead of failing with an opaque traceback.

### 4. NCBI BLAST+ and the human proteome database (required for Phase 4)

Phase 4 runs BLASTp locally, never in the cloud. You need:

**a) To install the `blastp` / `makeblastdb` binary** (NCBI BLAST+ package):

```bash
# conda (recommended)
conda install -c bioconda blast

# or Debian/Ubuntu
sudo apt install ncbi-blast+
```

**b) To download the human proteome and index it locally** in
`reference_db/`:

```bash
mkdir -p reference_db
# Download the Homo sapiens reference proteome (UniProt) as FASTA:
# https://www.uniprot.org/proteomes/UP000005640 -> "Download" -> FASTA (canonical)
# Save it as reference_db/human_proteome.fasta

makeblastdb -in reference_db/human_proteome.fasta \
            -dbtype prot \
            -out reference_db/human_proteome_db
```

**c) Configurable path, without hardcoding.** Phase 4 reads the database
path from the environment variable `BLAST_HUMAN_DB` (by default
`reference_db/human_proteome_db`, the layout of the previous step). If you use another
location:

```bash
export BLAST_HUMAN_DB=/ruta/a/tu/base_de_datos
```

If `blastp` is not on the `PATH` or the configured database does not exist,
Phase 4 stops with a clear message (just like Phase 2 with BepiPred), instead
of failing with an opaque traceback.

**d) Dynamic selection of algorithm and E-value.** Each candidate peptide is
routed automatically according to its length: `<= 30 aa` uses `-task blastp-short`
with a lax E-value (50, which prevents BLAST statistics from discarding identical
hits of short peptides as "not significant"); `31–100 aa` uses `-task
blastp` with E-value 0.1; `> 100 aa` uses `-task blastp` with E-value 0.05
(strict, avoiding noise from irrelevant homologies in long queries).
Configurable via `BLAST_SHORT_PEPTIDE_MAX_LEN`, `BLAST_MEDIUM_PEPTIDE_MAX_LEN`
and `BLAST_EVALUE_SHORT` / `BLAST_EVALUE_MEDIUM` / `BLAST_EVALUE_LONG`.

**e) Minimum alignment coverage filter.** The lax e-value of
`blastp-short` (intended not to lose real short homologues) has a
side effect: a tiny fragment (5-6 aa) 100% identical inside
a peptide of 9-30 aa is statistically expected *by pure chance* against
the complete human proteome (~11M residues), not a real homology.
Without filtering this, almost any short peptide was rejected for
"Autoinmunidad" regardless of its origin. A
hit only counts towards `max_pident` if its alignment covers at least
`BLAST_MIN_QUERY_COVERAGE` (90% by default) of the length of the queried
peptide.

### 5. Local NetMHCIIpan-4.3 (required for Phase 5)

A proprietary DTU Health Tech binary (MHC-II / HLA-DR prediction), it
requires an academic licence and manual download, the same pattern as BepiPred.

**a) Mandatory manual download.** Request it at
`https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/` (the
"Downloads" section, an academic account is required). There is no separate public
`data.tar.gz`: the `.tar.gz` package that DTU delivers already includes the complete
`data/` folder (pseudosequences, allele lists, network weights).

**b) Installation.**
```bash
tar -xvf netMHCIIpan-4.3.Linux.tar.gz
mv netMHCIIpan-4.3 /ruta/al/proyecto/netMHCIIpan-4.3
```
Edit the `NMHOME` line at the start of the `netMHCIIpan-4.3/netMHCIIpan` script
with the absolute installation path (a mandatory manual step according to DTU's own
instructions, it cannot be resolved by an environment variable):
```tcsh
setenv NMHOME /ruta/absoluta/al/proyecto/netMHCIIpan-4.3
```

**c) System dependency.** The `netMHCIIpan` script is a wrapper in
`tcsh` (not `bash`): install the interpreter if you do not have it (`apt-get install
tcsh` on Debian/Ubuntu).

**d) Allele panel.** Phase 5 never evaluates a single allele: it uses
`IEDB_REFERENCE_PANEL` (the 27 HLA-DR/DRB3/DRB4/DRB5/DQ/DP alleles most
representative of IEDB population reference, see
`src/engines/netmhciipan_engine.py`) to estimate broad population
coverage.

### 6. Local DiscoTope-3.0 (optional, structural engine 1/2, Paths 2/3)

Unlike BepiPred-3.0/NetMHCIIpan-4.3, **DiscoTope-3.0 is installable
directly via git+pip** (Creative Commons licence, without a separate academic
request): [github.com/Magnushhoie/DiscoTope-3.0](https://github.com/Magnushhoie/DiscoTope-3.0/).
It pins its own stack (`torch`, `torch_geometric`, `xgboost`, `biotite`)
incompatible with the rest of the pipeline: it requires a dedicated isolated environment,
the same pattern as `.venv-bepipred`.

> **Note (confirmed in this environment):** `biotite==1.6.*` (the exact pin of the
> official `requirements.txt`) has no wheel for Python < 3.11. If your
> default `python3` is 3.10 or lower, create the venv with a newer
> interpreter (`python3.12`, etc.). If your system does not have `python3.12-venv`
> installed and you do not want to use `sudo`, `pip install --user virtualenv` followed
> by `python3 -m virtualenv -p python3.12 .venv-discotope` works without
> administrator privileges.

```bash
git clone https://github.com/Magnushhoie/DiscoTope-3.0/
python3 -m venv .venv-discotope   # or virtualenv -p python3.12, see note above
.venv-discotope/bin/pip install -r DiscoTope-3.0/requirements.txt
.venv-discotope/bin/pip install ./DiscoTope-3.0

# The weights of its own XGBoost ensemble come in the repo, compressed:
cd DiscoTope-3.0 && python3 -c "import zipfile; zipfile.ZipFile('models.zip').extractall('.')" && cd ..
```

Environment variables (all optional, with a sensible default):
`DISCOTOPE_INSTALL_PATH` (default `DiscoTope-3.0/`), `DISCOTOPE_PYTHON_BIN`
(default `.venv-discotope/bin/python`), `DISCOTOPE_WEIGHTS_CACHE_DIR`
(persistent cache of the ESM-IF1 weights, downloaded automatically on the
first run, ~350 MB, outside the repo by default:
`~/.cache/bcell-epitope-pipeline/discotope-weights`, so as not to download them
again in every new environment).

### 7. Local ScanNet (optional, structural engine 2/2, Paths 2/3)

[github.com/jertubiana/ScanNet](https://github.com/jertubiana/ScanNet), without
external software beyond its own Python stack, but a **very
old** one: exactly Python 3.6.12, TensorFlow 1.14, Keras 2.2.5. The
pre-trained weights (~43 MB) already come in the repo, there is no need to download them
separately.

> **No modern system ships Python 3.6.12 preinstalled any more.** The
> reproducible way to get it (confirmed in this environment) is with **conda**,
> not with a plain venv (which needs to start from an already installed interpreter):

```bash
git clone https://github.com/jertubiana/ScanNet
conda create -n scannet_env python=3.6.12 -y
conda run -n scannet_env pip install -r ScanNet/requirements.txt
```

If you installed the conda environment with the name `scannet_env` (the one used above),
**there is no need to export anything**: `./run.sh` (see "Usage" below) detects that
environment automatically and uses the `venv` runtime instead of the default
`docker`. If you gave it another name, or you prefer to invoke `pipeline.py`
directly without going through `run.sh`:

```bash
export SCANNET_PYTHON_BIN=$HOME/miniconda3/envs/mi_entorno/bin/python
export SCANNET_RUNTIME=venv   # default is 'docker' (see below)
```

**Docker alternative** (`SCANNET_RUNTIME=docker`, official image
`jertubiana/scannet`): it avoids resolving the old stack by hand. **Validated**
(`docker pull jertubiana/scannet` + a real run against a PDB, results
byte-for-byte identical to the `venv` runtime). The image's `WORKDIR` is
effectively `/ScanNet` (the default of `SCANNET_DOCKER_WORKDIR`, confirmed
with `docker inspect`), and both `python` and `python3` resolve to the correct
interpreter (3.6.12) inside the image. It is the simplest alternative
for anyone who does not want to deal with conda:

```bash
docker pull jertubiana/scannet
export SCANNET_RUNTIME=docker   # already the default; explicit only if something else changed it
```

Environment variables: `SCANNET_RUNTIME` (`docker` default | `venv`),
`SCANNET_INSTALL_PATH` (default `ScanNet/`), `SCANNET_PYTHON_BIN`,
`SCANNET_DOCKER_IMAGE` (default `jertubiana/scannet`),
`SCANNET_DOCKER_WORKDIR`.

### 8. Local NetMHCpan-4.2 (required for Phase 5b)

A proprietary DTU Health Tech binary (MHC-I / HLA-A/B/C prediction), the same
academic licence and manual download pattern as NetMHCIIpan-4.3.

**a) Mandatory manual download.** Request it at
`https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/` (the
"Downloads" section, an academic account is required).

**b) Installation.**
```bash
tar -xvf netMHCpan-4.2.Linux.tar.gz
mv netMHCpan-4.2 /ruta/al/proyecto/netMHCpan-4.2
```
Edit the `NMHOME` line at the start of the `netMHCpan-4.2/netMHCpan` script with the
absolute installation path, just as in NetMHCIIpan-4.3 (Section 5b).

**c) Known buffer overflow.** The `netMHCpan` binary (exact peptide mode,
`-p`) blows up with `*** buffer overflow detected ***` (exit code 0, the
`tcsh` wrapper does not propagate the failure) for peptides > ~55 aa with the panel of 23
default alleles (re-verified when extending the panel from 12 to 23: the limit
does not change, it is a property of the peptide length, not of how many alleles are
passed via `-a`). The pipeline automatically routes long peptides to
protein mode (sliding window) to avoid it, the same mechanism as
NetMHCIIpan-4.3.

Environment variables: `NETMHCPAN_HOME` (default `netMHCpan-4.2/`),
`NETMHCPAN_BINARY_NAME` (default `netMHCpan`).

### 9. Local AlgPred 2.0 (required for Phase 4b)

Open source (GPSR group, without an academic request), installable via `pip`.
It requires a separate venv: it pins its own dependency stack.

```bash
python3 -m venv .venv-algpred
.venv-algpred/bin/pip install algpred2
```

Environment variables: `ALGPRED_PYTHON_BIN` (the venv's interpreter),
`ALGPRED_SCRIPT_PATH` (path to `algpred2.py` inside the installed package,
normally `<venv>/lib/python3.X/site-packages/algpred2/python_scripts/algpred2.py`),
`ALGPRED_THRESHOLD` (`ML_Score` threshold, 0.3 by default, the same as
`algpred2.py` itself).

> **Known upstream script bug** (verified empirically): it blows up
> with `ValueError: Expected 2D array, got 1D array` if the input batch
> has EXACTLY 1 sequence. This pipeline's wrapper avoids it by
> duplicating the single sequence before invoking the binary and discarding the
> extra row from the result. Transparent, it requires no action.

### 10. Local NetCleave (required for the Phase 5b annotation)

Open source (MIT, [github.com/APeriolo/NetCleave](https://github.com/APeriolo/NetCleave)),
prediction of proteasomal cleavage sites (MHC-I). This pipeline uses
only the bundled pre-trained model (`data/models/I_mass-spectrometry_HLA/`,
trained on IEDB/UniProt/UniParc). It **never retrains at runtime**.

```bash
git clone https://github.com/APeriolo/NetCleave.git
python3 -m venv .venv-netcleave
.venv-netcleave/bin/pip install -r NetCleave/requirements.txt
.venv-netcleave/bin/pip install openpyxl  # pandas does not ship .xlsx support by default
```

Environment variables: `NETCLEAVE_PYTHON_BIN`, `NETCLEAVE_SCRIPT_PATH` (path to
`NetCleave.py`).

### 11. Local StackGlyEmbed (required for Phase 4c)

Open source ([github.com/GaryChan-lab/StackGlyEmbed](https://github.com/GaryChan-lab/StackGlyEmbed)),
N-glycosylation prediction via an already trained ProteinBERT + ESM-2 650M + ProtT5
stack. The original repo expects candidate positions written by hand
and calls the network on every run (ESM-2 via `torch.hub`, ProtT5 against the
remote Hub ID). This pipeline replaces it with
`src/engines/stackglyembed_predict_local.py` (versioned in this same repo,
not inside the external clone: see the module's docstring), 100% offline once
the weights are cached.

**a) Clone the repo and install its venv:**
```bash
git clone https://github.com/GaryChan-lab/StackGlyEmbed.git
python3 -m venv StackGlyEmbed/.venv-stackglyembed
StackGlyEmbed/.venv-stackglyembed/bin/pip install numpy pandas "tensorflow==2.14.*" \
    "tensorflow_addons==0.22.0" torch xgboost scikit-learn "transformers<5" h5py lxml pyfaidx
```
> `transformers>=5` requires `torch>=2.4`: if your venv pins `torch==2.2.2` (like
> the reference one), install a `4.x` version of `transformers` or the
> PyTorch backend is silently disabled (without breaking the import).

**b) Install ProteinBERT** (it is not on PyPI under that name):
```bash
StackGlyEmbed/.venv-stackglyembed/bin/pip install --no-deps \
    "git+https://github.com/nadavbra/protein_bert.git"
```

**c) Download the weights once only** (a SETUP step, never at runtime):
```bash
StackGlyEmbed/.venv-stackglyembed/bin/python -c "
from proteinbert import load_pretrained_model
load_pretrained_model(validate_downloading=False)"  # ~183MB to ~/proteinbert_models/

HF_HUB_OFFLINE=0 StackGlyEmbed/.venv-stackglyembed/bin/python -c "
from transformers import AutoTokenizer, EsmModel
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"  # ~2.5GB to the HF Hub cache
```

**d) ProtT5**: if you already have the weights of `Rostlab/prot_t5_xl_half_uniref50-enc`
downloaded for another tool (e.g. TMbed), reuse them by pointing
`STACKGLYEMBED_T5_MODEL_PATH` at that local folder. It avoids a duplicated
~3GB download (the same encoder, only the precision/packaging changes with respect to the
`Rostlab/prot_t5_xl_uniref50` that the original repo asks for).

Environment variables: `STACKGLYEMBED_PYTHON_BIN`, `STACKGLYEMBED_SCRIPT_PATH`
(default: `src/engines/stackglyembed_predict_local.py` of this repo, it does not
usually need touching), `STACKGLYEMBED_MODELS_DIR` (the `prediction/` folder
of the external clone, where the pickles of the already trained classifier live),
`STACKGLYEMBED_T5_MODEL_PATH`, `STACKGLYEMBED_ESM_MODEL_NAME` (default
`facebook/esm2_t33_650M_UR50D`).

### 12. LANL + CATNAP data (required for Phase 6)

Without any software installation: they are flat CSVs/TSVs, queried with pure
pandas, without any subprocess or network call at runtime.

```bash
mkdir -p reference_db/lanl_immunology reference_db/catnap
# LANL HIV Molecular Immunology Database, "Antibody" table:
# https://www.hiv.lanl.gov/components/sequence/HIV/asearch/query_one.comp?se_id=ab
# -> export as reference_db/lanl_immunology/ab_all.csv

# CATNAP (neutralization, antibodies):
# https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp
# -> export as reference_db/catnap/abs_<fecha>.txt
```

Environment variables: `LANL_AB_ALL_PATH`, `CATNAP_ABS_PATH`,
`LANL_CATNAP_MIN_OVERLAP` (minimum substring overlap threshold, 6 aa
by default; for reference epitopes shorter than the threshold, a full
match is required, never one more lax than the epitope itself).

### 13. ToxinPred2 (required for Phase 8, construct toxicity)

Open source (Raghava group), installable via `pip`. The model (Random
Forest, ONNX) and the `blastp` binary it uses internally come EMBEDDED in
the package, zero separate downloads.

```bash
python3 -m venv .venv-toxinpred2
.venv-toxinpred2/bin/pip install toxinpred2
```

> **Note (confirmed in this environment):** create the venv with **Python 3.10**,
> not with the system default `python3` if it is 3.13+. The
> packaged script writes an intermediate file with
> `to_csv(..., sep="\n")`, which `pandas>=2` rejects (`ValueError: bad
> delimiter value`), and there is no CLI flag that avoids that step. Mutually
> compatible versions have to be pinned:
> ```bash
> .venv-toxinpred2/bin/pip install "pandas==1.5.3" "numpy<2"
> ```
> (pandas 1.5.3 is compiled against the numpy<2 ABI; without this second
> pin, importing pandas blows up with
> `ValueError: numpy.dtype size changed...`).

Environment variables: `TOXINPRED2_PYTHON_BIN`, `TOXINPRED2_BINARY_NAME`
(default `toxinpred2`), `TOXINPRED2_THRESHOLD` (0.6 by default, the same
default as `toxinpred2` itself), `TOXINPRED2_MODEL` (1 by default).

> **Hybrid mode (`TOXINPRED2_MODEL=2`) and upstream bug.** Model 1 is
> a Random Forest over amino acid composition and does not compare against
> any real toxin; model 2 adds BLAST against the embedded toxin database
> and MERCI motifs, and exposes the three scores separately, just like
> `ALGPRED_MODEL=2`. Model 2 of `toxinpred2` 1.1 is **broken out of the box**:
> `BLAST_processor()` iterates over `seqid`, which only exists as a local
> variable of `main()`, so it blows up with `NameError: name 'seqid' is not
> defined` whenever BLAST returns no hit, which is the normal case for
> a short non-toxic peptide. The project fixes it at runtime
> with `src/engines/toxinpred2_hybrid_shim.py`, without patching the installed
> package (the same policy as with AlgPred2), so that a reinstall
> does not lose the fix. No extra installation step is needed.
>
> The default is still `1` **on purpose**: enabling 2 adds three
> columns to `*_constructo_chequeo.csv` and changes the meaning of
> `toxinpred_score` (it becomes `Hybrid Score`), so it forces a re-run of
> the validation panel.

### 14. IApred (required for Phase 8, intrinsic antigenicity of the construct)

Open source ([github.com/sebamiles/IApred](https://github.com/sebamiles/IApred),
Miles et al. 2025), a replacement for VaxiJen (discarded: it is not open-source nor
does it have a local standalone/API). A pure SVM over physicochemical features, without
PyTorch/TensorFlow.

```bash
git clone https://github.com/sebamiles/IApred.git
python3 -m venv IApred/.venv-iapred
IApred/.venv-iapred/bin/pip install -r IApred/requirements.txt
```

> **Note (confirmed in this environment):** the repo's `requirements.txt` is
> incomplete. `functions.py` imports, without declaring them anywhere,
> `imbalanced-learn`, `matplotlib` and `seaborn`. Install them by hand:
> ```bash
> IApred/.venv-iapred/bin/pip install imbalanced-learn matplotlib seaborn
> ```

Environment variables: `IAPRED_PYTHON_BIN`, `IAPRED_HOME` (the root of the clone;
`models_folder` inside the original script is a path RELATIVE to the cwd, so
the wrapper always invokes the subprocess with `cwd=IAPRED_HOME`),
`IAPRED_SCRIPT_NAME` (default `IApred.py`).

### 15. SignalP-6.0 (required for Phase 8, signal peptide of the construct)

A proprietary DTU Health Tech binary (academic licence, the same pattern as
BepiPred-3.0/NetMHCIIpan-4.3/NetMHCpan-4.2).

**a) Mandatory manual download.** Request it at
`https://services.healthtech.dtu.dk/services/SignalP-6.0/` (the
"Downloads" section, an academic account is required). Choose the **`slow_sequential`**
mode (the same RAM footprint as `fast`, ~6x slower, intended for machines without
a GPU; the `slow` mode in parallel requires >14GB free).

**b) Installation.**
```bash
tar -xvf signalp-6.0i.slow_sequential.tar.gz
python3 -m venv .venv-signalp   # Python 3.10, see note below
.venv-signalp/bin/pip install ./signalp-6-package
.venv-signalp/bin/pip install "numpy<2"   # ABI, see note below
```

> **Notes (confirmed in this environment):**
> - Create the venv with **Python 3.10**: the package's `requirements.txt` pins
>   `torch>1.7.0,<2`, with no wheel left in the official CPU-only PyTorch index
>   for modern installations. It is installed from normal PyPI instead,
>   which does have wheels for Python 3.10.
> - `pip install` drags in `numpy>=2` (via `matplotlib`, without an upper pin in
>   `requirements.txt`), incompatible with the ABI of `torch==1.13`, the same
>   kind of bug as ToxinPred2 (`RuntimeError: Numpy is not available`).
> - **There is no need to follow step 4 of the official `README.md`** (copying the
>   weights inside the installed package): point `SIGNALP_MODEL_DIR`
>   directly at the folder that already contains `sequential_models_signalp6/`
>   (the `--model_dir` flag of `signalp6` itself uses it as-is). It avoids
>   duplicating ~9.2GB.

Environment variables: `SIGNALP_PYTHON_BIN`, `SIGNALP_BINARY_NAME` (default
`signalp6`), `SIGNALP_MODEL_DIR`, `SIGNALP_ORGANISM` (`other` by default).

### 16. TMbed (required for Phase 3b, transmembrane/signal-peptide/intracellular masking)

Open source (Apache-2.0, Bernhofer & Rost 2022), pip-installable, but its
ProtT5-XL-U50 encoder weights (~2.4 GB) do not come bundled and are normally
downloaded from HuggingFace on first use, which is not allowed by this
project's local-only/no-scraping policy: both pieces are installed
once only, by hand.

This pipeline **reuses the same venv/weights** already installed for the
sibling Scipion plugin `scipion-chem-tmbed` (see its `README.rst`), without installing
anything new if that plugin is already configured.

**a) Dedicated venv.**
```bash
python3 -m venv /path/to/venv-tmbed
/path/to/venv-tmbed/bin/pip install tmbed
```

**b) ProtT5-XL-U50 weights**, downloaded once only (e.g. from a
machine with internet access, or via
`huggingface-cli download Rostlab/prot_t5_xl_uniref50`) into a local folder
with `config.json`, `model.safetensors`, `spiece.model`,
`special_tokens_map.json` and `tokenizer_config.json`. **The same encoder that
StackGlyEmbed reuses** (Section 11, `STACKGLYEMBED_T5_MODEL_PATH`): if you already
have those weights downloaded for StackGlyEmbed, point `TMBED_MODEL_DIR` at
the same folder instead of duplicating ~2.4 GB.

Environment variables: `TMBED_PYTHON_BIN`, `TMBED_BINARY_NAME` (default
`tmbed`), `TMBED_MODEL_DIR`, `TMBED_USE_GPU` (`0`/`1`, `0` by default, a
CPU-only machine), `TMBED_THREADS` (`4` by default), `TMBED_MIN_REGION_LENGTH`
(`1` by default, without a minimum length filter).

### 17. IEDB data (required for Phase 6c)

Without any software installation: it is a flat, already filtered CSV, queried with
pure pandas, without any subprocess or network call at runtime. **It already comes
included in the repo** (`reference_db/iedb/bcell_protective_epitopes.csv`,
420 KB, 3429 records). This step is only needed if you want to regenerate it
(e.g. IEDB publishes a more recent version of the bulk export).

```bash
mkdir -p reference_db/iedb
curl -o /tmp/bcell_full_v3_single_file.zip \
  "https://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3_single_file.zip"
python3 - <<'PY'
import zipfile
with zipfile.ZipFile("/tmp/bcell_full_v3_single_file.zip") as z:
    z.extractall("/tmp")
PY
```

Filter the raw CSV (`/tmp/bcell_full_v3.csv`, ~3.24 GB uncompressed, a 2-level
header `pd.read_csv(header=[0,1], encoding="latin-1")`) by
`('Epitope','Object Type') == 'Linear peptide'`,
`('Assay','Response measured')` in {`neutralization`, `protection from
pathogen/tumor/other challenge`(+ the `after adoptive transfer` variants),
`protection from fertility`} and `('Assay','Qualitative Measure')` starting
with `Positive`. See the full docstring of `src/engines/iedb_engine.py`
for the exact criterion and the columns to keep. The raw CSV is **never**
distributed with the repo, only the filtered subset.

Environment variables: `IEDB_BCELL_REFERENCE_PATH`, `IEDB_MIN_OVERLAP` (minimum
substring overlap threshold, 6 aa by default, the same criterion as
`LANL_CATNAP_MIN_OVERLAP`).

### 18. Allele frequencies for population coverage (optional, annotates Phase 5/5b)

Without any software installation: it is a flat CSV of 50 rows (one per allele of
`NETMHCPAN_REFERENCE_PANEL`/`IEDB_REFERENCE_PANEL`), queried with pure
pandas. `n_alelos_promiscuos` (Phase 5/5b) counts alleles equally without weighting
by how frequent each one is in a real population. `population_coverage_pct`
does weight them, via the same base methodology as the IEDB Population
Coverage tool (Bui et al. 2006, see the full docstring of
`src/engines/population_coverage.py`). Purely informative, it never changes
the `'Candidato Valido'` verdict.

```bash
mkdir -p reference_db/allele_frequencies
curl -sL -o /tmp/afnd.tsv \
  https://raw.githubusercontent.com/slowkow/allelefrequencies/master/afnd.tsv
python -m src.engines.build_allele_frequency_reference /tmp/afnd.tsv
```

Source: the Allele Frequency Net Database (allelefrequencies.net), via the
MIT-licensed tab-delimited mirror `github.com/slowkow/allelefrequencies` (without
live scraping of the AFND site). Each allele of the panel is resolved to a
sample-size-weighted average across ALL available populations/studies,
a pooled worldwide average, not stratified by region/ethnicity
(a deliberate simplification with respect to the full IEDB calculation, documented in
the CSV itself via its `method` column). 4 alleles of the MHC-II panel
(`DRB3_0101`/`DRB3_0202`/`DRB4_0101`/`DRB5_0101`, secondary DR genes without
consistent data in AFND) end up without a frequency; they are excluded from the
per-candidate calculation instead of assuming frequency 0. Re-run the script if
`NETMHCPAN_REFERENCE_PANEL`/`IEDB_REFERENCE_PANEL` change.

Environment variable: `ALLELE_FREQUENCY_PATH` (default:
`reference_db/allele_frequencies/world_pooled_afnd.csv`). Empty or a missing
file disables the annotation without breaking anything, the same pattern as
`--panel-conservacion`.

## Usage

`./run.sh` is a thin wrapper over `pipeline.py` (the same arguments) that
auto-detects the local installations of BepiPred-3.0 and ScanNet (venv/conda)
if they are in their default location, so that you do not have to export anything by
hand. DiscoTope-3.0 never needs exports: its defaults are already relative to
the project root. Prefer it over invoking `python pipeline.py` directly
unless you need to point at installations in other paths (there you do use the
environment variables of Sections 2-7).

```bash
# Place your FASTA file(s) in inputs/ (multi-record is supported), then:
./run.sh --input inputs/secuencia.fasta

# Structure input (Paths 2/3, requires DiscoTope-3.0/ScanNet installed,
# see Sections 6 and 7). The file type is detected automatically:
./run.sh --input inputs/estructura.pdb --pdb-mode structure_only
./run.sh --input inputs/estructura.pdb
# Without --pdb-mode, Settings.PDB_PROCESSING_MODE is used (default 'structure_and_sequence'),
# so the second line above already runs the 4 engines without needing the flag.

# The panel of 27 HLA-DR/DQ/DP alleles (IEDB_REFERENCE_PANEL) is always evaluated
# by default, without needing to specify anything. To append extra allele(s)
# (NetMHCIIpan format, comma-separated WITHOUT spaces; the format is validated
# immediately, before running any phase):
./run.sh --input inputs/secuencia.fasta \
    --alelo-extra "DRB1_1602,HLA-DQA10501-DQB10201"

# The Phase 3 thresholds/minimum length are independent per engine
# (the score scales are not comparable with each other):
./run.sh --input inputs/secuencia.fasta \
    --bepipred-threshold 0.1512 --bepipred-min-length 9 \
    --epidope-threshold 0.818 --epidope-min-length 9 \
    --discotope-threshold 0.90 --discotope-min-length 9 \
    --scannet-min-length 9
# ScanNet is different: by default (without --scannet-threshold) it uses an
# ADAPTIVE percentile threshold, computed per accession (ScanNet does not publish an
# official absolute threshold, unlike DiscoTope-3.0). To force a
# fixed value instead of the adaptive one:
./run.sh --input inputs/estructura.pdb --scannet-threshold 0.15

# Phase 6b (OPTIONAL): annotates conservation breadth against a local panel
# of other strains/clades/variants of the same pathogen (without this flag, the
# phase is skipped completely, without invoking blastp/makeblastdb):
./run.sh --input inputs/secuencia.fasta \
    --panel-conservacion reference_db/hiv_clade_panel.fasta
```

Run `python pipeline.py --help` (or `./run.sh --help`) to see all available flags
(BLAST threshold/E-value, output folder, etc.), all with their default
value documented.

### Console output

The Phase 5 (and Phase 5b) tables highlight in colour the 9 aa binding core
inside the evaluated 15 aa window (terminal only, it does not affect the
CSV). When the input FASTA has several proteins (several
`accession`s), the tables of Phase 2 (BepiPred-3.0/EpiDope/DiscoTope-3.0/
ScanNet), Phase 3b and Phase 6 separate each protein with a dividing line,
so as not to read them as one continuous list.

Phase 3b, in addition to the table of detected transmembrane/signal-peptide/
intracellular regions, also prints which regions of the annotated
union were discarded for overlapping with them
(`accession`/`start`/`end`/`tipo`), where applicable.

Phase 6 wraps the `Secuencia` column every 40 characters instead of
stretching the table across the width (a candidate HIV Env peptide can be 70+
aa). The stretch that matches the bnAb epitope is highlighted in yellow. The
antibody/domain names are shortened semantically (never a word
cut in half) and any raw HTML markup that the original LANL data carries is
cleaned up (e.g. `<sub>`).

Phase 8 (the construct check) only shows score/verdict/summary
per engine, without repeating the construct's complete sequence (already visible in
the Phase 7 table).

### Files generated in `outputs/`

`<nombre>` is the name of the input file without its extension (`--input`).

| File | Phase | Contents |
|---|---|---|
| `<nombre>_clean.fasta` | 1 | (Path 1) Sanitized FASTA sent to the following phases |
| `<nombre>_derived.fasta` | 1.5 | (Paths 2/3) Canonical ATMSEQ FASTA derived from the structure |
| `<nombre>_chain_<cadena>.pdb` | 1.5 | (Paths 2/3) Single-chain PDB (the chosen one), the real input of the structural engines |
| `<nombre>_position_mapping.csv` | 1.5 | (Paths 2/3) Derived PDB↔FASTA position mapping |
| `<nombre>_bepipred_raw.csv` | 2 | Raw per-residue scores from BepiPred-3.0 (cache) |
| `<nombre>_epidope_raw.csv` | 2 | Raw per-residue scores from EpiDope (cache) |
| `<nombre>_discotope_raw.csv` | 2 | Raw per-residue scores from DiscoTope-3.0 (cache) |
| `<nombre>_scannet_raw.csv` | 2 | Raw per-residue scores from ScanNet (cache) |
| `<nombre>_bepipred_epitopes.csv` | 3 | Epitope regions mapped locally (BepiPred-3.0) |
| `<nombre>_epidope_epitopes.csv` | 3 | Epitope regions mapped locally (EpiDope) |
| `<nombre>_discotope_epitopes.csv` | 3 | Epitope regions mapped locally (DiscoTope-3.0) |
| `<nombre>_scannet_epitopes.csv` | 3 | Epitope regions mapped locally (ScanNet) |
| `<nombre>_union_epitopes.csv` | 3 | Annotated union of the active engines (`origen` column), the input of Phase 3b |
| `<nombre>_tmbed_raw.pred` | 3b | Raw TMbed output (3-line-per-protein format), for traceability |
| `<nombre>_tmbed_regions.csv` | 3b | Detected transmembrane/signal-peptide/intracellular regions (`accession`/`start`/`end`/`type`) |
| `<nombre>_union_epitopes_masked.csv` | 3b | Annotated union after discarding regions overlapping TM/signal peptide/intracellular, the real input of Phase 4 |
| `<nombre>_blast_report.csv` | 4 | Tolerance verdict (Segura / Autoinmunidad) per region |
| `<nombre>_algpred_raw.csv` | 4b | Raw AlgPred 2.0 output, for traceability |
| `<nombre>_alergenicidad_report.csv` | 4b | Allergenicity verdict (Allergen / Non-Allergen) per `'Segura'` peptide |
| `<nombre>_stackglyembed_features.csv` | 4c | Raw StackGlyEmbed features (ProteinBERT+ESM-2+ProtT5), for traceability |
| `<nombre>_stackglyembed_raw.csv` | 4c | Raw StackGlyEmbed prediction (0/1 + probability), for traceability |
| `<nombre>_glicosilacion_report.csv` | 4c | N-glycosylation verdict (Glicosilado / No glicosilado) per candidate sequon |
| `<nombre>_netmhciipan_raw_peptide_mode.xls` | 5 | Raw NetMHCIIpan-4.3 output, exact peptide mode (`<= 40` aa), for traceability. Only generated if there was at least one peptide in that range |
| `<nombre>_netmhciipan_raw_protein_mode.xls` | 5 | Raw NetMHCIIpan-4.3 output, protein/sliding-window mode (`> 40` aa), for traceability. Only generated if there was at least one fragment in that range |
| `<nombre>_candidatos_finales.csv` | 5 | **Final MHC-II report** (see format below) |
| `<nombre>_netmhcpan_raw_peptide_mode.xls` / `_protein_mode.xls` | 5b | Raw NetMHCpan-4.2 output, same criterion as NetMHCIIpan |
| `<nombre>_netcleave_raw.csv` | 5b | Raw NetCleave output (all evaluated cleavage windows, not just those that match) |
| `<nombre>_candidatos_finales_mhc1.csv` | 5b | **Final MHC-I report**, with `netcleave_c_term_match`/`netcleave_c_term_score` annotation |
| `<nombre>_bnab_crossref.csv` | 6 | Cross-reference with known bnAb epitopes (empty if the input is not HIV Env, which is the expected result) |
| `<nombre>_conservacion_report.csv` | 6b | **Only if `--panel-conservacion` was passed**: conservation breadth (`block`/`sequence`/`n_panel_matches`/`n_panel_total`/`conservation_pct`) per B-cell/HTL/CTL candidate |
| `<nombre>_iedb_crossref.csv` | 6c | Cross-reference with IEDB regions of documented protection/neutralization, any pathogen (empty is a valid result, not a failure) |
| `<nombre>_constructo.fasta` | 7 | Sequence of the assembled multi-epitope construct |
| `<nombre>_constructo_metadata.csv` | 7 | 100% traceability: one row per segment (epitope or linker), position in the construct, source accession/position, score that drove the selection |
| `<nombre>_constructo_algpred_raw.csv` | 8 | Raw AlgPred2 output on the construct |
| `<nombre>_constructo_toxinpred_raw.csv` | 8 | Raw ToxinPred2 output on the construct |
| `<nombre>_constructo_iapred_raw.csv` | 8 | Raw IApred output on the construct |
| `<nombre>_constructo_signalp_raw.txt` | 8 | Raw SignalP-6.0 output on the construct |
| `<nombre>_constructo_chequeo.csv` | 8 | **Combined construct report**: allergenicity + toxicity + intrinsic antigenicity + signal peptide |
| `<archivo>.inputhash` | 3b/4/4b/4c/5/5b/6/6b/7/8 | Checkpointing sidecar (hash of that phase's input), see "Checkpointing" above |

### Format of `<nombre>_candidatos_finales.csv`

This is the output of Phase 5 after the cross-reference with Phase 3/4 (traceback) and the
deduplication of redundant windows: it only contains candidates with
`veredicto == 'Candidato Valido'`, already enriched with their region of origin.

| Column | Meaning |
|---|---|
| `accession` | Identifier of the source protein (first token of the FASTA header) |
| `sequence_f5` | Peptide/window evaluated by NetMHCIIpan (the complete peptide in exact mode, or the winning 15 aa window in protein mode) |
| `core_9aa` | 9 aa binding core of the allele with the best %Rank **in normal orientation** (inverted alleles never determine this value) |
| `start` / `end` | Absolute coordinates in the source protein (1-indexed), recomputed by traceback against the Phase 3/4 region |
| `origen` | Phase 3 engines that detected the region, abbreviated (`Bp`/`Ed`/`Dt`/`Sn`) and joined by `'+'` (e.g. `'Bp'`, `'Bp+Ed'`, `'Dt+Sn'`, `'Bp+Dt'`); `'Consenso total'` if all 4 contribute |
| `n_alelos_promiscuos` | How many alleles of the panel (out of 27, or more if you used `--alelo-extra`) classify the peptide as SB/WB **in normal orientation**. This number decides the verdict |
| `n_alelos_evaluados` | Total size of the evaluated panel |
| `min_rank_el` | Best (lowest) %Rank among the alleles in normal orientation |
| `<motor>_score` | One column per engine active in that run (e.g. `bepipred_score`, `discotope_score`, ...): mean antigenicity score from Phase 3 (`NaN` if that engine did not detect that region) |

`<nombre>_candidatos_finales_mhc1.csv` (Phase 5b) has the same format
(the MHC-I analogue: `NETMHCPAN_REFERENCE_PANEL` instead of the MHC-II panel,
its own %Rank thresholds), plus two exclusive columns:
`netcleave_c_term_match` (bool, is there an EXACT predicted proteasomal cleavage at
the residue immediately after the binding core?) and
`netcleave_c_term_score` (raw score of that cleavage, `NA` if there was no match).

### Format of `<nombre>_constructo_metadata.csv` and `<nombre>_constructo_chequeo.csv`

`_constructo_metadata.csv` (Phase 7) has one row per SEGMENT of the
construct (epitope or linker, in the order in which they appear): `block`
(`'B-cell'`/`'HTL'`/`'CTL'`/`'Linker (...)'`/`'Adjuvante'`), `sequence`,
`start`/`end` (1-indexed, the position inside the construct, not of the
source protein), `source_accession`/`source_start`/`source_end`
(`None` for linker segments) and `source_score_note` (a readable summary
of the score that drove the selection of that epitope). Concatenating the
`sequence` column in order reconstructs `<nombre>_constructo.fasta` exactly.

`_constructo_chequeo.csv` (Phase 8) has a single row (the complete
construct) with the columns of the 4 engines combined: `algpred_score`/
`algpred_veredicto`, `toxinpred_score`/`toxinpred_veredicto`,
`iapred_score`/`iapred_categoria`, `signalp_prediction`/
`signalp_prob_other`/`signalp_prob_sp`/`signalp_cs_position`.

## Tests

The test suite (`tests/`, `pytest`, 219 tests) covers the pure logic of
each phase: input routing (FASTA vs. structure), structure
extraction (modified residues, chain selection), annotated union of N
engines in Phase 3, task/E-value selection in Phase 4, parsing of the NetMHCIIpan
`.xls` and exclusion of inverted alleles, coordinate traceback,
window deduplication and validation of `--alelo-extra`/`--pdb-mode`, without
depending on BepiPred, EpiDope, DiscoTope-3.0, ScanNet, BLAST+ or NetMHCIIpan
being installed: it does not invoke any real subprocess (the integration tests of
the 3 paths mock the 4 engines). The same criterion applies to Phases
4b/4c/5b/6/8: `test_algpred_engine.py` (the batch=1 workaround),
`test_netcleave_engine.py` (exact C-terminal cleavage matching),
`test_stackglyembed_engine.py` (the N-X-[S/T] sequon scanner),
`test_toxinpred_engine.py` (the batch=1 workaround), `test_iapred_engine.py`,
`test_signalp_engine.py` (parsing of `prediction_results.txt`) mock
`subprocess.run` so as not to depend on the external venvs/models;
`test_lanl_catnap_engine.py` does not need to mock anything (the engine never
invokes a subprocess, it is pure pandas over CSVs). `test_construct_assembly.py`
covers the whole of Phase 7 (top-N selection per class, dedup by `core_9aa`,
linkers, empty blocks, optional adjuvant) with the traceability
invariant (`"".join(metadata_df['sequence']) == construct_sequence`)
verified explicitly.

```bash
pip install -r requirements-dev.txt
pytest
```

These unit tests validate the logic of each engine in isolation, not
that the complete pipeline works end to end with the real binaries/venvs
installed. For that, running `pipeline.py` against a real input (see
"Usage" above) is still necessary. `STATUS.md` documents the last real
end-to-end validation (a complete run of the 15 phases against
`inputs/GP120.fasta`, a real HIV-1 Env).
