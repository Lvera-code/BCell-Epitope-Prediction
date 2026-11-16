# STATUS: standalone vaccine epitope discovery pipeline

A clean state document (not a session diary): it reflects the FINAL verified
state, not the history of back and forth to get there. When
updating, edit the corresponding section instead of adding a new one
marked "resolved later".

## Table A: 5-phase B-cell pipeline (already ported to Scipion-Chem)

| Original phase | Real destination | Branch | Status |
|---|---|---|---|
| Phase 2/3 BepiPred | `scipion-chem-bepipred` | `feat/gap-tolerant-window-mode` | Done, PR pending |
| Phase 2/3 EpiDope | `scipion-chem-fork` (= `Lvera-code/scipion-chem`) | `feat/epidope-protocol` | Done, PR pending |
| Phase 3 consensus | generic `ProtOperateSeqROI` | — | No code required |
| Phase 4 BLASTp | `scipion-chem-blast` | `feat/batch-roi-input` | Done, PR pending |
| Phase 5 NetMHCIIpan | `scipion-chem-netmhciipan` | `master` | Done and validated |
| — | `scipion-chem-bcellepitope` | `chore/migrate-to-upstream-plugins` | Deprecated, reference only |

## Table B: Scipion plugin `scipion-chem-tmbed`

Built and validated: `scipion3 test tmbed.tests...` passes.

TMbed is also wired into the standalone script, as Phase 3b (see Table C,
the TMbed row). This is not a duplicated installation: the
engine `src/engines/tmbed_engine.py` reuses the SAME venv/weights of this
plugin (`TMBED_PYTHON_BIN`/`TMBED_MODEL_DIR` in `settings.py`, pointing
directly at `scipion-chem-tmbed/.venv-tmbed` and
`scipion-chem-tmbed/tmbed_src/tmbed/models/t5`) through pure subprocess, WITHOUT
importing any module of the plugin (which depends on `pwchem`, not installed in
the pipeline's main venv). Only the parsing of the output format was
reimplemented (`extract_masking_regions`/`parse_predictions` of
`tmbed/utils/tmbed.py`), verified byte-for-byte against the plugin's own real
test case (VDAC1_P21796, 18 `TM_beta_strand` regions,
identical coordinates).

## Table C: tools wired into the STANDALONE SCRIPT (`pipeline.py`)

Real verified state. They all run 100% locally (subprocess over a dedicated
venv or an installed binary), without any network call at run
time.

| Tool | Installation | Python engine (`src/engines/`) | Phase | Notes |
|---|---|---|---|---|
| TMbed (TM/signal/intracellular masking) | venv at `scipion-chem-tmbed/.venv-tmbed` (REUSED from the Scipion plugin, see Table B), ProtT5-XL-U50 weights at `scipion-chem-tmbed/tmbed_src/tmbed/models/t5` (REUSED by StackGlyEmbed, same encoder) | `tmbed_engine.py` | 3b, BEFORE BLASTp (Phase 4) -- it runs on the COMPLETE sequence of each accession, not per candidate peptide | Output format `--out-format 1`: 'B'/'H'/'S'/'i' (TM strand/helix, signal, intracellular) are collapsed into masking regions; only 'o' (extracellular, non-membrane) is left untouched. 'i' is masked just like TM/signal (it used to be ignored) -- an accession without TM and without a signal peptide but 100% cytoplasmic (e.g. PSMD7) ends up covered end to end by a single `intracellular` region and loses all its candidates, an effect equivalent to excluding it completely without a separate flag. Parsing reimplemented in pure form (without importing the Scipion plugin, which depends on `pwchem`, not installed in the main venv) -- verified byte-for-byte against VDAC1_P21796 (18 `TM_beta_strand` regions, the same coordinates as the plugin's test). Cache by hash of the complete sequences (not of `union_df`): a change of a Phase 3 threshold does not invalidate the TMbed cache. |
| NetMHCpan-4.2 (MHC-I) | `B-Cell-Epitope-Prediction/netMHCpan-4.2/` | `netmhcpan_engine.py` | 5b, parallel to Phase 5 (MHC-II), NOT merged (different antigen presentation pathways) | **Panel of 23 alleles (12 HLA-A/B + 11 HLA-C, see below).** Buffer overflow of the binary in exact peptide mode for inputs >55aa (silent exit code 0, the limit does not change with the panel size) -- automatically routed to protein mode to avoid it. No `Inverted` column (unlike NetMHCIIpan, verified, not assumed). |
| AlgPred 2.0 (allergenicity) | venv at `scipion-chem-algpred/.venv-algpred` | `algpred_engine.py` | 4b (per peptide) and reused in 8 (complete construct) | Real bug of the upstream script: it blows up with `ValueError` if the batch has exactly 1 sequence (an sklearn reshape bug). Workaround: the sequence is duplicated and the extra row discarded. In Phase 8 this is the NORMAL path (always 1 sequence per run), not an edge case. |
| NetCleave (MHC-I cleavage) | venv at `scipion-chem-netcleave/.venv-netcleave`, bundled pre-trained model | `netcleave_engine.py` | Annotation inside the Phase 5b report | It checks whether there is an EXACT proteasomal cleavage at the residue immediately after the candidate accepted by NetMHCpan (not just "there is some cleavage in the region"). A complementary signal, not a filter. The output .xlsx is named `<stem>_<primer-token-del-header-fasta>_NetCleave.xlsx`; the wrapper uses glob, not the exact name. |
| StackGlyEmbed (N-glycosylation) | Repo cloned at `StackGlyEmbed/` (venv `.venv-stackglyembed`), `protein_bert` installed with `--no-deps`, ProteinBERT/ESM-2 650M/ProtT5 cached locally | `stackglyembed_engine.py` (its own sequon scanner) + `src/engines/stackglyembed_predict_local.py` (extraction+prediction, replaces the original scripts that called the network) | 4c (per peptide) | `StackGlyEmbed/` is a nested git repo (its own `.git`): git does NOT allow un-ignoring a file inside a nested repo with any `.gitignore` pattern -- that is why `stackglyembed_predict_local.py` lives in `src/engines/` (a normal versioned tree), not inside the clone. ESM-2 via `transformers.EsmModel` (really offline) instead of the original script's `torch.hub.load(...)` (which always hits the network). ProtT5 REUSES the TMbed weights (`Rostlab/prot_t5_xl_half_uniref50-enc`, same encoder). |
| LANL Immunology DB + CATNAP (bnAb cross-ref) | Local CSVs in `reference_db/` | `lanl_catnap_engine.py` (pure pandas, no subprocess) | 6, informative (only relevant for HIV Env) | Replaces bNAber (dead/parked domain). Substring cross-reference (longest-common-substring) against the 771 linear epitopes of `ab_all.csv` with a reportable epitope (out of 3799 total records; the rest are conformational, out of scope). Configurable threshold `LANL_CATNAP_MIN_OVERLAP` (6 aa default). Validated with real bnAbs (10E8, 2F5, Z13e1, m66) with real IC50 cross-referenced from CATNAP. |
| IEDB (regions with documented protection/neutralization) | Local CSV in `reference_db/iedb/` | `iedb_engine.py` (pure pandas, no subprocess) | 6c, informative, ALWAYS runs, B-cell only | Generalization of Phase 6 to any pathogen. Same substring cross-reference mechanism (`_longest_common_substring_len` REUSED from `lanl_catnap_engine.py`, without duplicating it), without an organism filter (the match already filters implicitly). `reference_db/iedb/bcell_protective_epitopes.csv`: an ALREADY FILTERED subset (3429 rows / 561 organisms) of the IEDB B-cell bulk export (1,688,617 rows / 3.24 GB raw, NEVER distributed -- filtered once only as a SETUP step, see README.md Section 17). Filter: `Object Type == 'Linear peptide'` + `Response measured` in {neutralization, protection from *} + `Qualitative Measure` starts with 'Positive'. Does NOT apply to HTL/CTL (the export is of B-cell/antibody assays, a different mechanism from MHC presentation). |
| Construct assembly | N/A (pure logic) | `construct_assembly.py` | 7 | See Table D. |
| ToxinPred2 (construct toxicity) | `pip install toxinpred2` in a dedicated Python 3.10 venv (`.venv-toxinpred2/`) | `toxinpred_engine.py` | 8 | ONNX model + blastp + MERCI database EMBEDDED in the wheel, zero separate downloads. Python 3.10 venv + `pandas==1.5.3` + `numpy<2` pinned (the packaged script uses `to_csv(sep="\n")`, which pandas>=2 rejects; the numpy>=2 ABI breaks pandas 1.5.3). Same batch=1 bug as AlgPred2. |
| IApred (intrinsic antigenicity of the construct) | `git clone github.com/sebamiles/IApred` + its own venv (`IApred/.venv-iapred/`) | `iapred_engine.py` | 8 | Replaces VaxiJen (not open-source, no local standalone/API). Pure SVM over physicochemical features. The repo's `requirements.txt` is incomplete (`imbalanced-learn`/`matplotlib`/`seaborn` are missing, installed by hand). `models_folder` is a path relative to the cwd: subprocess always with `cwd=IAPRED_HOME`. |
| SignalP-6.0 (signal peptide of the construct) | DTU Health Tech package (academic licence), copied to `signalp-6.0/` (9.2GB) + a dedicated Python 3.10 venv (`.venv-signalp/`) | `signalp_engine.py` | 8 | `slow-sequential` mode (same RAM as `fast`, ~6x slower, for CPU without GPU). Weights referenced by `--model_dir` directly, without duplicating. Python 3.10 venv + `torch>1.7,<2` + `numpy<2`. Real parsing bug: `prediction_results.txt` carries 2 comment lines `#`, not 1 -- `comment="#"` is used in `pd.read_csv`, not a fixed `skiprows`. |

## Table D: Phase 7 (construct assembly) + Phase 8 (construct check)

It evaluates allergenicity/toxicity/antigenicity on the FINAL assembled
MULTI-EPITOPE CONSTRUCT, not per individual peptide (that is already covered by
Phase 4b/4c, insufficient on its own for this check).

**Phase 6b, conservation (OPTIONAL, `conservation_engine.py`):** it only runs
if `--panel-conservacion <fasta>` is passed; if not, it is skipped completely (neither
`blastp` nor `makeblastdb` is invoked). It measures breadth (DISTINCT sequences
of the panel matched at ≥90% identity, not just the best hit) against a panel
of other strains/variants of the same pathogen, indexed once and cached
locally by content hash. It reuses the Phase 4 BLASTp execution
without duplicating it. It runs on B-cell/HTL/CTL BEFORE Phase 7, so that
Phase 7 remains pure logic (it receives `conservation_df` already computed).
Purely informative (it annotates `conservation_pct`, it does not discard or rank).

**Phase 7, assembly (`construct_assembly.py`, pure logic, no subprocess):**

- It selects the **top-3 candidates per class** (`Settings.CONSTRUCT_TOP_N_PER_CLASS`,
  configurable by environment variable, not exposed as a CLI flag):
  - **B-cell**: from `safe_df` (Phase 4 'Segura'), excluding `Allergen` (Phase 4b);
    ranked by the best available `{motor}_score`. It NO longer excludes by
    N-glycosylation (Phase 4c) -- antibodies against glycosylated
    regions have been described, so each candidate is only
    annotated with `glycosylated=True/False` (visible in the traceability
    metadata), without discarding it.
  - **HTL/CTL**: from the `'Candidato Valido'` entries of Phase 5/5b, collapsed by
    `core_9aa` (the same logic as the window deduplication of
    NetMHCIIpan/NetMHCpan) keeping the best row; CTL additionally
    prioritizes `netcleave_c_term_match == True` over promiscuity/%Rank.
    Same N-glycosylation criterion as B-cell: it is annotated, not excluded.
    The sequence inserted into the construct is `sequence_f5` (the complete
    evaluated window, with flanks), not just `core_9aa` -- the flanking
    residues also contribute to binding/recognition, they are not
    just filler.
- **Linkers** (standard convention of the field, not a fixed biological rule):
  `AAY` intra-CTL (proteasome cleavage site), `GPGPG` intra-HTL and
  inter-block (universal spacer, Livingston et al. 2002), `KK`
  intra-B-cell.
- **Block order: B-cell → HTL → CTL.** Final decision: there is no strong
  consensus in the literature on the optimal order (the linkers already guarantee
  correct release by processing, independently of position); it is
  anchored on B-cell because that was the project's original humoral focus (bnAb/HIV).
- **Without adjuvant.** An active decision NOT to include one by default (the
  choice of adjuvant -beta-defensin, PADRE, flagellin, L7/L12, etc.-
  requires biological/strategic criteria specific to the pathogen/host,
  out of scope). The design hook is already implemented (`adjuvant_sequence` in
  `assemble_construct`, with a rigid EAAAK linker) to add one without
  redesigning if it is decided later.
- **Without merging of overlapping epitopes between classes.** Final decision:
  merging epitopes of different classes (e.g. a B-cell that overlaps in
  position with an HTL) would break the semantics of the linkers -each
  block expects a peptide of THAT class, not a hybrid-. The INTRA-class
  merge is already resolved by Phase 3 (union of overlapping regions of the
  same engine type, before the classes are separated).
- 100% traceable metadata (`<input_stem>_constructo_metadata.csv`): one row
  per segment (epitope or linker), with its position in the construct, source accession/
  position, and the score that drove the selection. Invariant
  verified in tests and in real runs:
  `"".join(metadata_df['sequence']) == construct_sequence`.

**Phase 8, check (4 engines, see Table C):** AlgPred2 (reused, without a new
installation), ToxinPred2, IApred, SignalP-6.0. All 4 are informative
(none filters or aborts the pipeline).

## Validation performed

**End-to-end with real data** (`inputs/GP120.fasta`, a real HIV-1 Env,
861 aa, chosen because it contains real N-glyco sequons and known bnAb
epitopes, to force biologically sensible results instead of
synthetic ones):

| Phase | Real result |
|---|---|
| 1 (sanitation) | 1 record, 861 aa, OK |
| 2 (BepiPred+EpiDope) | both run |
| 3 (union) | 10 regions (9 Ed, 1 Bp+Ed) |
| 4 (BLASTp) | 10/10 'Segura' |
| 4b (AlgPred2) | 5 Allergen / 5 Non-Allergen |
| 4c (StackGlyEmbed) | 10 sequons evaluated, 6/10 'Glicosilado' |
| 5 (NetMHCIIpan, MHC-II) | 20/108 promiscuous candidates |
| 5b (NetMHCpan, MHC-I) + NetCleave | 7/452 promiscuous candidates, 7/7 with confirmed C-terminal cleavage |
| 6 (bnAb cross-ref) | 8/10 peptides match, 3 with a confirmed neutralizer (2F5/Z13e1/m66, gp41 MPER -- biologically correct) |
| 7 (assembly) | 127 aa construct (3 B-cell + 3 HTL + 3 CTL) |
| 8 (construct check) | Non-Allergen, Non-Toxin, intrinsic antigenicity "Low" (IApred), no signal peptide |

**Publication validation panel (17 real PDB structures):** in addition
to the single-case end-to-end validation (GP120.fasta) documented
above, the complete pipeline was run end to end, with the 14 real
external tools invoked by subprocess, without mocks, over
the 17 PDB structures of the structural validation panel cited in the
paper (`run_validation_panel.sh`, output in
`outputs/validation_panel/<antigeno>/`, one directory + one log per
structure). Each run is backed by verifiable evidence: the
real execution log (`outputs/validation_panel/<antigeno>.log`), the raw
CSVs of each phase with their content hash (`*.csv.inputhash`, confirming
that the result corresponds exactly to that input) and the final result
(`*_candidatos_finales.csv`, `*_constructo_metadata.csv`). The
content-hash auto-cache mechanism (see "Checkpointing" below) means that
a re-execution over the same input reuses the already verified
result instead of recomputing it. A "CACHE LOCAL" log entry is
not a simulated run, it is the same reproducibility guarantee already
validated in the Checkpointing block of this same section.

**Structure path** (`inputs/7c4s.pdb`, `structure_and_sequence` mode,
all 4 Phase 2 engines at once): Phase 3 produced regions with mixed
origins (`Bp+Ed+Sn`, `Ed+Dt`, confirming that the union of 4 simultaneous engines
works), and Phases 4b-8 ran without failing over the resulting
candidates. PIPELINE COMPLETED without errors on both paths.

**Checkpointing** (Phase 3b/4/4b/4c/5/5b/6/6b/7/8, content-hash auto-cache
of each phase's input): verified with 2-pass runs, the second
pass instantaneous (38s -> 0.4s in a real case), and changing a parameter
invalidates the checkpoint in cascade correctly.

**Phase 3b (TMbed):** `predict_tm_signal_regions` (real subprocess, not
mocked) against `VDAC1_P21796` (a mitochondrial channel of 19 beta strands, without a
signal peptide) reproduces EXACTLY the 18 `TM_beta_strand` regions
(the same 1-indexed coordinates) that the Scipion plugin's test validates,
confirming that the parsing reimplemented in `tmbed_engine.py` (without importing the
plugin, which depends on `pwchem`) is equivalent byte-for-byte.

Masking verified in both directions with real end-to-end runs:
on `GP120.fasta` (HIV-1 envelope, with a C-terminal cytoplasmic TM) and
the ovalbumin record in `positive_controls_suite.fasta` (not
transmembrane) TMbed detects the expected regions
but no Phase 3 candidate overlaps with them, so the union
passes unchanged to Phase 4. On `SLC8A1_P32418_AF.pdb` (a multi-pass
transporter, 10 TM helices + signal peptide) the masking DOES discard
real candidates: of 15 in `union_epitopes.csv`, 3 are removed in
`union_epitopes_masked.csv` for overlapping with TM/signal regions (32-51 with the
signal peptide 1-34; 251-325 and 924-941 with TM helices), confirming that Phase
3b filters actively, not just detects without effect. In all 3 cases the rest
of the phases (4b-8) completes without errors through to `PIPELINE COMPLETADO`.

**Complete topology (intra/extracellular).** The
previous validation of this paragraph was done with the original logic, which only
masked B/H/S ('i'/'o' were ignored), so an accession without TM and without a signal
peptide but entirely cytoplasmic (a real case: PSMD7, a subunit of the
26S proteasome) had nothing to mask under that rule, and therefore its
candidates passed intact to Phase 4 despite being inaccessible to antibodies.
Research prior to implementing:
TMbed with `--out-format 1` ALREADY reports complete per-residue topology ('i'
cytoplasmic / 'o' extracellular in addition to B/H/S), so there was no need
to add DeepLoc/DeepTMHMM. `_MASKED_CLASS_TYPES` was corrected to treat
'i' as one more maskable region (`intracellular`), the same mechanism as
TM/signal, without a new flag at the protein level. A desired side effect,
confirmed with Enzo before implementing: the cytoplasmic loops of
multi-pass proteins (e.g. SLC8A1, 199 'i' residues between its 11 TM helices)
are now also masked, correcting the same bug in its most
subtle form. Validated by reprocessing the raw `.pred` files ALREADY cached from real
runs (without re-executing TMbed): PSMD7 (`iiii...i`, 325/325 residues)
produces a single `intracellular` region 1-324 that discards the 3
candidate rows of `union_epitopes.csv` (kept=0), a complete exclusion
confirmed; THBS2 (signal 1-18 + 'o' in the rest) produces only
`signal_peptide` 1-18, and of 19 candidate rows only the one that
already overlapped with the signal peptide is discarded (pre-existing behaviour, not new),
the remaining 18 pass intact, without undue exclusion. 2 new dedicated
tests (`test_psmd7_intracelular_sin_tm_ni_senal_se_excluye_por_completo`,
`test_thbs2_secretada_pasa_integra_sin_exclusion_indebida`) in
`tests/test_tmbed_engine.py`, 221/221 tests of the complete suite pass.

**Test suite** (`pytest tests/`): it includes dedicated coverage of
`tmbed_engine.py`, without depending on any external venv/binary being installed
(pure logic + mocked `subprocess.run`).

**Real bug found and fixed** (exposed by the first real run
of Phase 3b with candidates that DO reach Phase 4b):
`algpred_engine.py`/`toxinpred_engine.py`/`iapred_engine.py` (Phase 4b and 8)
forced the subprocess `cwd=` to the folder of the installed external script
(necessary so that that script finds its own relative resources),
but passed it as an output argument `output_dir / archivo` WITHOUT
resolving to absolute. With the default relative `output_dir`
(`Settings.FASTA_OUTPUT_DIR = 'outputs'`), the child process
interpreted that relative path against ITS OWN cwd (not that of
`pipeline.py`), blowing up with `OSError: Cannot save file into a
non-existent directory: 'outputs'`. It never fired before because (a)
the unit tests of these 3 engines always pass `tmp_path` (already
absolute) as `output_dir`, and (b) no previous real run with
non-empty 'Segura' candidates had reached Phase 4b since the
mocking mechanism has existed. It only fired with GP120 (10/10 'Segura'), the
FIRST real run with data that genuinely reaches Phase 4b/AlgPred2 since
Phase 3b was added. Fix: resolve the output path to absolute
(`.resolve()`) BEFORE building the subprocess command, in all 3
engines. `stackglyembed_engine.py`/`netcleave_engine.py` did NOT have this bug
(their subprocess I/O already uses absolute paths from a `tempfile.TemporaryDirectory`,
and the copy to `output_dir` happens afterwards in pure Python, without going through the
subprocess). 3 regression tests added (one per engine, relative `output_dir`
+ `monkeypatch.chdir`), verifying that the output argument passed
to the subprocess is always absolute.

**The NetMHCpan (MHC-I) allele panel includes HLA-C:** the original panel
(`NETMHCPAN_REFERENCE_PANEL`, 12 alleles) only covered HLA-A/B, without any
justification documented in the
repo for the absence of HLA-C. It turned out to be an inherited omission, not a
deliberate decision: neither Sidney et al. 2008 (the supertypes that this
panel already used) nor the homologous 27-allele IEDB panel for MHC-I (Weiskopf et al.
2013, PNAS) cover HLA-C; both references of the field are exclusively
HLA-A/B. 11 HLA-C alleles were added (`NETMHCPAN_REFERENCE_PANEL` now
has 23), chosen for appearing consistently in two
independent sources: Rasmussen et al. 2014 (J Immunol, an experimental
characterization of the binding motifs of the most globally common HLA-C) and the
population frequency criterion >=1% that IEDB recommends (applied in a
2020 SARS-CoV-2 vaccine design study that evaluated HLA-A/B/C together).
See the full docstring of `netmhcpan_engine.py` for the detail and the
exact citations.

Real validation against the local binary (not just literature): (a) the 11
alleles confirmed present in `netMHCpan-4.2/data/allelenames` and with a
pseudo-sequence in `MHC_pseudo.dat` BEFORE adding them to the panel; (b) a
real run with the 23-allele panel against 2 known reference epitopes
(NLVPMVATV/CMV pp65, GILGFVFTL/Flu M1, both HLA-A*02:01) without
errors, with biologically sensible results; (c) the buffer overflow limit
of the binary in exact peptide mode verified (55 aa OK, 57 aa crash)
with the 23-allele panel. The limit does not change, it is a property of the
peptide length, not of the size of the allele panel.

**Explicit decision that was left unchanged:**
`Settings.NETMHCPAN_MIN_PROMISCUOUS_ALLELES` stays at 3 even though the panel
went from 12 to 23 alleles (3/23 is a more lax bar than 3/12). It follows
the same convention already established by the MHC-II panel (27 alleles, the same
fixed threshold of 3, never scaled as a fraction). Redefining "promiscuity"
is out of scope for adding HLA-C to the panel, so this threshold was
left intact on purpose. If in the future it is decided that it should scale with
the panel size, review `Settings.NETMHCPAN_MIN_PROMISCUOUS_ALLELES`
in `settings.py`.

## Non-negotiable constraint

Everything must run locally, never network calls at pipeline run
time. Installing/downloading weights is a one-off SETUP step, not
runtime. Every new wrapper is audited so that it never fires `requests`,
`from_pretrained`/`torch.hub.load` without a local path, or any equivalent.

## Design decisions in force (so as not to ask again)

- **MHC-I (NetMHCpan-4.2) is an independent phase in parallel to MHC-II**
  (Phase 5b vs. Phase 5), never merged: they are different antigen
  presentation pathways.
- **bNAber replaced by the LANL Immunology DB** (there is no recoverable
  mirror of bNAber).
- **AlgPred2 = Phase 4b (per peptide, parallel to 5/5b), NetCleave = annotation
  inside Phase 5b, StackGlyEmbed = Phase 4c (per peptide), bnAb cross-ref =
  Phase 6 (informative)**: allergenicity/N-glycosylation are properties of
  the sequence itself, not tied to a presentation pathway; proteasomal
  cleavage IS mechanically tied to MHC-I.
- **Phase 4b (per peptide) does NOT replace the construct-level check**:
  they are different questions ("is this candidate peptide safe" vs. "is
  the final assembled construct safe"). Resolved with Phase 7/8, see
  Table D.
- **NetMHCpan-4.2 lives inside `B-Cell-Epitope-Prediction/`**; AlgPred2/
  NetCleave/StackGlyEmbed were NOT moved (they live in their sibling repos):
  moving a venv drags embedded absolute paths along and can break it, so they are
  referenced by a configurable absolute path in `Settings`.
- **StackGlyEmbed/IApred are nested git repos** (their own `.git`): git does not
  allow un-ignoring a file inside a nested repo with any
  `.gitignore` pattern. If it is necessary to write integration code
  of our own (as with StackGlyEmbed), that code lives in `src/engines/`, never
  inside the clone. If the original CLI already serves as-is without modifying it
  (as with IApred), there is no problem in ignoring the whole repo.
- **ToxDL2 evaluated and discarded** (see Table D): ToxinPred2 covers the
  construct toxicity check without the InterPro domain problem
  that blocked ToxDL2.
- **VaxiJen discarded** (not open-source, no local standalone/API),
  replaced by IApred.

## Robustness testing

Edge cases tested, beyond the end-to-end validation already
documented above (they confirm that the pipeline is robust, not just
functional on the happy path):

- **Structure path with Phase 7/8** (not tested until now): `7c4s.pdb`
  produced 2 B-cell candidates and ZERO HTL/CTL. The construct was assembled
  correctly with a single block (38 aa), without empty blocks dangling or
  orphan linkers. Phase 8 ran without failing over that short construct.
- **Completely empty construct** (via `--identity-threshold 1`, forcing
  0 'Segura' candidates): Phase 7/8 handle it cleanly, `PIPELINE
  COMPLETADO` without error.
- **Multi-record FASTA** (`MonkeyPoxSequences.fasta`, 6 proteins):
  a 314 aa construct assembled correctly combining candidates from 3
  distinct accessions (`WEN68160.1`, `AGR38652.1`, `AGR38316.1`),
  traceability verified exact.
- **Input errors** (non-existent file, empty FASTA, FASTA without a
  `>` header): all 3 cases end with a clear error message and
  `exit code 1`, never a raw unhandled traceback.
- **Extreme sequences directly against the 4 Phase 8 engines** (AlgPred2,
  ToxinPred2, IApred, SignalP-6.0): 1 aa, 2-3 aa, homopolymer (30x 'A'),
  a 1000 aa sequence. No crashes in any case.

**Real bug found and fixed:** IApred requires a MINIMUM
of 20 aa (verified by reading `IApred.py`). For shorter sequences it
writes the literal text `'Sequence too short'` in the score column (not
a number). `iapred_engine.py` did not handle it: `print_iapred_report`
formats the score with `:.4f`, which blows up with `TypeError` over a string.
This is a REAL scenario, not hypothetical: a construct with a single short
B-cell candidate (Phase 3 allows regions from 9 aa) falls below that
threshold, confirmed by reproducing it with `fase_8_chequeo_constructo`
directly over a real 15 aa construct. Fixed: the score is
coerced to an explicit `NaN` (`pd.to_numeric(errors='coerce')`, never a
string mixed into the column) with an informative category
(`'No evaluado (secuencia < 20 aa)'`), and the table formatter tolerates
`NaN`. 3 regression tests added (`test_iapred_engine.py`).

Complete suite: **248 tests**, without regressions (it includes `test_iedb_engine.py`
and the extension of `test_construct_assembly.py` for Phase 6c).

## Scipion-readiness audit

Sequencing decision in force: standalone-script-first, integration into
Scipion as the next step (see Table A/B for what is ALREADY ported from
the previous version of the pipeline). New Scipion protocols remain out of
scope of this document; it was audited that the current architecture does not
introduce anything that complicates that future migration:

- **No `argparse` inside `src/engines/`**: the only module with
  `argparse` is `stackglyembed_predict_local.py`, which is a standalone
  SCRIPT invoked by subprocess (never imported as a Python module),
  exactly like the CLIs of AlgPred2/NetCleave/ToxinPred2/IApred/
  SignalP-6.0. It is not orchestration logic mixed into an engine.
- **No interactive `input()`** in any engine (confirmed with grep).
  The only point that could in theory ask for interactive confirmation
  (`ProteinBERT.load_pretrained_model`) is always invoked with
  `download_model_dump_if_not_exists=False`.
- **No mutable global state** at module level in any engine (only
  constants: `_OUTPUT_COLUMNS`, reference panels, etc.), so concurrent/parallel
  runs of protocols would not share state by accident.
- **File I/O confined to the explicit parameters** (`output_dir`,
  temporary directories): no engine writes outside those paths.
- **Returns always typed DataFrames with documented columns**, a natural
  base on which to build Scipion Sets/Objects.
- **Errors via a typed hierarchy** (`EngineExecutionError` and subclasses,
  `src/utils/exceptions.py`), not generic exceptions, so a Scipion
  protocol can catch them and report them cleanly in the GUI.
- **Separation of `predict_*` (logic) vs. `print_*_report` (console only)**:
  a Scipion protocol would call only the former.

Conclusion: no refactor is needed before porting these engines to
Scipion protocols.

## Repository portability

All the `Settings` defaults that point INSIDE this repository
(venvs, weights, `reference_db/`) use paths relative to the project root,
never absolute ones, so any clone works without editing `settings.py`. The
defaults that point to tools living in sibling repos (AlgPred2 in
`scipion-chem-algpred/`, NetCleave in `scipion-chem-netcleave/`, TMbed in
`scipion-chem-tmbed/`) ARE absolute paths out of necessity (they are not part of
this repo) and must be adjusted via an environment variable if those sibling repos
live in another location. See the comment of each corresponding
`Settings.*_PYTHON_BIN` in `src/config/settings.py`.

## Open items

Nothing blocked for lack of information. The only thing out of scope of this document:

1. **Integration into Scipion**: a sequencing decision, standalone-script-
   first, Scipion-integration afterwards (see Table A/B for what is ALREADY
   ported, and "Scipion-readiness audit" above).
2. Re-run PSMD7/PODXL/THBS2 (AlphaFold structures, PDB path) with the
   current 15-phase pipeline, if it has not already been done since the last
   clearing of the `outputs/` cache. This is the same code path already confirmed with
   SLC8A1 (same family, a membrane protein) and GP120.
3. **Construct conformation vs. native protein**: deliberately discarded as a
   modification of this pipeline, it requires real structural
   prediction (GPU) and is left for Project 3 (bridge/TFG), not for
   `BCell-Epitope-Prediction`.

**Documented regions of interest: IMPLEMENTED** as Phase 6c (`iedb_engine.py`, see
Table C). It is no longer pending and is withdrawn from the list above.
