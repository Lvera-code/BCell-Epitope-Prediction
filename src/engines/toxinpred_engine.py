"""Toxicity of the construct via LOCAL ToxinPred 2.0 (dedicated venv, pure subprocess).

A 100% local wrapper over ``toxinpred2`` (Raghava group, open source,
``pip install toxinpred2``), the same pattern as ``algpred_engine.py``. Unlike
ToxinPred 3.0 (intended for short peptides), the group that develops it
recommends ToxinPred2 for full-length proteins/constructs, which is
exactly the Phase 8 use case (checking the assembled multi-epitope
construct, not individual peptides).

The model (Random Forest, exported to ONNX) and the ``blastp`` binary (used
by the ``-m 2`` hybrid mode) come EMBEDDED in the pip package
(``toxinpred2/model/RF_model.onnx.zip``, ``toxinpred2/blast_binaries/``):
there is no separate weight download step, neither at setup nor at runtime.

A dedicated venv with Python 3.10 (not 3.13, the system default) + pandas
``1.5.3`` + ``numpy<2`` pinned, because of two real bugs verified
empirically during installation:

1. The packaged script writes its intermediate FASTA with
   ``CM.to_csv("Sequence_1", header=False, index=None, sep="\n")``, a
   separator of more than 1 character, which ``pandas>=2`` rejects outright with
   ``ValueError: bad delimiter value`` (pandas requires a ``sep`` of length 1).
   There is no CLI flag that avoids this step: it always executes, before
   any ``Model``/``Display`` branch. The only viable way out without
   patching the installed package: pin a version of pandas earlier than
   that strict validation.
2. ``pandas==1.5.3`` is compiled against the ``numpy<2`` ABI: with
   ``numpy>=2`` (what ``pip install toxinpred2`` brings by default without
   pins) the pandas import fails with
   ``ValueError: numpy.dtype size changed, may indicate binary incompatibility``.

Model (``Settings.TOXINPRED2_MODEL``, 1 by default): the upstream script
supports two modes with DIFFERENT output CSV formats, just like
``algpred_engine.py``, which is why this wrapper parses each one separately:

    Model 1 (pure ML, DEFAULT): columns ``ID,Sequence,ML_Score,Prediction``,
        a Random Forest over amino acid composition only, without
        any comparison against real toxins or motifs.
    Model 2 (hybrid): columns ``Subject,ML Score,MERCI Score,BLAST Score,
        Hybrid Score,Prediction``, the same RF plus BLAST against the embedded
        toxin database and MERCI against documented motifs (``Hybrid
        Score`` = the sum of the three). It does NOT carry a ``Sequence`` column: the
        sequence is reconstructed BY POSITION, additionally verifying that the
        identifiers returned match one to one with those sent.

The default remains model 1 on purpose, so as not to alter already
generated outputs. Changing it to 2 modifies ``*_constructo_chequeo.csv`` (three
new evidence columns) and forces a re-run of the validation panel,
which is exactly what was done at the time with AlgPred2.

The hybrid mode is NOT invoked through the venv's ``toxinpred2``
executable, but through the venv's interpreter against
``src/engines/toxinpred2_hybrid_shim.py``, which fixes at run
time a real upstream ``NameError`` (``seqid`` as a free variable
in ``BLAST_processor``) that fails whenever BLAST returns no
hit, that is, in the normal case of a short non-toxic peptide. See the
docstring of that module for the detail and for why it is fixed there and not
by patching the installed package.

Batch-of-size-1 bug (verified empirically, the same pattern as
``algpred_engine.py``): the ONNX model expects a rank 2 tensor
(``(n_muestras, n_features)``); with a single input sequence, the
script's feature pipeline produces a rank 1 array and ONNX Runtime
fails with ``INVALID_ARGUMENT: Invalid rank for input``. This is NOT a
marginal edge case here: Phase 8 evaluates a single assembled construct
per run, so this workaround (duplicate the sequence, discard the extra
row from the result) is activated on the normal path, not only in tests.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import EngineExecutionError
from src.utils.logger_config import setup_logger
from src.utils.table_format import Column, print_fixed_width_table

logger = setup_logger(__name__)

_OUTPUT_COLUMNS = ["sequence", "toxinpred_score", "toxinpred_veredicto"]


def _resolve_binary() -> Path:
    """Valida que el interprete y el script de consola de ToxinPred2 existan.

    ``toxinpred2`` se instala como entry point de consola (``pip install
    toxinpred2``, sin ``__main__.py`` propio: ``python -m toxinpred2`` NO
    funciona), asi que se invoca el ejecutable ``toxinpred2`` del ``bin/``
    del venv directamente, no el interprete + un path de script como en
    ``algpred_engine.py``/``netcleave_engine.py`` (esos si son scripts .py sueltos).
    """
    python_bin = Path(Settings.TOXINPRED2_PYTHON_BIN)
    if not python_bin.is_file():
        raise EngineExecutionError(
            f"No se encontro el interprete Python del venv de ToxinPred2 en '{python_bin}'. "
            "Ver README.md (Seccion 13) o apunta TOXINPRED2_PYTHON_BIN a la ubicacion correcta."
        )
    binary = python_bin.parent / Settings.TOXINPRED2_BINARY_NAME
    if not binary.is_file():
        raise EngineExecutionError(
            f"No se encontro el ejecutable '{binary}'. Ver README.md (Seccion 13) o reinstala ToxinPred2 "
            f"en el venv de TOXINPRED2_PYTHON_BIN ('pip install toxinpred2')."
        )
    return binary


def predict_toxicity(
    sequences: List[str], output_dir: Path, filename_prefix: str = "", threshold: float = None
) -> pd.DataFrame:
    """Evalua toxicidad de cada secuencia con ToxinPred 2.0 local.

    Args:
        sequences: Secuencias a evaluar (tipicamente un unico constructo
            ensamblado en Fase 8, pero acepta cualquier batch). Vacio -> DataFrame vacio.
        output_dir: Carpeta donde persistir el CSV crudo devuelto por ToxinPred2.
        filename_prefix: Prefijo (tipicamente ``f"{input_stem}_"``) para el CSV crudo.
        threshold: Umbral de score (por defecto ``Settings.TOXINPRED2_THRESHOLD``, 0.6).

    Returns:
        DataFrame con columnas ``sequence``, ``toxinpred_score``, ``toxinpred_veredicto``
        (``'Toxin'`` / ``'Non-Toxin'``, texto crudo de ToxinPred2).

    Raises:
        EngineExecutionError: Si el venv no esta instalado, el subproceso
            falla/excede el timeout, o el CSV de salida no tiene el formato esperado.
    """
    if not sequences:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    binary = _resolve_binary()
    threshold = threshold if threshold is not None else Settings.TOXINPRED2_THRESHOLD

    # Workaround del bug de tamano de batch == 1 (ver docstring del modulo).
    padded = len(sequences) == 1
    input_seqs = sequences + [sequences[0]] if padded else sequences

    output_dir.mkdir(parents=True, exist_ok=True)
    # Resuelto a absoluto: el subprocess de abajo corre con 'cwd=tmp' (el
    # directorio temporal del batch), asi que un 'output_dir' relativo
    # (default de Settings.FASTA_OUTPUT_DIR) se resolveria contra ESE
    # directorio, no el de pipeline.py -- mismo bug real confirmado en
    # algpred_engine.py, ver su docstring inline.
    raw_csv_path = (output_dir / f"{filename_prefix}toxinpred_raw.csv").resolve()

    with tempfile.TemporaryDirectory(prefix="toxinpred_") as tmp:
        fasta_path = Path(tmp) / "candidates.fasta"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for i, seq in enumerate(input_seqs):
                fh.write(f">candidato_{i}\n{seq}\n")

        cmd = [
            str(binary),
            "-i", str(fasta_path), "-o", str(raw_csv_path),
            "-t", str(threshold), "-d", "2",
        ]
        logger.info("Ejecutando ToxinPred2 local: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=Settings.TOXINPRED2_TIMEOUT_SECONDS, cwd=tmp)
        except subprocess.CalledProcessError as exc:
            raise EngineExecutionError(
                f"ToxinPred2 termino con exit code {exc.returncode}: {(exc.stderr or '<sin stderr>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(f"ToxinPred2 excedio el tiempo limite de {Settings.TOXINPRED2_TIMEOUT_SECONDS}s.") from exc

    if not raw_csv_path.is_file():
        raise EngineExecutionError(f"ToxinPred2 termino sin error pero no genero el CSV esperado en '{raw_csv_path}'.")

    raw = pd.read_csv(raw_csv_path)
    if not {"Sequence", "ML_Score", "Prediction"}.issubset(raw.columns):
        raise EngineExecutionError(
            f"El formato del CSV de ToxinPred2 no coincide con lo esperado. Columnas encontradas: {list(raw.columns)}."
        )

    if padded:
        raw = raw.iloc[: len(sequences)]

    result = pd.DataFrame(
        {
            "sequence": raw["Sequence"],
            "toxinpred_score": raw["ML_Score"],
            "toxinpred_veredicto": raw["Prediction"],
        }
    )
    return result[_OUTPUT_COLUMNS]


def print_toxicity_report(report_df: pd.DataFrame) -> None:
    """Imprime el informe de toxicidad: analogo a ``algpred_engine.print_allergenicity_report``."""
    if report_df.empty:
        print("No hay secuencias candidatas para evaluar toxicidad.")
        return

    columns = [
        Column("ML_Score", lambda r: f"{r.toxinpred_score:.4f}", 12, ">"),
        Column("Veredicto", lambda r: r.toxinpred_veredicto, 16, ">"),
    ]
    print_fixed_width_table(report_df.itertuples(index=False), columns)

    n_toxin = int((report_df["toxinpred_veredicto"] == "Toxin").sum())
    n_non_toxin = int((report_df["toxinpred_veredicto"] == "Non-Toxin").sum())
    print(f"\nResumen toxicidad: {n_non_toxin} no toxico(s) / {n_toxin} toxico(s) potencial(es).")
