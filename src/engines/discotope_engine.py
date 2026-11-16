"""Phase 2 (structural engine 1/2, local) antigenicity prediction via DiscoTope-3.0.

The same architectural pattern as ``bepipred_engine.py``/``epidope_engine.py``:
this class is a pure ``subprocess`` wrapper over the official CLI of
DiscoTope-3.0 (https://github.com/Magnushhoie/DiscoTope-3.0, DTU Health Tech,
Creative Commons licence for free academic use), without any network call
at inference time, other than the initial, single download of the ESM-IF1
weights, cached outside the repo (see ``Settings.DISCOTOPE_WEIGHTS_CACHE_DIR``
and ``_build_env`` below).

The key difference with respect to BepiPred-3.0/EpiDope: ``TIn`` is a path to a PDB, not
to a FASTA (``src.utils.structure_parser.StructureRecord.chain_pdb_path``, a
SINGLE-chain PDB, already isolated by Phase 1.5, never the potentially
multi-chain original PDB).

ADR: why a single-chain PDB and not the original
-----------------------------------------------------------
DiscoTope-3.0 separates the input PDB into individual chains and writes ONE
output CSV PER CHAIN (see the official README, section "DiscoTope-3.0
output"). If it were given the original (potentially multi-chain) PDB there
would be no reliable way to know, from this engine, which of those CSVs
corresponds to the chain that Phase 1.5 chose as the reference. Giving it a
single-chain PDB removes the ambiguity at the root: it can only produce one CSV.

ADR: accession reconciliation
-----------------------------------
This engine, like BepiPred/EpiDope, makes NO business adjustment to the
accession: it reports ``Path(pdb_path).stem`` as-is (for example
``'7C4S_chain_A'``, the name of the single-chain PDB it received), never
the raw ``'pdb'`` column of DiscoTope's CSV (which carries its own composite
format ``'{stem}_{cadena}'``, not directly comparable). Reconciling that
value with the "real" accession (``StructureRecord.accession``, without the
``_chain_<id>`` suffix) is the orchestrator's responsibility (``pipeline.py``, Phase 2),
which already has the complete ``StructureRecord`` in hand, the same division of
responsibilities that ``consensus.accession_id()`` already applies for BepiPred/EpiDope.
"""

import glob
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.config.settings import Settings
from src.engines.base_engine import BaseEngine
from src.engines.epitope_mapping import extract_epitope_regions
from src.engines.epitope_mapping import print_epitope_table as _print_epitope_table
from src.utils.exceptions import DiscoTopeExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

ACCESSION_COLUMN = "Accession"
RESIDUE_COLUMN = "Residue"
SCORE_COLUMN = "DiscoTope-3.0 calibrated score"
RSA_COLUMN = "DiscoTope-3.0 rsa"
PLDDT_COLUMN = "DiscoTope-3.0 pLDDT"
ALPHAFOLD_STRUC_FLAG_COLUMN = "DiscoTope-3.0 alphafold_struc_flag"

# Columnas confirmadas EMPIRICAMENTE corriendo la instalacion real (el CSV
# real trae mas columnas que las documentadas explicitamente en el README:
# 'pdb,chain,res_id,residue,DiscoTope-3.0_score,calibrated_score,epitope,
# rsa,pLDDTs,length,alphafold_struc_flag'). Se usa 'calibrated_score', NO la
# 'DiscoTope-3.0_score' cruda (0-1) -- ver ADR "Por que calibrated_score" mas
# abajo. 'res_id' se ignora a favor del orden de filas (mismo criterio
# posicional que el resto del pipeline), 'epitope' (booleano propio de
# DiscoTope) y 'pdb' se ignoran (ver ADR de reconciliacion de accession).
_RAW_RESIDUE_COLUMN = "residue"
_RAW_SCORE_COLUMN = "calibrated_score"

# Retenidas para Proyecto 3/BoltzGen (ver ADR "Por que se retienen
# rsa/pLDDTs/alphafold_struc_flag" mas abajo). OPCIONALES a proposito: ningun
# consumidor de Project 1 las requeria hasta ahora (por eso se descartaban),
# asi que si una version futura de DiscoTope-3.0 deja de emitirlas, el
# pipeline de Project 1 no debe romperse -- se completan con NA, nunca con un
# KeyError.
_RAW_RSA_COLUMN = "rsa"
_RAW_PLDDT_COLUMN = "pLDDTs"
_RAW_ALPHAFOLD_STRUC_FLAG_COLUMN = "alphafold_struc_flag"

# ADR -- por que se retienen rsa/pLDDTs/alphafold_struc_flag
# ----------------------------------------------------------------
# Encontradas auditando el input de Proyecto 3 (BoltzGen): DiscoTope-3.0 ya
# las calcula, pero hasta ahora se descartaban en '_load_raw_scores' junto
# con el resto de columnas no usadas. Ninguna influye en 'calibrated_score'
# como filtro adicional en Project 1 -- se retienen como metadata pura,
# mismo patron no-decisorio que 'tmbed_masked'/'glycosylated'.
# - 'rsa' (accesibilidad solvente relativa, por residuo): distinto de
#   'tmbed_masked' -- TMbed detecta "esta insertado en membrana/es
#   intracelular" (contexto de dominio), 'rsa' detecta "esta enterrado
#   dentro del propio plegamiento del dominio soluble" (fenomeno distinto,
#   un motor de antigenicidad puede puntuar alto una ventana secuencialmente
#   plausible que en 3D no es accesible).
# - 'pLDDTs'/'alphafold_struc_flag' (confianza por residuo de la estructura,
#   y si el input es un modelo predicho -AlphaFold- o experimental):
#   relevante para dianas futuras de Proyecto 3 sin estructura experimental
#   resuelta -- disenar contra una ventana de baja confianza estructural es
#   construir sobre una hipotesis 3D poco fiable.

# ADR -- por que 'calibrated_score' y no 'DiscoTope-3.0_score'
# ----------------------------------------------------------------
# La columna cruda 'DiscoTope-3.0_score' (0.00-1.00) requeriria un umbral
# CALIBRADO A MANO contra una estructura de ejemplo (p. ej. 7c4s) -- una
# limitacion real, sin respaldo de los autores. 'calibrated_score' es una
# columna DISTINTA que el propio
# DiscoTope-3.0 ya calcula (normalizada por longitud de cadena y superficie
# accesible, ver README oficial) y para la que los autores SI publican
# umbrales de referencia con recall esperado (confirmado via el paper,
# Frontiers in Immunology 2024): ~0.40 (top 70mo percentil, recall ~70%,
# "low"), ~0.90 ("moderate", el default del propio flag CLI
# '--calibrated_score_epi_threshold'), ~1.51 (top 30mo percentil, mayor
# precision/menor recall, "higher"). Usar esta columna reemplaza una
# calibracion casera por la calibracion oficial de los autores.


class DiscoTopeEngine(BaseEngine[str, pd.DataFrame]):
    """Ejecuta DiscoTope-3.0 LOCALMENTE (subprocess) para cada PDB de una sola cadena.

    Uso exclusivo de inferencia cruda: no aplica ningun umbral de mapeo de
    regiones aqui mas alla de lo estrictamente necesario para el CLI (ver
    :func:`extract_epitopes` para el filtrado local de Fase 3).
    """

    def __init__(
        self,
        install_path: Path = Settings.DISCOTOPE_INSTALL_PATH,
        python_bin: str = Settings.DISCOTOPE_PYTHON_BIN,
        weights_cache_dir: Path = Settings.DISCOTOPE_WEIGHTS_CACHE_DIR,
        struc_type: str = Settings.DISCOTOPE_STRUC_TYPE,
    ):
        self._install_path = Path(install_path)
        self._python_bin = python_bin
        self._weights_cache_dir = Path(weights_cache_dir)
        self._struc_type = struc_type

    def _main_script_path(self) -> Path:
        return self._install_path / "discotope3" / "main.py"

    def _models_dir(self) -> Path:
        return self._install_path / "models"

    def _validate_installation(self) -> None:
        """Comprueba que el entorno local de DiscoTope-3.0 este instalado y accesible.

        Raises:
            DiscoTopeExecutionError: Con instrucciones de instalacion completas
                (a diferencia de BepiPred-3.0, DiscoTope-3.0 es instalable
                directo via git+pip, sin solicitud academica) si falta el
                entorno, el script principal o los pesos del ensemble XGBoost.
        """
        python_path = Path(self._python_bin)
        if not python_path.is_file():
            raise DiscoTopeExecutionError(
                f"No se encontro el interprete Python de DiscoTope-3.0 en '{python_path}'. "
                f"Instala DiscoTope-3.0 en un entorno virtual dedicado ({Settings.DISCOTOPE_DOWNLOAD_URL}):\n"
                f"  git clone {Settings.DISCOTOPE_DOWNLOAD_URL} {self._install_path}\n"
                f"  python3 -m venv .venv-discotope && .venv-discotope/bin/pip install -r "
                f"{self._install_path}/requirements.txt\n"
                f"  .venv-discotope/bin/pip install {self._install_path}\n"
                "Ver README.md - Seccion de Instalacion."
            )

        if not self._main_script_path().is_file():
            raise DiscoTopeExecutionError(
                f"No se encontro '{self._main_script_path()}'. Verifica que "
                f"DISCOTOPE_INSTALL_PATH ('{self._install_path}') apunte al clon de "
                f"{Settings.DISCOTOPE_DOWNLOAD_URL}."
            )

        if not self._models_dir().is_dir() or not any(self._models_dir().glob("*.json")):
            raise DiscoTopeExecutionError(
                f"No se encontraron los pesos del ensemble XGBoost de DiscoTope-3.0 en "
                f"'{self._models_dir()}'. Descomprime 'models.zip' dentro del repo clonado:\n"
                f"  cd {self._install_path} && unzip models.zip"
            )

    def _build_env(self) -> dict:
        """Redirige el cache de pesos de ESM-IF1 (torch hub) a una ruta persistente propia.

        ESM-IF1 (usado internamente por DiscoTope-3.0 para las
        representaciones de inverse folding) descarga sus pesos via el cache
        estandar de ``torch.hub`` en la primera corrida. Fijar ``TORCH_HOME``
        a ``Settings.DISCOTOPE_WEIGHTS_CACHE_DIR`` (fuera del repo del
        proyecto) evita volver a descargarlos en cada entorno/corrida nueva.
        """
        self._weights_cache_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["TORCH_HOME"] = str(self._weights_cache_dir.resolve())
        return env

    def run(self, items: Sequence[str], output_dir: Optional[Path] = None) -> List[pd.DataFrame]:
        """Corre DiscoTope-3.0 localmente sobre cada PDB de una sola cadena en ``items``.

        Args:
            items: Rutas locales a PDB de una sola cadena (ver
                ``StructureRecord.chain_pdb_path``).
            output_dir: Carpeta donde guardar los artefactos crudos generados
                por DiscoTope-3.0 para cada PDB de ``items`` (subcarpeta por
                ``stem``). Si es ``None``, usa ``Settings.DISCOTOPE_OUTPUT_DIR``.

        Returns:
            Lista de DataFrames con los scores crudos por residuo (columnas
            ``Accession``/``Residue``/``DiscoTope-3.0 score``), en el mismo
            orden que ``items``, sin ningun filtrado.

        Raises:
            DiscoTopeExecutionError: Si la instalacion local no existe, el
                subproceso falla, excede el timeout, o la salida no tiene el
                formato esperado.
        """
        self._validate_installation()

        return [
            self._run_single(
                pdb_path,
                result_dir=(output_dir / Path(pdb_path).stem) if output_dir else None,
            )
            for pdb_path in items
        ]

    def _run_single(self, pdb_path: str, result_dir: Optional[Path] = None) -> pd.DataFrame:
        pdb = Path(pdb_path)
        if not pdb.is_file():
            raise FileNotFoundError(f"No se encontro el PDB de entrada: {pdb}")

        result_dir = result_dir or (Settings.DISCOTOPE_OUTPUT_DIR / pdb.stem)
        result_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._python_bin, str(self._main_script_path()),
            "--pdb_or_zip_file", str(pdb.resolve()),
            "--out_dir", str(result_dir.resolve()),
            "--struc_type", self._struc_type,
            "--models_dir", str(self._models_dir().resolve()),
        ]

        logger.info("Ejecutando DiscoTope-3.0 local para '%s': %s", pdb.name, " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=Settings.DISCOTOPE_TIMEOUT_SECONDS,
                env=self._build_env(),
            )
        except subprocess.CalledProcessError as exc:
            raise DiscoTopeExecutionError(
                f"DiscoTope-3.0 termino con exit code {exc.returncode} para '{pdb.name}'. "
                f"stderr: {(exc.stderr or '<vacio>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DiscoTopeExecutionError(
                f"DiscoTope-3.0 excedio el tiempo limite de {Settings.DISCOTOPE_TIMEOUT_SECONDS}s "
                f"para '{pdb.name}'. Aumenta DISCOTOPE_TIMEOUT_SECONDS si la estructura es muy "
                "grande o el hardware es lento (CPU vs GPU)."
            ) from exc

        return self._load_raw_scores(result_dir, accession=pdb.stem)

    @staticmethod
    def _load_raw_scores(result_dir: Path, accession: str) -> pd.DataFrame:
        """Localiza y carga el (unico) CSV '*_discotope3.csv' que escribe DiscoTope-3.0."""
        csv_files = sorted(Path(p) for p in glob.glob(str(result_dir / "**" / "*_discotope3.csv"), recursive=True))
        if not csv_files:
            found = sorted(p.name for p in result_dir.rglob("*") if p.is_file())
            raise DiscoTopeExecutionError(
                f"No se encontro ningun CSV de salida '*_discotope3.csv' en '{result_dir}'. "
                f"Archivos generados por DiscoTope-3.0: {found or '<ninguno>'}."
            )
        if len(csv_files) > 1:
            raise DiscoTopeExecutionError(
                f"Se encontro mas de un CSV de salida en '{result_dir}' ({[c.name for c in csv_files]}): "
                "se esperaba exactamente uno, ya que el PDB de entrada deberia contener una sola "
                "cadena (ver StructureRecord.chain_pdb_path). Verifica que el PDB de entrada no "
                "tenga mas de una cadena."
            )

        df = pd.read_csv(csv_files[0])
        missing = {_RAW_RESIDUE_COLUMN, _RAW_SCORE_COLUMN} - set(df.columns)
        if missing:
            raise DiscoTopeExecutionError(
                f"El CSV de salida '{csv_files[0]}' no contiene las columnas esperadas "
                f"{sorted(missing)}. Columnas encontradas: {list(df.columns)}."
            )

        df = df.rename(columns={_RAW_RESIDUE_COLUMN: RESIDUE_COLUMN, _RAW_SCORE_COLUMN: SCORE_COLUMN})
        df.insert(0, ACCESSION_COLUMN, accession)

        # rsa/pLDDTs/alphafold_struc_flag son OPCIONALES (ver ADR arriba): si la
        # instalacion real de DiscoTope-3.0 no las emite, se completan con NA en
        # vez de fallar -- ningun consumidor existente de Project 1 las requeria
        # antes de esto, asi que no deben convertirse en un nuevo punto de fallo.
        for raw_col, public_col in (
            (_RAW_RSA_COLUMN, RSA_COLUMN),
            (_RAW_PLDDT_COLUMN, PLDDT_COLUMN),
            (_RAW_ALPHAFOLD_STRUC_FLAG_COLUMN, ALPHAFOLD_STRUC_FLAG_COLUMN),
        ):
            df[public_col] = df[raw_col] if raw_col in df.columns else pd.NA

        return df[[
            ACCESSION_COLUMN, RESIDUE_COLUMN, SCORE_COLUMN,
            RSA_COLUMN, PLDDT_COLUMN, ALPHAFOLD_STRUC_FLAG_COLUMN,
        ]]


def extract_epitopes(
    raw_scores_df: pd.DataFrame,
    threshold: float = Settings.DISCOTOPE_THRESHOLD,
    min_length: int = Settings.DISCOTOPE_MIN_EPITOPE_LENGTH,
    window_size: int = Settings.DISCOTOPE_WINDOW_SIZE,
    max_gap_residues: int = Settings.DISCOTOPE_MAX_GAP_RESIDUES,
) -> pd.DataFrame:
    """Fase 3 (DiscoTope-3.0): mapea regiones de epitopo con la misma ventana deslizante.

    Logica identica a ``bepipred_engine.extract_epitopes``/``epidope_engine.extract_epitopes``
    (ver ``src.engines.epitope_mapping.extract_epitope_regions``), aplicada
    sobre los scores crudos de DiscoTope-3.0. El umbral por defecto
    (``Settings.DISCOTOPE_THRESHOLD`` = 0.90) es el propio de DiscoTope-3.0 y
    NO es comparable en escala al de BepiPred/EpiDope: cada motor conserva su
    propio umbral y sus propios parametros de ventana.

    Nota biologica: a diferencia de BepiPred-3.0/EpiDope, DiscoTope-3.0 puntua
    epitopos CONFORMACIONALES (parches 3D que pueden ser discontinuos en la
    secuencia lineal). Colapsar su score a regiones contiguas via ventana
    deslizante es una simplificacion deliberada para mantener el resto del
    pipeline (Fase 4 BLASTp, Fase 5 NetMHCIIpan) operando sobre peptidos
    lineales sintetizables, igual que con EpiDope.

    Returns:
        DataFrame con una fila por region de epitopo: ``accession``,
        ``start``, ``end``, ``length``, ``mean_score``, ``max_score`` y
        ``sequence``.
    """
    missing = {ACCESSION_COLUMN, SCORE_COLUMN} - set(raw_scores_df.columns)
    if missing:
        raise DiscoTopeExecutionError(
            f"El DataFrame de entrada no contiene las columnas requeridas {sorted(missing)}. "
            f"Columnas encontradas: {list(raw_scores_df.columns)}."
        )

    return extract_epitope_regions(
        raw_scores_df,
        accession_col=ACCESSION_COLUMN,
        score_col=SCORE_COLUMN,
        residue_col_candidates=(RESIDUE_COLUMN,),
        threshold=threshold,
        min_length=min_length,
        window_size=window_size,
        max_gap_residues=max_gap_residues,
    )


def print_epitope_table(epitopes_df: pd.DataFrame) -> None:
    """Imprime la tabla final de epitopos filtrados (DiscoTope-3.0) en consola."""
    _print_epitope_table(
        epitopes_df,
        empty_message="No se encontraron epitopos (DiscoTope-3.0) que superen el threshold y la longitud minima.",
    )
