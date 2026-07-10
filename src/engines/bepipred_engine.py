"""Fase 2 (local) + Fase 3 (local) de prediccion de antigenicidad via BepiPred-3.0.

ADR — pivote a ejecucion 100% local (2026-07-10)
--------------------------------------------------
Se abandono definitivamente la estrategia via BioLib (API/nube): bajo carga
publica, los cold-start de los contenedores ESM-2 subyacentes resultaron
impracticos (peticiones que no completaban ni siquiera tras ~1h de espera).
BepiPred-3.0 ahora se ejecuta como un subproceso local contra el codigo
fuente oficial descargado manualmente por el usuario (licencia academica DTU
Health Tech, ver ``Settings.BEPIPRED_DOWNLOAD_URL``). No se usa la libreria
``requests`` ni ninguna llamada de red: este modulo es puramente un wrapper
de ``subprocess.run`` sobre ``bepipred3_CLI.py``.

Division de responsabilidades resultante:

* Fase 2 (``BepiPredEngine``, esta clase): invoca el CLI local de BepiPred-3.0
  con ``subprocess.run``, sin aplicar ningun filtrado. Devuelve el DataFrame
  crudo de scores por residuo leido desde ``raw_output.csv``.
* Fase 3 (``extract_epitopes``): toda la logica de negocio (umbral,
  agrupamiento de residuos contiguos, longitud minima) corre en local sobre
  ese DataFrame, sin depender de los propios ficheros de prediccion que
  genera BepiPred (``Bcell_epitope_preds.fasta``, etc.), que se ignoran.

Columnas del CSV crudo confirmadas leyendo el codigo fuente oficial
(``bp3/bepipred3.py::create_csvfile``): ``Accession``, ``Residue`` y
``BepiPred-3.0 score`` (mas una columna de score lineal que no se usa aqui).

Rutas configurables: ninguna ruta al paquete de BepiPred se hardcodea en este
modulo. Todo se resuelve desde ``Settings`` (variables de entorno con
defaults razonables), ver ``src/config/settings.py``.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.config.settings import Settings
from src.engines.base_engine import BaseEngine
from src.utils.exceptions import BepiPredExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Columnas confirmadas empiricamente en el raw_output.csv real de BepiPred-3.0
# (ver ADR arriba). Si una version futura de BepiPred cambia el esquema,
# `_load_raw_scores` falla con un error que lista las columnas reales
# encontradas, en vez de un KeyError opaco.
ACCESSION_COLUMN = "Accession"
SCORE_COLUMN = "BepiPred-3.0 score"

# Nombre de columna de residuo confirmado en el codigo fuente oficial
# (bp3/bepipred3.py::create_csvfile). Se mantienen alternativas como
# fallback best-effort por si una version futura renombra la columna.
_RESIDUE_COLUMN_CANDIDATES = ("Residue", "residue", "Residues", "AA", "aa")


class BepiPredEngine(BaseEngine[str, pd.DataFrame]):
    """Ejecuta BepiPred-3.0 LOCALMENTE (subprocess) para cada FASTA de entrada.

    Uso exclusivo de inferencia cruda: no aplica threshold ni top_n aqui mas
    alla de lo estrictamente necesario para satisfacer el CLI (ver
    :func:`extract_epitopes` para el filtrado local de Fase 3).

    Validacion de entorno: antes de lanzar el subproceso se comprueba que el
    script ``bepipred3_CLI.py`` exista en ``Settings.BEPIPRED_HOME``. Si no
    esta presente, se lanza :class:`BepiPredExecutionError` con un mensaje
    claro (incluyendo el enlace de descarga oficial) en vez de dejar que el
    proceso colapse con un ``FileNotFoundError`` opaco.
    """

    def __init__(
        self,
        bepipred_home: Path = Settings.BEPIPRED_HOME,
        cli_script_name: str = Settings.BEPIPRED_CLI_SCRIPT_NAME,
        python_bin: str = Settings.BEPIPRED_PYTHON_BIN,
    ):
        self._bepipred_home = Path(bepipred_home)
        self._cli_script = self._bepipred_home / cli_script_name
        self._python_bin = python_bin

    @property
    def cli_script(self) -> Path:
        """Ruta resuelta al script ``bepipred3_CLI.py`` de la instalacion local."""
        return self._cli_script

    @property
    def python_bin(self) -> str:
        """Interprete de Python configurado para invocar el CLI de BepiPred."""
        return self._python_bin

    def _validate_installation(self) -> None:
        """Comprueba que el CLI local de BepiPred-3.0 este instalado y accesible.

        Raises:
            BepiPredExecutionError: Con un mensaje accionable (incluyendo el
                enlace de descarga oficial) si el script no existe. Es la
                unica forma en que esta clase reporta una instalacion
                faltante: nunca deja escapar un ``FileNotFoundError`` crudo.
        """
        if not self._cli_script.is_file():
            raise BepiPredExecutionError(
                "No se encontro la instalacion local de BepiPred-3.0 en "
                f"'{self._cli_script}'. Por restricciones de licencia academica, "
                "DTU Health Tech no permite redistribuir el codigo fuente: debes "
                "descargarlo manualmente desde "
                f"{Settings.BEPIPRED_DOWNLOAD_URL} , descomprimirlo en la raiz del "
                "proyecto como 'bepipred-3.0b.src/' (o apuntar la variable de "
                "entorno BEPIPRED_HOME a su ubicacion) y volver a intentarlo. "
                "Ver README.md - Seccion de Instalacion."
            )

    def run(self, items: Sequence[str], output_dir: Optional[Path] = None) -> List[pd.DataFrame]:
        """Corre BepiPred-3.0 localmente sobre cada ruta FASTA de ``items``.

        Args:
            items: Rutas locales a archivos FASTA a procesar.
            output_dir: Carpeta donde guardar los artefactos crudos generados
                por BepiPred-3.0 para cada FASTA de ``items`` (subcarpeta por
                ``stem``). Si es ``None``, usa ``Settings.BEPIPRED_OUTPUT_DIR``.

        Returns:
            Lista de DataFrames con los scores crudos por residuo (una fila
            por residuo, columnas ``Accession`` / ``BepiPred-3.0 score`` como
            minimo), en el mismo orden que ``items``, sin ningun filtrado.

        Raises:
            BepiPredExecutionError: Si la instalacion local no existe, el
                subproceso falla, excede el timeout, o la salida no tiene el
                formato esperado.
        """
        self._validate_installation()

        return [
            self._run_single(
                fasta_path,
                result_dir=(output_dir / Path(fasta_path).stem) if output_dir else None,
            )
            for fasta_path in items
        ]

    def _run_single(self, fasta_path: str, result_dir: Optional[Path] = None) -> pd.DataFrame:
        fasta = Path(fasta_path)
        if not fasta.is_file():
            raise FileNotFoundError(f"No se encontro el FASTA de entrada: {fasta}")

        result_dir = result_dir or (Settings.BEPIPRED_OUTPUT_DIR / fasta.stem)
        result_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._python_bin, str(self._cli_script.resolve()),
            "-i", str(fasta.resolve()),
            "-o", str(result_dir.resolve()),
            "-pred", Settings.BEPIPRED_PRED_MODE,
            "-t", str(Settings.BEPIPRED_THRESHOLD),
        ]

        logger.info("Ejecutando BepiPred-3.0 local para '%s': %s", fasta.name, " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                cwd=str(self._bepipred_home),
                check=True,
                capture_output=True,
                text=True,
                timeout=Settings.BEPIPRED_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as exc:
            raise BepiPredExecutionError(
                f"BepiPred-3.0 termino con exit code {exc.returncode} para '{fasta.name}'. "
                f"stderr: {(exc.stderr or '<vacio>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BepiPredExecutionError(
                f"BepiPred-3.0 excedio el tiempo limite de {Settings.BEPIPRED_TIMEOUT_SECONDS}s "
                f"para '{fasta.name}'. Aumenta BEPIPRED_TIMEOUT_SECONDS si la secuencia es muy larga "
                "o el hardware es lento (CPU vs GPU)."
            ) from exc

        csv_path = self._locate_raw_output(result_dir)
        return self._load_raw_scores(csv_path)

    @staticmethod
    def _locate_raw_output(result_dir: Path) -> Path:
        """Ubica el CSV de scores crudos dentro de los artefactos generados.

        Primero intenta el nombre confirmado empiricamente en el codigo
        fuente oficial (``Settings.BEPIPRED_RAW_OUTPUT_FILENAME``). Si una
        version futura de BepiPred lo renombra, cae a cualquier ``.csv``
        presente y lo loggea para que quede registro de la desviacion.
        """
        primary = list(result_dir.rglob(Settings.BEPIPRED_RAW_OUTPUT_FILENAME))
        if primary:
            return primary[0]

        candidates = sorted(result_dir.rglob("*.csv"))
        if not candidates:
            found = sorted(p.name for p in result_dir.rglob("*") if p.is_file())
            raise BepiPredExecutionError(
                f"No se encontro ningun CSV de salida en '{result_dir}'. "
                f"Archivos generados por BepiPred-3.0: {found or '<ninguno>'}."
            )

        logger.warning(
            "No se encontro '%s' en '%s'; usando '%s' como fallback. "
            "Verifica si la version de BepiPred renombro el archivo de salida.",
            Settings.BEPIPRED_RAW_OUTPUT_FILENAME,
            result_dir,
            candidates[0].name,
        )
        return candidates[0]

    @staticmethod
    def _load_raw_scores(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        missing = {ACCESSION_COLUMN, SCORE_COLUMN} - set(df.columns)
        if missing:
            raise BepiPredExecutionError(
                f"El CSV de salida '{csv_path}' no contiene las columnas confirmadas "
                f"{sorted(missing)}. Columnas encontradas: {list(df.columns)}."
            )
        return df


def _resolve_residue_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in _RESIDUE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    logger.warning(
        "No se encontro ninguna columna de residuo entre %s. Columnas disponibles: %s. "
        "Las regiones de epitopo se reportaran sin secuencia de aminoacidos.",
        _RESIDUE_COLUMN_CANDIDATES,
        list(df.columns),
    )
    return None


def extract_epitopes(
    raw_scores_df: pd.DataFrame,
    threshold: float = Settings.BEPIPRED_THRESHOLD,
    min_length: int = Settings.BEPIPRED_MIN_EPITOPE_LENGTH,
) -> pd.DataFrame:
    """Fase 3: agrupa residuos contiguos por encima de ``threshold`` en epitopos.

    Logica 100% local. La posicion de cada residuo se deriva del orden de las
    filas dentro de cada ``Accession`` (1-indexado), no de una columna de
    posicion.

    Args:
        raw_scores_df: DataFrame crudo devuelto por ``BepiPredEngine.run``.
        threshold: Score minimo (inclusive) para considerar un residuo parte
            de un epitopo candidato.
        min_length: Longitud minima (en residuos) de una region contigua para
            reportarse como epitopo.

    Returns:
        DataFrame con una fila por region de epitopo: ``accession``,
        ``start``, ``end``, ``length``, ``mean_score``, ``max_score`` y
        ``sequence`` (vacio si no se pudo resolver la columna de residuo).
    """
    missing = {ACCESSION_COLUMN, SCORE_COLUMN} - set(raw_scores_df.columns)
    if missing:
        raise BepiPredExecutionError(
            f"El DataFrame de entrada no contiene las columnas requeridas {sorted(missing)}. "
            f"Columnas encontradas: {list(raw_scores_df.columns)}."
        )

    residue_col = _resolve_residue_column(raw_scores_df)
    records = []

    for accession, group in raw_scores_df.groupby(ACCESSION_COLUMN, sort=False):
        group = group.reset_index(drop=True)
        above_threshold = group[SCORE_COLUMN] >= threshold
        block_id = (above_threshold != above_threshold.shift(fill_value=False)).cumsum()

        for _, block in group[above_threshold].groupby(block_id[above_threshold]):
            length = len(block)
            if length < min_length:
                continue

            sequence = "".join(block[residue_col].astype(str)) if residue_col else ""
            records.append(
                {
                    "accession": accession,
                    "start": int(block.index[0]) + 1,
                    "end": int(block.index[-1]) + 1,
                    "length": length,
                    "mean_score": float(block[SCORE_COLUMN].mean()),
                    "max_score": float(block[SCORE_COLUMN].max()),
                    "sequence": sequence,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=["accession", "start", "end", "length", "mean_score", "max_score", "sequence"],
    )


def print_epitope_table(epitopes_df: pd.DataFrame) -> None:
    """Imprime la tabla final de epitopos filtrados en consola."""
    if epitopes_df.empty:
        print("No se encontraron epitopos que superen el threshold y la longitud minima.")
        return

    header = f"{'accession':<28}{'start':>7}{'end':>7}{'len':>6}{'mean':>8}{'max':>8}  sequence"
    print(header)
    print("-" * len(header))
    for row in epitopes_df.itertuples(index=False):
        print(
            f"{row.accession:<28}{row.start:>7}{row.end:>7}{row.length:>6}"
            f"{row.mean_score:>8.4f}{row.max_score:>8.4f}  {row.sequence}"
        )
