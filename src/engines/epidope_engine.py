"""Fase 2 (segundo motor, local) + Fase 3 de prediccion de antigenicidad via EpiDope.

Mismo patron arquitectonico que ``bepipred_engine.py`` (ver ADR alli): esta
clase es un wrapper puro de ``subprocess`` sobre el CLI oficial de EpiDope
(https://github.com/rnajena/EpiDope, fork activamente mantenido, licencia
MIT), sin ninguna llamada de red en tiempo de inferencia: los pesos del
modelo y los embeddings ELMo vienen empaquetados en el propio repo
(``epidope/epidope_weights``, ``epidope/elmo_settings``).

Diferencia clave frente a BepiPred: EpiDope es codigo abierto e instalable
via conda sin solicitud academica, pero fija un stack de dependencias muy
antiguo (Python 3.6, TensorFlow 1.13, Keras 2.3, PyTorch 0.4, AllenNLP 0.7.2
para embeddings ELMo) incompatible con el resto del pipeline. Requiere por
tanto un entorno conda dedicado creado EXACTAMENTE con el ``epidope.yml``
del repo oficial (``EPIDOPE_CONDA_PREFIX``, por defecto ``.conda-epidope/``
en la raiz del proyecto, mismo rol que ``.venv-bepipred``), invocado via
``conda run`` (o directamente via ``EPIDOPE_BIN`` si se prefiere apuntar a un
ejecutable ya resuelto, saltandose conda).

Formato de salida de EpiDope confirmado leyendo su codigo fuente oficial
(``epidope/epidope2.py::output_results``): por cada secuencia de entrada
escribe un CSV propio en ``<outdir>/epidope/<accession>.csv`` con columnas
``position``, ``aminoacid`` y ``score`` (scores crudos por residuo, 1
indexados). Esta clase concatena esos CSV por-accession en un unico
DataFrame crudo, IGNORANDO los ficheros de resumen que el propio EpiDope
genera (``predicted_epitopes.csv``, ``predicted_epitopes_sliced.faa``,
plots HTML): la Fase 3 (``extract_epitopes``) rehace el mapeo de regiones
en local con la misma ventana deslizante que ``bepipred_engine.py``,
reutilizada desde ``src.engines.epitope_mapping``.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.config.settings import Settings
from src.engines.base_engine import BaseEngine
from src.engines.epitope_mapping import extract_epitope_regions
from src.engines.epitope_mapping import print_epitope_table as _print_epitope_table
from src.utils.exceptions import EpidopeExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Nombres de columna propios de este motor (deliberadamente distintos de los
# de BepiPred: los scores de ambos motores NO son comparables en escala, ver
# Settings.EPIDOPE_THRESHOLD).
ACCESSION_COLUMN = "Accession"
RESIDUE_COLUMN = "Residue"
SCORE_COLUMN = "EpiDope score"

# Columnas confirmadas empiricamente en <outdir>/epidope/<accession>.csv
# (ver epidope/epidope2.py::output_results, linea `outfile6.write('position\\taminoacid\\tscore\\n')`).
_RAW_COLUMNS = ("position", "aminoacid", "score")


class EpidopeEngine(BaseEngine[str, pd.DataFrame]):
    """Ejecuta EpiDope LOCALMENTE (subprocess via conda run) para cada FASTA de entrada.

    Uso exclusivo de inferencia cruda: no aplica threshold de mapeo de
    regiones aqui (EpiDope si aplica su propio ``-t`` internamente para sus
    ficheros de resumen, que este motor ignora) mas alla de lo estrictamente
    necesario para satisfacer el CLI. Ver :func:`extract_epitopes` para el
    filtrado local de Fase 3.
    """

    def __init__(
        self,
        conda_prefix: Path = Settings.EPIDOPE_CONDA_PREFIX,
        conda_env_name: str = Settings.EPIDOPE_CONDA_ENV,
        epidope_bin: str = Settings.EPIDOPE_BIN,
    ):
        self._conda_prefix = Path(conda_prefix)
        self._conda_env_name = conda_env_name
        self._epidope_bin = epidope_bin

    def _resolved_bin_path(self) -> Optional[Path]:
        """Ruta resuelta al ejecutable ``epidope``, solo cuando es verificable de antemano.

        Si se usa ``-n <env>`` (entorno por nombre, no por prefijo de ruta),
        no hay una ruta local que comprobar antes de invocar ``conda run``:
        en ese caso devuelve ``None`` y la validacion se delega al propio
        subproceso.
        """
        if self._epidope_bin:
            return Path(self._epidope_bin)
        if self._conda_env_name:
            return None
        return self._conda_prefix / "bin" / "epidope"

    def _validate_installation(self) -> None:
        """Comprueba que el entorno local de EpiDope este instalado y accesible.

        Raises:
            EpidopeExecutionError: Con instrucciones de instalacion completas
                (EpiDope no requiere licencia academica, a diferencia de
                BepiPred/NetMHCIIpan) si no se encuentra el ejecutable.
        """
        bin_path = self._resolved_bin_path()
        if bin_path is not None and not bin_path.is_file():
            raise EpidopeExecutionError(
                f"No se encontro la instalacion local de EpiDope en '{bin_path}'. "
                "A diferencia de BepiPred-3.0 y NetMHCIIpan-4.3, EpiDope es codigo "
                f"abierto (licencia MIT, {Settings.EPIDOPE_DOWNLOAD_URL}) y no requiere "
                "solicitud academica: instalalo en un entorno conda dedicado con el "
                "'epidope.yml' oficial del repo (NO instales los paquetes a mano: la "
                "resolucion de ese stack es fragil y version por version puede resultar "
                "en un entorno inconsistente):\n"
                f"  git clone {Settings.EPIDOPE_DOWNLOAD_URL}.git /tmp/EpiDope\n"
                f"  conda env create -f /tmp/EpiDope/epidope.yml -p {self._conda_prefix}\n"
                "Ver README.md - Seccion de Instalacion. Si ya lo instalaste en otra "
                "ubicacion, apunta EPIDOPE_CONDA_PREFIX (por prefijo de ruta), "
                "EPIDOPE_CONDA_ENV (por nombre de entorno) o EPIDOPE_BIN (ejecutable "
                "directo) a esa instalacion."
            )

    def _build_command(self, extra_args: List[str]) -> List[str]:
        """Resuelve el comando a ejecutar, evitando ``conda run`` siempre que sea posible.

        ``conda run --no-capture-output`` demostro ser poco fiable con
        stdout/stderr capturados por pipe (``subprocess.run(capture_output=True)``,
        nuestro caso): EpiDope generaba correctamente todos sus artefactos de
        salida pero ``conda run`` reportaba, de forma intermitente, un exit
        code distinto de cero (fallo espurio, aparentemente una carrera entre
        el wrapper asincrono de ``conda run`` y los procesos hijos que EpiDope
        lanza via ``multiprocessing`` en su fase de graficado). El ejecutable
        ``epidope`` que instala conda en ``<prefix>/bin/epidope`` es un shim
        autocontenido (shebang apuntando al python del propio entorno): se
        puede invocar directamente sin pasar por ``conda run`` en absoluto,
        que es lo que hacemos por defecto cuando se resuelve por prefijo de
        ruta. Solo se recurre a ``conda run -n <env>`` cuando se configura un
        entorno por NOMBRE (``EPIDOPE_CONDA_ENV``), caso en el que no hay una
        ruta de archivo resoluble de antemano.
        """
        if self._epidope_bin:
            return [self._epidope_bin] + extra_args
        if self._conda_env_name:
            return ["conda", "run", "--no-capture-output", "-n", self._conda_env_name, "epidope"] + extra_args
        return [str(self._conda_prefix.resolve() / "bin" / "epidope")] + extra_args

    def run(self, items: Sequence[str], output_dir: Optional[Path] = None) -> List[pd.DataFrame]:
        """Corre EpiDope localmente sobre cada ruta FASTA de ``items``.

        Args:
            items: Rutas locales a archivos FASTA a procesar.
            output_dir: Carpeta donde guardar los artefactos crudos generados
                por EpiDope para cada FASTA de ``items`` (subcarpeta por
                ``stem``). Si es ``None``, usa ``Settings.EPIDOPE_OUTPUT_DIR``.

        Returns:
            Lista de DataFrames con los scores crudos por residuo (una fila
            por residuo, columnas ``Accession`` / ``Residue`` / ``EpiDope
            score``), en el mismo orden que ``items``, sin ningun filtrado.

        Raises:
            EpidopeExecutionError: Si la instalacion local no existe, el
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

        result_dir = result_dir or (Settings.EPIDOPE_OUTPUT_DIR / fasta.stem)
        result_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(
            ["-i", str(fasta.resolve()), "-o", str(result_dir.resolve()), "-t", str(Settings.EPIDOPE_THRESHOLD)]
        )

        logger.info("Ejecutando EpiDope local para '%s': %s", fasta.name, " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=Settings.EPIDOPE_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as exc:
            raise EpidopeExecutionError(
                f"EpiDope termino con exit code {exc.returncode} para '{fasta.name}'. "
                f"stderr: {(exc.stderr or '<vacio>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EpidopeExecutionError(
                f"EpiDope excedio el tiempo limite de {Settings.EPIDOPE_TIMEOUT_SECONDS}s "
                f"para '{fasta.name}'. Aumenta EPIDOPE_TIMEOUT_SECONDS si la secuencia es muy "
                "larga o el hardware es lento (CPU vs GPU)."
            ) from exc

        return self._load_raw_scores(result_dir)

    @staticmethod
    def _load_raw_scores(result_dir: Path) -> pd.DataFrame:
        """Concatena los CSV por-accession que escribe EpiDope en ``<result_dir>/epidope/``."""
        per_gene_dir = result_dir / "epidope"
        csv_files = sorted(per_gene_dir.glob("*.csv"))
        if not csv_files:
            found = sorted(p.name for p in result_dir.rglob("*") if p.is_file())
            raise EpidopeExecutionError(
                f"No se encontro ningun CSV de scores por-accession en '{per_gene_dir}'. "
                f"Archivos generados por EpiDope: {found or '<ninguno>'}."
            )

        frames = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path, sep="\t")
            missing = set(_RAW_COLUMNS) - set(df.columns)
            if missing:
                raise EpidopeExecutionError(
                    f"El CSV de salida '{csv_path}' no contiene las columnas confirmadas "
                    f"{sorted(missing)}. Columnas encontradas: {list(df.columns)}."
                )
            df = df.sort_values("position").reset_index(drop=True)
            df.insert(0, ACCESSION_COLUMN, csv_path.stem)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.rename(columns={"aminoacid": RESIDUE_COLUMN, "score": SCORE_COLUMN})
        return combined[[ACCESSION_COLUMN, RESIDUE_COLUMN, SCORE_COLUMN]]


def extract_epitopes(
    raw_scores_df: pd.DataFrame,
    threshold: float = Settings.EPIDOPE_THRESHOLD,
    min_length: int = Settings.EPIDOPE_MIN_EPITOPE_LENGTH,
    window_size: int = Settings.EPIDOPE_WINDOW_SIZE,
    max_gap_residues: int = Settings.EPIDOPE_MAX_GAP_RESIDUES,
) -> pd.DataFrame:
    """Fase 3 (EpiDope): mapea regiones de epitopo con la misma ventana deslizante que BepiPred.

    Logica identica a ``bepipred_engine.extract_epitopes`` (ver
    ``src.engines.epitope_mapping.extract_epitope_regions`` para el
    algoritmo compartido), aplicada sobre los scores crudos de EpiDope. El
    umbral por defecto (``Settings.EPIDOPE_THRESHOLD`` = 0.818) es el propio
    de EpiDope y NO es comparable en escala al de BepiPred (0.1512): cada
    motor conserva su propio umbral y sus propios parametros de ventana.

    Returns:
        DataFrame con una fila por region de epitopo: ``accession``,
        ``start``, ``end``, ``length``, ``mean_score``, ``max_score`` y
        ``sequence``.
    """
    missing = {ACCESSION_COLUMN, SCORE_COLUMN} - set(raw_scores_df.columns)
    if missing:
        raise EpidopeExecutionError(
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
    """Imprime la tabla final de epitopos filtrados (EpiDope) en consola."""
    _print_epitope_table(
        epitopes_df,
        empty_message="No se encontraron epitopos (EpiDope) que superen el threshold y la longitud minima.",
    )
