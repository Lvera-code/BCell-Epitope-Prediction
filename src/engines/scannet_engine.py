"""Fase 2 (motor estructural 2/2, local) de prediccion de antigenicidad via ScanNet.

Mismo patron arquitectonico que ``discotope_engine.py``: wrapper puro de
``subprocess`` sobre el CLI oficial de ScanNet
(https://github.com/jertubiana/ScanNet), sin ninguna llamada de red en
tiempo de inferencia. ``TIn`` es una ruta a PDB de una sola cadena (ver
``StructureRecord.chain_pdb_path`` en ``src.utils.structure_parser`` -- mismo
motivo que DiscoTope-3.0: evitar ambiguedad sobre a que cadena corresponde
cada archivo de salida).

Runtime dual (``Settings.SCANNET_RUNTIME``):

* ``'venv'``: entorno aislado ``.venv-scannet`` (stack antiguo: Python
  3.6.12, TensorFlow/Keras -- mismo motivo que EpiDope para requerir un
  entorno dedicado), invocando ``predict_bindingsites.py`` directamente.
* ``'docker'`` (default): invoca la imagen oficial ``jertubiana/scannet`` via
  ``docker run``, montando el PDB de entrada y ``output_dir`` como volumenes.
  Evita tener que resolver el stack antiguo a mano.

ADR -- runtime Docker, validado empiricamente
------------------------------------------------
AMBOS runtimes fueron instalados y corridos contra un PDB real en esta tarea
(2026-07-20). Para Docker: ``docker pull jertubiana/scannet`` +
``docker inspect`` confirmaron que el ``WORKDIR`` de la imagen oficial es
efectivamente ``/ScanNet`` (default de ``Settings.SCANNET_DOCKER_WORKDIR``,
sin necesidad de ajuste), que tanto ``python`` como ``python3`` resuelven al
interprete correcto (3.6.12) dentro de la imagen, y que una corrida real
produjo resultados IDENTICOS byte a byte al runtime ``venv`` sobre el mismo
PDB. Si en el futuro la imagen oficial cambia de layout,
``_validate_installation`` fallara de forma clara (contenedor no arranca /
script no encontrado) en vez de silenciosamente producir un resultado
incorrecto.

Modo de prediccion: SIEMPRE ``--mode epitope --noMSA``. ``--noMSA`` es
deliberado y no configurable: el modo con MSA requiere una base de datos de
secuencias local (UniRef30) + HH-blits instalados, una dependencia pesada que
rompe el ADR de este pipeline ("100% local, sin infraestructura adicional
pesada"); ScanNet documenta el modo sin MSA como "less accurate, faster", un
trade-off aceptado aqui a cambio de mantenerse dentro de ese ADR.
"""

import glob
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.engines.base_engine import BaseEngine
from src.engines.epitope_mapping import extract_epitope_regions
from src.engines.epitope_mapping import print_epitope_table as _print_epitope_table
from src.utils.exceptions import ScanNetExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

ACCESSION_COLUMN = "Accession"
RESIDUE_COLUMN = "Residue"
SCORE_COLUMN = "ScanNet score"

# Columnas confirmadas leyendo el codigo fuente oficial
# (predict_bindingsites.py::write_predictions): 'Model,Chain,Residue Index,
# Sequence,Binding site probability'. Solo se usan 'Sequence' y 'Binding site
# probability' aqui; 'Residue Index'/'Chain' se ignoran a favor del orden de
# filas (mismo criterio posicional que el resto del pipeline), y 'Model' se
# ignora (siempre el mismo modelo epitope_noMSA para esta clase).
_RAW_RESIDUE_COLUMN = "Sequence"
_RAW_SCORE_COLUMN = "Binding site probability"

_VALID_RUNTIMES = ("venv", "docker")


class ScanNetEngine(BaseEngine[str, pd.DataFrame]):
    """Ejecuta ScanNet LOCALMENTE (subprocess venv o Docker) para cada PDB de una sola cadena."""

    def __init__(
        self,
        runtime: str = Settings.SCANNET_RUNTIME,
        install_path: Path = Settings.SCANNET_INSTALL_PATH,
        python_bin: str = Settings.SCANNET_PYTHON_BIN,
        docker_image: str = Settings.SCANNET_DOCKER_IMAGE,
        docker_workdir: str = Settings.SCANNET_DOCKER_WORKDIR,
    ):
        if runtime not in _VALID_RUNTIMES:
            raise ScanNetExecutionError(
                f"SCANNET_RUNTIME='{runtime}' no reconocido (valores validos: {_VALID_RUNTIMES})."
            )
        self._runtime = runtime
        self._install_path = Path(install_path)
        self._python_bin = python_bin
        self._docker_image = docker_image
        self._docker_workdir = docker_workdir

    def _validate_installation(self) -> None:
        """Comprueba que el runtime configurado (venv o Docker) este listo para usarse.

        Raises:
            ScanNetExecutionError: Con instrucciones de instalacion completas,
                distintas segun el runtime configurado.
        """
        if self._runtime == "venv":
            python_path = Path(self._python_bin)
            script_path = self._install_path / "predict_bindingsites.py"
            if not python_path.is_file():
                raise ScanNetExecutionError(
                    f"No se encontro el interprete Python de ScanNet en '{python_path}'. "
                    f"ScanNet requiere Python 3.6.12 exacto ({Settings.SCANNET_DOWNLOAD_URL}), que "
                    "ningun sistema moderno trae preinstalado: la forma reproducible de obtenerlo "
                    "es con conda (no un venv comun, que necesita partir de un interprete ya "
                    "instalado en el sistema):\n"
                    f"  git clone {Settings.SCANNET_DOWNLOAD_URL} {self._install_path}\n"
                    "  conda create -n scannet_env python=3.6.12 -y\n"
                    f"  conda run -n scannet_env pip install -r {self._install_path}/requirements.txt\n"
                    "Y apunta SCANNET_PYTHON_BIN al 'python' de ese entorno conda (o de tu propio "
                    "venv/virtualenv si conseguiste un Python 3.6.12 por otra via). "
                    "Ver README.md - Seccion de Instalacion."
                )
            if not script_path.is_file():
                raise ScanNetExecutionError(
                    f"No se encontro '{script_path}'. Verifica que SCANNET_INSTALL_PATH "
                    f"('{self._install_path}') apunte al clon de {Settings.SCANNET_DOWNLOAD_URL}."
                )
            return

        # runtime == 'docker'
        try:
            subprocess.run(
                ["docker", "info"], check=True, capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError as exc:
            raise ScanNetExecutionError(
                "El binario 'docker' no esta instalado o no esta en PATH. Instala Docker o "
                "cambia SCANNET_RUNTIME a 'venv'."
            ) from exc
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ScanNetExecutionError(
                "El daemon de Docker no responde ('docker info' fallo). Verifica que Docker "
                "este corriendo, o cambia SCANNET_RUNTIME a 'venv'."
            ) from exc

        images = subprocess.run(
            ["docker", "images", "-q", self._docker_image], capture_output=True, text=True, timeout=30,
        )
        if not images.stdout.strip():
            raise ScanNetExecutionError(
                f"La imagen Docker '{self._docker_image}' no esta descargada localmente. "
                f"Descargala con:\n  docker pull {self._docker_image}"
            )

    def run(self, items: Sequence[str], output_dir: Optional[Path] = None) -> List[pd.DataFrame]:
        """Corre ScanNet localmente (runtime venv o Docker) sobre cada PDB de una sola cadena.

        Args:
            items: Rutas locales a PDB de una sola cadena (ver
                ``StructureRecord.chain_pdb_path``).
            output_dir: Carpeta donde guardar los artefactos crudos generados
                por ScanNet para cada PDB de ``items`` (subcarpeta por
                ``stem``). Si es ``None``, usa ``Settings.SCANNET_OUTPUT_DIR``.

        Returns:
            Lista de DataFrames con los scores crudos por residuo (columnas
            ``Accession``/``Residue``/``ScanNet score``), en el mismo orden
            que ``items``, sin ningun filtrado.

        Raises:
            ScanNetExecutionError: Si el runtime configurado no esta listo, el
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

        result_dir = result_dir or (Settings.SCANNET_OUTPUT_DIR / pdb.stem)
        result_dir.mkdir(parents=True, exist_ok=True)
        accession = pdb.stem

        cmd = self._build_command(pdb, result_dir, accession)
        # ScanNet resuelve 'utilities/paths.py::model_folder' (y el resto de
        # rutas del propio repo) como relativas al directorio de trabajo del
        # proceso ('library_folder' = '' en ese archivo), NO relativas a la
        # ubicacion del script: sin fijar cwd al repo clonado, buscaria
        # 'models/' en el cwd de pipeline.py y fallaria con un error opaco de
        # pesos no encontrados. Solo aplica al runtime 'venv': en 'docker' el
        # WORKDIR de la imagen ya cumple ese rol (ver '-w' en _build_command).
        run_cwd = str(self._install_path) if self._runtime == "venv" else None

        logger.info("Ejecutando ScanNet local (runtime=%s) para '%s': %s", self._runtime, pdb.name, " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=Settings.SCANNET_TIMEOUT_SECONDS,
                cwd=run_cwd,
            )
        except subprocess.CalledProcessError as exc:
            raise ScanNetExecutionError(
                f"ScanNet termino con exit code {exc.returncode} para '{pdb.name}' (runtime="
                f"{self._runtime}). stderr: {(exc.stderr or '<vacio>')[:2000]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ScanNetExecutionError(
                f"ScanNet excedio el tiempo limite de {Settings.SCANNET_TIMEOUT_SECONDS}s "
                f"para '{pdb.name}'. Aumenta SCANNET_TIMEOUT_SECONDS si la estructura es muy "
                "grande o el hardware es lento (CPU vs GPU)."
            ) from exc

        return self._load_raw_scores(result_dir, accession=accession)

    def _build_command(self, pdb: Path, result_dir: Path, accession: str) -> List[str]:
        base_flags = ["--mode", "epitope", "--noMSA", "--pdb", "--name", accession]

        if self._runtime == "venv":
            # 'predict_bindingsites.py' SIN prefijo de install_path: _run_single
            # ya fija cwd=self._install_path para este runtime (ver ADR del
            # modulo sobre utilities/paths.py), asi que anteponer install_path
            # aqui tambien duplicaria el directorio (install_path/install_path/
            # predict_bindingsites.py) -- confirmado empiricamente, exit code 2
            # 'No such file or directory'. Mismo criterio que el runtime
            # 'docker', que ya usa el nombre pelado porque el WORKDIR del
            # contenedor cumple el mismo rol.
            return [
                self._python_bin,
                "predict_bindingsites.py",
                str(pdb.resolve()),
                "--predictions_folder", str(result_dir.resolve()),
            ] + base_flags

        # runtime == 'docker': monta el PDB (solo lectura) y la carpeta de resultados.
        container_pdb = f"/input/{pdb.name}"
        container_out = "/predictions"
        return [
            "docker", "run", "--rm",
            "-v", f"{pdb.resolve().parent}:/input:ro",
            "-v", f"{result_dir.resolve()}:{container_out}",
            "-w", self._docker_workdir,
            self._docker_image,
            "python", "predict_bindingsites.py",
            container_pdb,
            "--predictions_folder", container_out,
        ] + base_flags

    @staticmethod
    def _load_raw_scores(result_dir: Path, accession: str) -> pd.DataFrame:
        """Localiza y carga el (unico) CSV 'predictions_*.csv' que escribe ScanNet.

        Se busca por glob recursivo en vez de reconstruir la ruta exacta de
        ScanNet (``<name>_single_ScanNet_epitope_noMSA/predictions_<name>.csv``,
        confirmada leyendo su codigo fuente oficial): ese nombre de subcarpeta
        depende de detalles internos (sufijos por modo/version del modelo)
        que pueden cambiar entre versiones, igual criterio defensivo que
        ``discotope_engine.py``/``epidope_engine.py``.
        """
        csv_files = sorted(Path(p) for p in glob.glob(str(result_dir / "**" / "predictions_*.csv"), recursive=True))
        if not csv_files:
            found = sorted(p.name for p in result_dir.rglob("*") if p.is_file())
            raise ScanNetExecutionError(
                f"No se encontro ningun CSV de salida 'predictions_*.csv' en '{result_dir}'. "
                f"Archivos generados por ScanNet: {found or '<ninguno>'}."
            )
        if len(csv_files) > 1:
            raise ScanNetExecutionError(
                f"Se encontro mas de un CSV de salida en '{result_dir}' ({[c.name for c in csv_files]}): "
                "se esperaba exactamente uno, ya que el PDB de entrada deberia contener una sola "
                "cadena (ver StructureRecord.chain_pdb_path)."
            )

        df = pd.read_csv(csv_files[0])
        missing = {_RAW_RESIDUE_COLUMN, _RAW_SCORE_COLUMN} - set(df.columns)
        if missing:
            raise ScanNetExecutionError(
                f"El CSV de salida '{csv_files[0]}' no contiene las columnas esperadas "
                f"{sorted(missing)}. Columnas encontradas: {list(df.columns)}."
            )

        df = df.rename(columns={_RAW_RESIDUE_COLUMN: RESIDUE_COLUMN, _RAW_SCORE_COLUMN: SCORE_COLUMN})
        df.insert(0, ACCESSION_COLUMN, accession)
        return df[[ACCESSION_COLUMN, RESIDUE_COLUMN, SCORE_COLUMN]]


def extract_epitopes(
    raw_scores_df: pd.DataFrame,
    threshold: Optional[float] = None,
    threshold_percentile: float = Settings.SCANNET_THRESHOLD_PERCENTILE,
    min_length: int = Settings.SCANNET_MIN_EPITOPE_LENGTH,
    window_size: int = Settings.SCANNET_WINDOW_SIZE,
    max_gap_residues: int = Settings.SCANNET_MAX_GAP_RESIDUES,
) -> pd.DataFrame:
    """Fase 3 (ScanNet): mapea regiones de epitopo con la misma ventana deslizante.

    Logica identica al resto de motores (ver
    ``src.engines.epitope_mapping.extract_epitope_regions``), aplicada sobre
    los scores crudos de ScanNet, con una diferencia clave: ScanNet no tiene
    un umbral absoluto publicado por sus autores (a diferencia de
    DiscoTope-3.0, ver ADR en ``Settings.SCANNET_THRESHOLD_PERCENTILE``), asi
    que por defecto (``threshold=None``) el umbral se calcula de forma
    ADAPTATIVA, por separado para CADA accession, como el percentil
    ``threshold_percentile`` de los scores de esa cadena especifica -- el
    mismo principio de normalizacion por antigeno que DiscoTope-3.0 aplica
    internamente con 'calibrated_score', aqui aplicado de este lado porque
    ScanNet no lo hace.

    Args:
        raw_scores_df: Scores crudos por residuo (una o mas accessions).
        threshold: Umbral ABSOLUTO fijo (0-1), igual en escala para todas
            las accessions. Si es ``None`` (default), se usa el umbral
            adaptativo por percentil descrito arriba en su lugar.
        threshold_percentile: Percentil (0-100) de los scores de cada
            accession usado como umbral adaptativo cuando ``threshold`` es
            ``None`` (``Settings.SCANNET_THRESHOLD_PERCENTILE`` = 90 por
            defecto: se consideran "altos" los residuos en el 10% superior
            de la cadena).

    Nota biologica: igual que DiscoTope-3.0, ScanNet puntua epitopos
    CONFORMACIONALES (parches 3D potencialmente discontinuos en la secuencia
    lineal); colapsar a regiones contiguas via ventana deslizante es una
    simplificacion deliberada para mantener Fase 4/5 operando sobre peptidos
    lineales sintetizables.

    Returns:
        DataFrame con una fila por region de epitopo: ``accession``,
        ``start``, ``end``, ``length``, ``mean_score``, ``max_score`` y
        ``sequence``.
    """
    missing = {ACCESSION_COLUMN, SCORE_COLUMN} - set(raw_scores_df.columns)
    if missing:
        raise ScanNetExecutionError(
            f"El DataFrame de entrada no contiene las columnas requeridas {sorted(missing)}. "
            f"Columnas encontradas: {list(raw_scores_df.columns)}."
        )

    if threshold is not None:
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

    frames = []
    for accession, group in raw_scores_df.groupby(ACCESSION_COLUMN, sort=False):
        adaptive_threshold = float(np.percentile(group[SCORE_COLUMN], threshold_percentile))
        logger.info(
            "Accession '%s': umbral adaptativo ScanNet (percentil %.1f de %d residuo(s)) = %.4f",
            accession, threshold_percentile, len(group), adaptive_threshold,
        )
        frames.append(
            extract_epitope_regions(
                group,
                accession_col=ACCESSION_COLUMN,
                score_col=SCORE_COLUMN,
                residue_col_candidates=(RESIDUE_COLUMN,),
                threshold=adaptive_threshold,
                min_length=min_length,
                window_size=window_size,
                max_gap_residues=max_gap_residues,
            )
        )

    if not frames:
        # raw_scores_df vacio (sin ninguna accession): ningun umbral aplica,
        # se delega la forma de la tabla vacia resultante (mismas columnas)
        # a extract_epitope_regions sin necesidad de un threshold real.
        return extract_epitope_regions(
            raw_scores_df,
            accession_col=ACCESSION_COLUMN,
            score_col=SCORE_COLUMN,
            residue_col_candidates=(RESIDUE_COLUMN,),
            threshold=0.0,
            min_length=min_length,
            window_size=window_size,
            max_gap_residues=max_gap_residues,
        )
    return pd.concat(frames, ignore_index=True)


def print_epitope_table(epitopes_df: pd.DataFrame) -> None:
    """Imprime la tabla final de epitopos filtrados (ScanNet) en consola."""
    _print_epitope_table(
        epitopes_df,
        empty_message="No se encontraron epitopos (ScanNet) que superen el threshold y la longitud minima.",
    )
