"""Fase 5: Prediccion de presentacion celular (afinidad HLA / IC50) para peptidos.

Soporta dos motores intercambiables (patron Strategy, seleccionado via
``--inmuno`` en el orquestador):

* ``mhcflurry``: libreria Python pip-instalable, corre 100% en local con los
  modelos pre-entrenados descargados via ``mhcflurry-downloads fetch``.
* ``netmhcpan``: binario propietario de DTU Health Tech, requiere licencia e
  instalacion manual del usuario (no distribuible via pip/conda publico). Se
  invoca via ``subprocess`` si esta presente en el PATH.

Regla de negocio comun (Fase 5): un peptido se considera presentable si su
IC50 predicho es < 500 nM (``Settings.IC50_THRESHOLD``).
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import Settings
from src.utils.exceptions import ImmunogenicityExecutionError
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

_ALLELE_PATTERN = re.compile(r"^(HLA-[A-Z]+)(\d.*)$")

# MHCflurry solo soporta peptidos de longitud 8-15 aa (rango de union a MHC-I).
_MHCFLURRY_MIN_LEN = 8
_MHCFLURRY_MAX_LEN = 15


def normalize_allele(allele: str) -> str:
    """Normaliza el alelo HLA al formato con asterisco esperado por los motores.

    Ejemplo: ``"HLA-A02:01"`` -> ``"HLA-A*02:01"``. Si ya trae el asterisco,
    se devuelve tal cual (solo en mayusculas).

    Args:
        allele: Alelo HLA en formato libre.

    Returns:
        Alelo normalizado.
    """
    allele = allele.strip().upper()
    if "*" in allele:
        return allele
    match = _ALLELE_PATTERN.match(allele)
    if match:
        return f"{match.group(1)}*{match.group(2)}"
    return allele


def predict_mhcflurry(peptides: List[str], allele: str) -> pd.DataFrame:
    """Predice IC50 (nM) para cada peptido usando los modelos locales de MHCflurry.

    Args:
        peptides: Peptidos candidatos (longitud tipica 9-25 aa; los que caigan
            fuera de 8-15 aa se omiten con un warning, por limitacion del modelo).
        allele: Alelo HLA objetivo (se normaliza internamente).

    Returns:
        DataFrame con columnas ``sequence``, ``allele``, ``ic50_nM``, ``metodo``.

    Raises:
        ImmunogenicityExecutionError: Si mhcflurry no esta instalado, el
            alelo no esta soportado por los modelos locales, o falla la
            prediccion.
    """
    try:
        from mhcflurry import Class1AffinityPredictor
    except ImportError as exc:
        raise ImmunogenicityExecutionError(
            "mhcflurry no esta instalado. Ejecuta 'pip install mhcflurry' y luego "
            "'mhcflurry-downloads fetch' antes de usar --inmuno mhcflurry "
            "(ver README.md - Seccion de Instalacion)."
        ) from exc

    allele_norm = normalize_allele(allele)

    valid_peptides = [p for p in peptides if _MHCFLURRY_MIN_LEN <= len(p) <= _MHCFLURRY_MAX_LEN]
    skipped = len(peptides) - len(valid_peptides)
    if skipped:
        logger.warning(
            "%d peptido(s) fuera del rango %d-%d aa soportado por MHCflurry fueron omitidos.",
            skipped, _MHCFLURRY_MIN_LEN, _MHCFLURRY_MAX_LEN,
        )
    if not valid_peptides:
        return pd.DataFrame(columns=["sequence", "allele", "ic50_nM", "metodo"])

    try:
        predictor = Class1AffinityPredictor.load()
        if allele_norm not in predictor.supported_alleles:
            ejemplo = ", ".join(sorted(predictor.supported_alleles)[:5])
            raise ImmunogenicityExecutionError(
                f"El alelo '{allele_norm}' no esta soportado por los modelos locales de MHCflurry. "
                f"Ejemplos soportados: {ejemplo}, ..."
            )
        affinities = predictor.predict(peptides=valid_peptides, allele=allele_norm)
    except ImmunogenicityExecutionError:
        raise
    except Exception as exc:
        raise ImmunogenicityExecutionError(f"Fallo en la prediccion de MHCflurry: {exc}") from exc

    return pd.DataFrame(
        {
            "sequence": valid_peptides,
            "allele": allele_norm,
            "ic50_nM": affinities,
            "metodo": "mhcflurry",
        }
    )


def predict_netmhcpan(peptides: List[str], allele: str, output_dir: Path) -> pd.DataFrame:
    """Predice IC50 (nM) via el binario local 'netMHCpan' (licencia DTU Health Tech).

    Args:
        peptides: Peptidos candidatos a evaluar.
        allele: Alelo HLA objetivo (se normaliza internamente).
        output_dir: Carpeta donde persistir el archivo xls crudo devuelto por
            netMHCpan, para trazabilidad.

    Returns:
        DataFrame con columnas ``sequence``, ``allele``, ``ic50_nM``, ``metodo``.

    Raises:
        ImmunogenicityExecutionError: Si el binario no esta en el PATH, el
            proceso falla/timeout, o el formato de salida no es el esperado.
    """
    binary = shutil.which("netMHCpan")
    if binary is None:
        raise ImmunogenicityExecutionError(
            "El binario 'netMHCpan' no esta disponible en el PATH. NetMHCpan requiere "
            "descarga e instalacion local bajo licencia academica de DTU Health Tech "
            "(no distribuible via pip/conda). Usa '--inmuno mhcflurry' como alternativa, "
            "o instala NetMHCpan y agregalo al PATH (ver README.md - Seccion de Instalacion)."
        )

    allele_norm = normalize_allele(allele)
    output_dir.mkdir(parents=True, exist_ok=True)
    persisted_xls = output_dir / "netmhcpan_raw.xls"

    with tempfile.TemporaryDirectory(prefix="netmhcpan_") as tmp:
        tmp_dir = Path(tmp)
        pep_path = tmp_dir / "peptides.pep"
        xls_path = tmp_dir / "netmhcpan_output.xls"

        pep_path.write_text("\n".join(peptides) + "\n", encoding="utf-8")

        cmd = [binary, "-p", str(pep_path), "-a", allele_norm, "-BA", "-xls", "-xlsfile", str(xls_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            raise ImmunogenicityExecutionError(
                f"netMHCpan termino con exit code {exc.returncode}: {exc.stderr[:2000] or '<sin stderr>'}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ImmunogenicityExecutionError("netMHCpan excedio el tiempo limite de 600s.") from exc

        try:
            raw = pd.read_csv(xls_path, sep="\t", skiprows=1)
        except Exception as exc:
            raise ImmunogenicityExecutionError(
                f"No se pudo parsear la salida de netMHCpan en '{xls_path}': {exc}"
            ) from exc

        shutil.copyfile(xls_path, persisted_xls)

    ic50_col = next((c for c in raw.columns if "aff(nm)" in c.lower().replace(" ", "")), None)
    pep_col = next((c for c in raw.columns if c.lower() == "peptide"), None)
    if ic50_col is None or pep_col is None:
        raise ImmunogenicityExecutionError(
            "El formato de salida de netMHCpan no incluyo las columnas esperadas "
            f"('Peptide', 'Aff(nM)'). Columnas encontradas: {list(raw.columns)}."
        )

    return pd.DataFrame(
        {
            "sequence": raw[pep_col],
            "allele": allele_norm,
            "ic50_nM": raw[ic50_col],
            "metodo": "netmhcpan",
        }
    )


def evaluate_immunogenicity(
    peptides: List[str],
    allele: str,
    method: str,
    output_dir: Path,
    ic50_threshold: float = Settings.IC50_THRESHOLD,
) -> pd.DataFrame:
    """Punto de entrada unico de la Fase 5: despacha al motor elegido y aplica el veredicto.

    Args:
        peptides: Peptidos supervivientes de la Fase 4 (status == 'Segura').
        allele: Alelo HLA objetivo.
        method: ``"mhcflurry"`` o ``"netmhcpan"``.
        output_dir: Carpeta de salida para artefactos crudos (solo netMHCpan).
        ic50_threshold: Umbral de aprobacion (nM), exclusivo.

    Returns:
        DataFrame de resultados con columna adicional ``veredicto``
        (``"Aprobado"`` / ``"Rechazado"``).
    """
    if method == "mhcflurry":
        df = predict_mhcflurry(peptides, allele)
    elif method == "netmhcpan":
        df = predict_netmhcpan(peptides, allele, output_dir)
    else:
        raise ValueError(f"Metodo de inmunogenicidad desconocido: '{method}'.")

    if df.empty:
        return df

    df = df.copy()
    df["veredicto"] = df["ic50_nM"].apply(lambda ic50: "Aprobado" if ic50 < ic50_threshold else "Rechazado")
    return df
