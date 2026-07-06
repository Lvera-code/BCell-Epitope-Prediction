"""Curacion de un dataset balanceado de calibracion IEDB (positivo) / housekeeping (negativo).

Descarga un panel representativo de epitopos lineales de celulas B validados
experimentalmente en IEDB (clase positiva) y genera fragmentos peptidicos de
control endogeno a partir de un panel curado de proteinas intracelulares
humanas "housekeeping" (clase negativa: enzimas metabolicas, citoesqueleto,
proteinas ribosomales, chaperonas, histonas -- ninguna de superficie celular ni
secretada, por lo que su probabilidad de constituir un epitopo de anticuerpo
in vivo es minima).

El split Train/Validation/Test (80/10/10) se realiza agrupando cada peptido
por su **proteina de origen** (``source_group``): todos los fragmentos que
provienen del mismo antigeno IEDB o de la misma proteina housekeeping caen en
un unico split. Esto evita fuga de datos (data leakage) por fragmentos
solapados de la misma region, un error metodologico comun en benchmarks de
prediccion de epitopos.
"""

import argparse
import csv
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from src.config.settings import Settings
from src.models import SequenceRecord
from src.utils.exceptions import DatasetPrepError
from src.utils.fasta_parser import CANONICAL_AA
from src.utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Panel curado de proteinas humanas "housekeeping" intracelulares (enzimas
# metabolicas centrales, citoesqueleto, ribosoma, chaperonas, histonas,
# proteasoma). Ninguna es secretada ni de superficie celular: se usan como
# control negativo de antigenicidad, en la misma linea metodologica que el
# conjunto de "no-antigenos" de VaxiJen. Accesiones Swiss-Prot (reviewed)
# verificadas contra UniProt REST.
HOUSEKEEPING_ACCESSIONS: Tuple[str, ...] = (
    "P04406",  # GAPDH
    "P60709",  # ACTB
    "P68371",  # TUBB4B
    "P68104",  # EEF1A1
    "P11142",  # HSPA8
    "P07900",  # HSP90AA1
    "P62937",  # PPIA
    "P00558",  # PGK1
    "P00338",  # LDHA
    "P06733",  # ENO1
    "P14618",  # PKM
    "P04075",  # ALDOA
    "P60174",  # TPI1
    "P08237",  # PFKM
    "P06744",  # GPI
    "O75390",  # CS
    "P40926",  # MDH2
    "P31040",  # SDHA
    "O75874",  # IDH1
    "Q99798",  # ACO2
    "P07954",  # FH
    "P08670",  # VIM
    "P40429",  # RPL13A
    "P05388",  # RPLP0
    "P60842",  # EIF4A1
    "P0DP23",  # CALM1
    "P20226",  # TBP
    "P24928",  # POLR2A
    "P12004",  # PCNA
    "P11388",  # TOP2A
    "P62805",  # H4C1 (Histona H4)
    "P0C0S5",  # H2AZ1
    "P28072",  # PSMB6
    "P0CG48",  # UBC
    "P55072",  # VCP
    "P00441",  # SOD1
    "P04040",  # CAT
    "Q06830",  # PRDX1
    "P10599",  # TXN
    "P00374",  # DHFR
    "P04818",  # TYMS
    "P23921",  # RRM1
    "P63104",  # YWHAZ
    "P62826",  # RAN
    "P17174",  # GOT1
    "P00367",  # GLUD1
    "P00966",  # ASS1
    "P00492",  # HPRT1
)

SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")


@dataclass(frozen=True)
class LabeledPeptide:
    """Un peptido etiquetado con su grupo de origen, para split sin fuga de datos.

    Attributes:
        record: Secuencia e identificador del peptido.
        label: ``1`` si es un epitopo positivo (IEDB), ``0`` si es un
            fragmento negativo de control (housekeeping).
        source_group: Identificador de la proteina/antigeno de origen. Todos
            los peptidos con el mismo ``source_group`` se asignan al mismo
            split (train/val/test).
    """

    record: SequenceRecord
    label: int
    source_group: str


def _http_get_json(url: str, params: dict) -> object:
    """Ejecuta un GET con reintentos y backoff exponencial, devolviendo JSON.

    Args:
        url: URL completa del endpoint.
        params: Parametros de query string.

    Returns:
        Cuerpo de la respuesta ya deserializado desde JSON.

    Raises:
        DatasetPrepError: Si todos los reintentos fallan.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, Settings.HTTP_MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=Settings.HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "Intento %d/%d fallido contra '%s': %s", attempt, Settings.HTTP_MAX_RETRIES, url, exc
            )
            time.sleep(min(2**attempt, 10))

    raise DatasetPrepError(f"Fallo irrecuperable consultando '{url}' tras reintentos: {last_error}")


def fetch_iedb_positive_peptides(
    target_count: int, min_length: int, max_length: int, seed: int
) -> List[LabeledPeptide]:
    """Descarga epitopos lineales de celulas B positivos desde la API de IEDB.

    Args:
        target_count: Numero deseado de peptidos positivos unicos.
        min_length: Longitud minima de peptido a incluir.
        max_length: Longitud maxima de peptido a incluir.
        seed: Semilla para el barajado deterministico de la seleccion final.

    Returns:
        Lista de :class:`LabeledPeptide` con ``label=1``, de tamano hasta
        ``target_count``.

    Raises:
        DatasetPrepError: Si la API no devuelve ningun peptido valido.
    """
    endpoint = f"{Settings.IEDB_API_BASE}/bcell_search"
    page_size = 1000
    offset = 0
    max_offset = 30000  # Limite duro de paginas para evitar bucles indefinidos.
    collected: Dict[str, LabeledPeptide] = {}

    logger.info(
        "Descargando epitopos positivos IEDB (longitud [%d, %d], objetivo=%d)...",
        min_length,
        max_length,
        target_count,
    )

    while offset < max_offset and len(collected) < target_count * 2:
        params = {
            "qualitative_measure": "eq.Positive",
            "structure_type": "eq.Linear peptide",
            "linear_sequence_length": [f"gte.{min_length}", f"lte.{max_length}"],
            "select": "structure_id,linear_sequence,curated_source_antigen",
            "order": "structure_id.asc",
            "limit": str(page_size),
            "offset": str(offset),
        }
        page = _http_get_json(endpoint, params)
        if not page:
            logger.info("La API de IEDB no devolvio mas resultados en offset=%d.", offset)
            break

        for entry in page:
            sequence = str(entry.get("linear_sequence", "")).upper().strip()
            if not sequence or not (min_length <= len(sequence) <= max_length):
                continue
            if set(sequence) - CANONICAL_AA:
                continue
            if sequence in collected:
                continue

            source_antigen = entry.get("curated_source_antigen") or {}
            accession = (
                source_antigen.get("accession")
                if isinstance(source_antigen, dict)
                else None
            )
            group = accession or f"IEDB_UNGROUPED_{entry.get('structure_id')}"

            collected[sequence] = LabeledPeptide(
                record=SequenceRecord(
                    id=f"IEDB_POS_{entry.get('structure_id')}",
                    sequence=sequence,
                    description="IEDB linear B-cell epitope (Positive)",
                ),
                label=1,
                source_group=str(group),
            )

        offset += page_size
        time.sleep(0.2)  # Cortesia hacia la API publica.

    if not collected:
        raise DatasetPrepError("La API de IEDB no devolvio ningun epitopo positivo valido.")

    if len(collected) < target_count:
        logger.warning(
            "Solo se obtuvieron %d peptidos positivos unicos de los %d solicitados.",
            len(collected),
            target_count,
        )

    peptides = list(collected.values())
    random.Random(seed).shuffle(peptides)
    return peptides[:target_count]


def fetch_housekeeping_proteins(accessions: Sequence[str]) -> Dict[str, str]:
    """Descarga las secuencias completas de un panel de proteinas via UniProt REST.

    Args:
        accessions: Accesiones Swiss-Prot a recuperar.

    Returns:
        Diccionario ``accession -> secuencia`` (solo alfabeto canonico de 20
        aminoacidos).

    Raises:
        DatasetPrepError: Si ninguna accesion pudo recuperarse.
    """
    endpoint = f"{Settings.UNIPROT_API_BASE}/uniprotkb/accessions"
    sequences: Dict[str, str] = {}
    chunk_size = 90

    for start in range(0, len(accessions), chunk_size):
        chunk = accessions[start : start + chunk_size]
        params = {"accessions": ",".join(chunk), "format": "fasta"}

        last_error: Optional[Exception] = None
        raw_text: Optional[str] = None
        for attempt in range(1, Settings.HTTP_MAX_RETRIES + 1):
            try:
                response = requests.get(endpoint, params=params, timeout=Settings.HTTP_TIMEOUT_SECONDS)
                response.raise_for_status()
                raw_text = response.text
                break
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Intento %d/%d fallido descargando panel housekeeping: %s",
                    attempt,
                    Settings.HTTP_MAX_RETRIES,
                    exc,
                )
                time.sleep(min(2**attempt, 10))

        if raw_text is None:
            logger.error("No se pudo descargar el lote de accesiones %s: %s", chunk, last_error)
            continue

        sequences.update(_parse_fasta_text(raw_text))

    if not sequences:
        raise DatasetPrepError("No se pudo descargar ninguna proteina housekeeping desde UniProt.")

    missing = set(accessions) - set(sequences.keys())
    if missing:
        logger.warning("Accesiones no resueltas por UniProt: %s", sorted(missing))

    return sequences


def _parse_fasta_text(fasta_text: str) -> Dict[str, str]:
    """Parsea texto FASTA crudo devuelto por UniProt en ``accession -> secuencia``."""
    sequences: Dict[str, str] = {}
    current_accession: Optional[str] = None
    current_chunks: List[str] = []

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_accession is not None:
                sequences[current_accession] = "".join(current_chunks)
            # Cabecera UniProt: >sp|ACCESSION|ENTRY_NAME ...
            parts = line[1:].split("|")
            current_accession = parts[1] if len(parts) >= 2 else parts[0]
            current_chunks = []
        else:
            current_chunks.append(line.upper())

    if current_accession is not None:
        sequences[current_accession] = "".join(current_chunks)

    return {acc: seq for acc, seq in sequences.items() if seq and not (set(seq) - CANONICAL_AA)}


def generate_negative_peptides(
    protein_sequences: Dict[str, str],
    target_count: int,
    length_pool: Sequence[int],
    seed: int,
) -> List[LabeledPeptide]:
    """Genera fragmentos peptidicos negativos por ventana deslizante aleatoria.

    Las longitudes de fragmento se muestrean de la distribucion empirica de
    longitudes de la clase positiva (``length_pool``), para evitar que la
    longitud por si sola sea una senal discriminante trivial entre clases.

    Args:
        protein_sequences: Mapa ``accession -> secuencia completa`` del panel
            housekeeping.
        target_count: Numero deseado de fragmentos negativos unicos.
        length_pool: Longitudes observadas en la clase positiva, usadas como
            distribucion empirica de muestreo.
        seed: Semilla para reproducibilidad.

    Returns:
        Lista de :class:`LabeledPeptide` con ``label=0``.

    Raises:
        DatasetPrepError: Si ``protein_sequences`` o ``length_pool`` estan vacios.
    """
    if not protein_sequences:
        raise DatasetPrepError("No hay proteinas housekeeping disponibles para generar negativos.")
    if not length_pool:
        raise DatasetPrepError("No hay distribucion de longitudes positiva disponible.")

    rng = random.Random(seed)
    accessions = list(protein_sequences.keys())
    collected: Dict[str, LabeledPeptide] = {}
    max_attempts = target_count * 50
    attempts = 0

    while len(collected) < target_count and attempts < max_attempts:
        attempts += 1
        accession = accessions[attempts % len(accessions)]
        sequence = protein_sequences[accession]

        window = rng.choice(list(length_pool))
        window = max(9, min(window, len(sequence)))
        if window > len(sequence):
            continue

        start = rng.randint(0, len(sequence) - window)
        fragment = sequence[start : start + window]

        if set(fragment) - CANONICAL_AA:
            continue
        if fragment in collected:
            continue

        collected[fragment] = LabeledPeptide(
            record=SequenceRecord(
                id=f"NEG_{accession}_{start + 1}_{start + window}",
                sequence=fragment,
                description=f"Housekeeping control fragment ({accession})",
            ),
            label=0,
            source_group=accession,
        )

    if len(collected) < target_count:
        logger.warning(
            "Solo se generaron %d fragmentos negativos unicos de los %d solicitados "
            "(agotado el presupuesto de intentos).",
            len(collected),
            target_count,
        )

    peptides = list(collected.values())
    rng.shuffle(peptides)
    return peptides[:target_count]


def _assign_groups_to_splits(
    group_sizes: Dict[str, int],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Asigna claves de grupo a train/val/test balanceando por conteo de items.

    Algoritmo de particion voraz: procesa los grupos en orden aleatorio y
    asigna cada grupo completo (nunca fragmentado) al split cuya proporcion
    actual este mas alejada (por debajo) de su objetivo.

    Args:
        group_sizes: Mapa ``clave_de_grupo -> numero de items en ese grupo``.
        ratios: Proporciones ``(train, val, test)``, deben sumar ~1.0.
        seed: Semilla para el orden aleatorio de asignacion de grupos.

    Returns:
        Diccionario ``split_name -> lista de claves de grupo`` asignadas.
    """
    group_keys = list(group_sizes.keys())
    random.Random(seed).shuffle(group_keys)

    total = sum(group_sizes.values())
    targets = {name: total * ratio for name, ratio in zip(SPLIT_NAMES, ratios)}
    current: Dict[str, int] = {name: 0 for name in SPLIT_NAMES}
    result: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}

    for key in group_keys:
        size = group_sizes[key]

        def deficit(name: str) -> float:
            target = targets[name] or 1e-9
            return current[name] / target

        chosen = min(SPLIT_NAMES, key=deficit)
        result[chosen].append(key)
        current[chosen] += size

    return result


def grouped_split(
    items: Sequence[LabeledPeptide],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[LabeledPeptide]]:
    """Divide ``items`` en train/val/test agrupando por ``source_group``.

    Garantiza que ningun grupo de origen (misma proteina/antigeno) se
    fragmente entre splits, delegando el balanceo a
    :func:`_assign_groups_to_splits`.

    Args:
        items: Peptidos etiquetados a dividir (de una unica clase).
        ratios: Proporciones ``(train, val, test)``, deben sumar ~1.0.
        seed: Semilla para el orden aleatorio de asignacion de grupos.

    Returns:
        Diccionario ``split_name -> lista de LabeledPeptide``.
    """
    groups: Dict[str, List[LabeledPeptide]] = {}
    for item in items:
        groups.setdefault(item.source_group, []).append(item)

    group_sizes = {key: len(group_items) for key, group_items in groups.items()}
    key_splits = _assign_groups_to_splits(group_sizes, ratios, seed)

    return {
        name: [item for key in keys for item in groups[key]] for name, keys in key_splits.items()
    }


def generate_hard_negative_macromolecules(
    protein_sequences: Dict[str, str],
    target_count: int,
    min_length: int,
    max_length: int,
    seed: int,
) -> List[LabeledPeptide]:
    """Genera fragmentos macromoleculares largos, 100% inertes, para hard negative mining.

    Corrige el sesgo de "distribution shift" del ``ResidueClassifier``: al
    entrenarse solo con peptidos cortos (9-25 aa), la red nunca observaba
    ejemplos negativos largos y terminaba hiper-activando (~98% densidad de
    falsos positivos) sobre proteinas nativas completas de cientos de
    residuos. Estos fragmentos, extraidos del mismo panel housekeeping y
    etiquetados 100% como no-epitopo en toda su longitud, enseñan a la red
    que una cadena proteica extensa NO implica antigenicidad generalizada.

    Args:
        protein_sequences: Mapa ``accession -> secuencia completa`` del panel
            housekeeping, ya restringido a las accesiones permitidas para
            este split (evita fuga de datos entre splits).
        target_count: Numero deseado de fragmentos largos unicos.
        min_length: Longitud minima del fragmento (aa).
        max_length: Longitud maxima del fragmento (aa); acotada por el limite
            fisico de contexto de ESM-2 (``Settings.ESM_MAX_SEQUENCE_LENGTH``).

    Returns:
        Lista de :class:`LabeledPeptide` con ``label=0``, longitud en
        ``[min_length, max_length]``.
    """
    if not protein_sequences or target_count <= 0:
        return []

    rng = random.Random(seed)
    accessions = [acc for acc, seq in protein_sequences.items() if len(seq) >= min_length]
    if not accessions:
        logger.warning(
            "Ningun accession del split tiene longitud >= %d; no se generaron hard negatives.",
            min_length,
        )
        return []

    collected: Dict[str, LabeledPeptide] = {}
    max_attempts = target_count * 50
    attempts = 0

    while len(collected) < target_count and attempts < max_attempts:
        attempts += 1
        accession = accessions[attempts % len(accessions)]
        sequence = protein_sequences[accession]

        upper_bound = min(max_length, len(sequence))
        if upper_bound < min_length:
            continue
        window = rng.randint(min_length, upper_bound)

        start = rng.randint(0, len(sequence) - window)
        fragment = sequence[start : start + window]

        if set(fragment) - CANONICAL_AA:
            continue
        if fragment in collected:
            continue

        collected[fragment] = LabeledPeptide(
            record=SequenceRecord(
                id=f"HARDNEG_{accession}_{start + 1}_{start + window}",
                sequence=fragment,
                description=f"Macromolecular hard negative ({accession}, {window} aa)",
            ),
            label=0,
            source_group=accession,
        )

    if len(collected) < target_count:
        logger.warning(
            "Solo se generaron %d hard negatives macromoleculares de los %d solicitados.",
            len(collected),
            target_count,
        )

    peptides = list(collected.values())
    rng.shuffle(peptides)
    return peptides[:target_count]


def write_split_outputs(
    split_name: str, items: Sequence[LabeledPeptide], output_dir: Path
) -> None:
    """Escribe el FASTA combinado, el manifiesto CSV y los FASTA por clase de un split.

    Args:
        split_name: Nombre del split (``"train"``, ``"val"`` o ``"test"``).
        items: Peptidos etiquetados que componen el split.
        output_dir: Directorio destino.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / f"{split_name}.fasta"
    manifest_path = output_dir / f"{split_name}_manifest.csv"
    positive_path = output_dir / f"{split_name}_positive.fasta"
    negative_path = output_dir / f"{split_name}_negative.fasta"

    with open(combined_path, "w", encoding="utf-8") as combined_handle, open(
        positive_path, "w", encoding="utf-8"
    ) as positive_handle, open(negative_path, "w", encoding="utf-8") as negative_handle, open(
        manifest_path, "w", encoding="utf-8", newline=""
    ) as manifest_handle:

        writer = csv.writer(manifest_handle)
        writer.writerow(["id", "label", "source_group", "length", "sequence"])

        for item in items:
            fasta_line = f">{item.record.id}\n{item.record.sequence}\n"
            combined_handle.write(fasta_line)
            if item.label == 1:
                positive_handle.write(fasta_line)
            else:
                negative_handle.write(fasta_line)

            writer.writerow(
                [
                    item.record.id,
                    item.label,
                    item.source_group,
                    len(item.record.sequence),
                    item.record.sequence,
                ]
            )

    n_pos = sum(1 for item in items if item.label == 1)
    n_neg = len(items) - n_pos
    logger.info(
        "Split '%s' escrito en '%s': %d positivos, %d negativos (%d total).",
        split_name,
        output_dir,
        n_pos,
        n_neg,
        len(items),
    )


def prepare_dataset(
    n_positive: int = Settings.TRAINING_N_POSITIVE,
    n_negative: int = Settings.TRAINING_N_NEGATIVE,
    min_length: int = Settings.TRAINING_MIN_PEPTIDE_LEN,
    max_length: int = Settings.TRAINING_MAX_PEPTIDE_LEN,
    seed: int = Settings.TRAINING_SEED,
    output_dir: Path = Settings.TRAINING_DATA_DIR,
    hard_negative_ratio: float = Settings.HARD_NEGATIVE_RATIO,
    hard_negative_min_len: int = Settings.HARD_NEGATIVE_MIN_LEN,
    hard_negative_max_len: int = Settings.HARD_NEGATIVE_MAX_LEN,
) -> Dict[str, List[LabeledPeptide]]:
    """Orquesta la curacion completa del dataset de calibracion.

    Ademas de los peptidos cortos positivos (IEDB) y negativos (housekeeping),
    integra "hard negatives" macromoleculares: fragmentos largos (``[100,
    1022]`` aa por defecto) del mismo panel housekeeping, 100% etiquetados
    como no-epitopo, en una proporcion ``hard_negative_ratio`` del total de
    cada split. La asignacion de proteinas housekeeping a cada split se
    reutiliza identica entre los negativos cortos y los hard negatives
    (misma semilla ``seed + 1``), garantizando que ninguna proteina aparezca
    simultaneamente en dos splits distintos.

    Args:
        n_positive: Numero objetivo de epitopos positivos (IEDB).
        n_negative: Numero objetivo de fragmentos negativos cortos (housekeeping).
        min_length: Longitud minima de peptido corto.
        max_length: Longitud maxima de peptido corto.
        seed: Semilla global de reproducibilidad.
        output_dir: Directorio destino de los FASTA/manifiestos generados.
        hard_negative_ratio: Proporcion objetivo de hard negatives sobre el
            total de cada split (p. ej. 0.30 = 30%).
        hard_negative_min_len: Longitud minima de un hard negative (aa).
        hard_negative_max_len: Longitud maxima de un hard negative (aa).

    Returns:
        Diccionario ``split_name -> lista de LabeledPeptide`` ya escrito a disco.
    """
    positives = fetch_iedb_positive_peptides(n_positive, min_length, max_length, seed)
    length_pool = [len(p.record.sequence) for p in positives]

    housekeeping_seqs = fetch_housekeeping_proteins(HOUSEKEEPING_ACCESSIONS)
    negatives = generate_negative_peptides(housekeeping_seqs, n_negative, length_pool, seed)

    ratios = (Settings.TRAIN_SPLIT_RATIO, Settings.VAL_SPLIT_RATIO, Settings.TEST_SPLIT_RATIO)
    positive_splits = grouped_split(positives, ratios, seed)

    negative_groups: Dict[str, List[LabeledPeptide]] = {}
    for item in negatives:
        negative_groups.setdefault(item.source_group, []).append(item)
    negative_group_sizes = {key: len(group_items) for key, group_items in negative_groups.items()}
    negative_key_splits = _assign_groups_to_splits(negative_group_sizes, ratios, seed + 1)
    negative_splits = {
        name: [item for key in keys for item in negative_groups[key]]
        for name, keys in negative_key_splits.items()
    }

    hard_negative_splits: Dict[str, List[LabeledPeptide]] = {name: [] for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        split_accessions = negative_key_splits[split_name]
        if not split_accessions:
            continue

        base_count = len(positive_splits[split_name]) + len(negative_splits[split_name])
        if base_count == 0 or hard_negative_ratio <= 0:
            continue
        n_hard = round(base_count * hard_negative_ratio / (1.0 - hard_negative_ratio))

        split_protein_seqs = {
            acc: housekeeping_seqs[acc] for acc in split_accessions if acc in housekeeping_seqs
        }
        hard_negative_splits[split_name] = generate_hard_negative_macromolecules(
            split_protein_seqs,
            n_hard,
            hard_negative_min_len,
            hard_negative_max_len,
            seed=seed + 1000 + hash(split_name) % 1000,
        )

    final_splits: Dict[str, List[LabeledPeptide]] = {}
    rng = random.Random(seed + 2)
    for split_name in SPLIT_NAMES:
        combined = (
            positive_splits[split_name] + negative_splits[split_name] + hard_negative_splits[split_name]
        )
        rng.shuffle(combined)
        final_splits[split_name] = combined
        write_split_outputs(split_name, combined, output_dir)

        n_hard = len(hard_negative_splits[split_name])
        logger.info(
            "Split '%s': %d hard negatives macromoleculares integrados (~%.1f%% del total).",
            split_name,
            n_hard,
            100.0 * n_hard / len(combined) if combined else 0.0,
        )

    total_hard = sum(len(items) for items in hard_negative_splits.values())
    total_items = sum(len(items) for items in final_splits.values())
    logger.info(
        "Curacion completada: %d positivos, %d negativos cortos, %d hard negatives "
        "macromoleculares, %d total.",
        len(positives),
        len(negatives),
        total_hard,
        total_items,
    )
    return final_splits


def _parse_arguments() -> argparse.Namespace:
    """Define la interfaz de linea de comandos para la curacion standalone."""
    parser = argparse.ArgumentParser(
        description="Curacion del dataset de calibracion IEDB (positivo) / housekeeping (negativo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-positive", type=int, default=Settings.TRAINING_N_POSITIVE)
    parser.add_argument("--n-negative", type=int, default=Settings.TRAINING_N_NEGATIVE)
    parser.add_argument("--min-length", type=int, default=Settings.TRAINING_MIN_PEPTIDE_LEN)
    parser.add_argument("--max-length", type=int, default=Settings.TRAINING_MAX_PEPTIDE_LEN)
    parser.add_argument("--seed", type=int, default=Settings.TRAINING_SEED)
    parser.add_argument("--output-dir", type=Path, default=Settings.TRAINING_DATA_DIR)
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=Settings.HARD_NEGATIVE_RATIO,
        help="Proporcion de hard negatives macromoleculares sobre el total de cada split.",
    )
    parser.add_argument("--hard-negative-min-len", type=int, default=Settings.HARD_NEGATIVE_MIN_LEN)
    parser.add_argument("--hard-negative-max-len", type=int, default=Settings.HARD_NEGATIVE_MAX_LEN)
    return parser.parse_args()


def main() -> int:
    """Punto de entrada standalone: ``python -m src.training.dataset_prep``."""
    args = _parse_arguments()
    try:
        prepare_dataset(
            n_positive=args.n_positive,
            n_negative=args.n_negative,
            min_length=args.min_length,
            max_length=args.max_length,
            seed=args.seed,
            output_dir=args.output_dir,
            hard_negative_ratio=args.hard_negative_ratio,
            hard_negative_min_len=args.hard_negative_min_len,
            hard_negative_max_len=args.hard_negative_max_len,
        )
        return 0
    except DatasetPrepError as exc:
        logger.critical("Fallo fatal durante la curacion del dataset: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
