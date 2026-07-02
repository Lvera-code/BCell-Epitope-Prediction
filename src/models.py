"""Modelos y contratos de datos inmutables para el pipeline HTS."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class OrganismClass(str, Enum):
    VIRAL = "viral"
    BACTERIAL = "bacterial"
    TUMOR = "tumor"
    GENERIC = "generic"


@dataclass(frozen=True)
class SequenceRecord:
    id: str
    sequence: str
    description: str = ""


@dataclass
class AntigenicityResult:
    record: SequenceRecord
    score: float
    is_antigenic: bool
    method: str
    organism_class: OrganismClass


@dataclass
class EpitopeResidue:
    position: int
    residue: str
    epitope_probability: float
    is_epitope: bool


@dataclass
class EpitopeResult:
    antigenicity: AntigenicityResult
    residues: List[EpitopeResidue]
    epitope_regions: List[Tuple[int, int]]