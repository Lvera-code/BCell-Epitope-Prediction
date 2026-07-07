"""Modelos y contratos de datos inmutables para el pipeline SOTA-B-Epitope-Pipeline.

Todas las entidades que cruzan una frontera de fase (Fase 1 -> Fase 2 -> Validacion)
se representan como ``dataclasses`` para garantizar contratos explicitos, tipado
estatico y trazabilidad completa desde la secuencia cruda hasta el reporte final.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class SequenceRecord:
    """Representa una secuencia proteica ya saneada por el modulo de aduana.

    Attributes:
        id: Identificador unico de la secuencia (primer token del header FASTA).
        sequence: Secuencia de aminoacidos canonicos en mayusculas, sin residuos
            ambiguos ni caracteres ilegales.
        description: Texto libre restante del header FASTA (metadatos opcionales).
    """

    id: str
    sequence: str
    description: str = ""


@dataclass
class AntigenicityResult:
    """Resultado de la Fase 1 (cribado grueso de antigenicidad via 1D-CNN).

    Attributes:
        record: Secuencia de origen evaluada.
        score: Probabilidad de antigenicidad en el rango [0, 1], calibrada via
            Platt Scaling sobre el logit crudo de la red convolucional (o un
            sigmoide sin calibrar como respaldo si no existe artefacto de
            calibracion; ver ``AntigenicityCNNEngine.is_calibrated``).
        is_antigenic: Indica si ``score`` supera el umbral de corte configurado.
        method: Identificador legible del metodo/version que genero el score.
    """

    record: SequenceRecord
    score: float
    is_antigenic: bool
    method: str


@dataclass
class EpitopeResidue:
    """Prediccion por residuo individual emitida por la Fase 2.

    Attributes:
        position: Posicion 1-indexada del residuo dentro de la secuencia original.
        residue: Codigo de una letra del aminoacido en esa posicion.
        epitope_probability: Probabilidad calibrada de pertenecer a un epitopo
            lineal de celulas B.
        is_epitope: Indica si ``epitope_probability`` supera el umbral configurado.
    """

    position: int
    residue: str
    epitope_probability: float
    is_epitope: bool


@dataclass
class EpitopeResult:
    """Resultado consolidado de la Fase 2 para una secuencia que supero la Fase 1.

    Attributes:
        antigenicity: Resultado de Fase 1 que habilito el paso a Fase 2.
        residues: Prediccion detallada residuo a residuo.
        epitope_regions: Lista de tuplas ``(inicio, fin)`` 1-indexadas que delimitan
            regiones contiguas de epitopo (longitud minima configurable).
    """

    antigenicity: AntigenicityResult
    residues: List[EpitopeResidue]
    epitope_regions: List[Tuple[int, int]]


@dataclass
class BenchmarkReport:
    """Reporte estadistico riguroso producido por la suite de auditoria cientifica.

    Attributes:
        true_positives: Numero de secuencias positivas (IEDB) correctamente
            clasificadas como antigenicas/epitopo.
        true_negatives: Numero de secuencias negativas (housekeeping) correctamente
            clasificadas como no antigenicas.
        false_positives: Numero de secuencias negativas mal clasificadas como
            antigenicas (fallo de especificidad).
        false_negatives: Numero de secuencias positivas mal clasificadas como no
            antigenicas (fallo de sensibilidad).
        sensitivity: Tasa de verdaderos positivos (recall) = TP / (TP + FN).
        specificity: Tasa de verdaderos negativos = TN / (TN + FP).
        false_positive_rate: FPR = FP / (FP + TN) = 1 - especificidad.
        roc_auc: Area bajo la curva ROC calculada sobre los scores continuos.
        threshold_used: Umbral de decision aplicado para binarizar los scores.
        n_positive: Numero total de secuencias en el FASTA positivo evaluado.
        n_negative: Numero total de secuencias en el FASTA negativo evaluado.
    """

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    sensitivity: float
    specificity: float
    false_positive_rate: float
    roc_auc: float
    threshold_used: float
    n_positive: int = 0
    n_negative: int = 0
    fpr_curve: List[float] = field(default_factory=list)
    tpr_curve: List[float] = field(default_factory=list)
