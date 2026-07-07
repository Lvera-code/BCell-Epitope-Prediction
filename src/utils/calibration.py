"""Calibracion de Platt (regresion logistica 1D) para los logits de la 1D-CNN de Fase 1.

La 1D-CNN de antigenicidad entrenada con ``BCEWithLogitsLoss`` produce logits
crudos, no probabilidades calibradas: con el desbalance de clases del dataset
(hard negatives macromoleculares sobre-representados), el sigmoide directo de
esos logits comprime la salida en un rango angosto (p. ej. ``[0.000, 0.005]``),
lo que dispara falsos negativos masivos si se compara contra un umbral fijo
como 0.5 o 0.6.

:class:`PlattScaler` corrige esto ajustando una unica regresion logistica 1D
(``A``, ``B``) sobre los logits de un hold-out de calibracion estratificado que
la red NUNCA vio durante el backpropagation (ver
``src/training/trainer.py::train_antigenicity_cnn``), evitando fuga de datos.
La probabilidad calibrada resultante es ``sigmoid(A * logit + B)``.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PlattScaler:
    """Mapea un logit crudo de la 1D-CNN a una probabilidad calibrada.

    Attributes:
        coef_a: Pendiente ``A`` de la regresion logistica de Platt.
        intercept_b: Intercepto ``B`` de la regresion logistica de Platt.
    """

    coef_a: float
    intercept_b: float

    def transform(self, logits: Sequence[float]) -> np.ndarray:
        """Aplica ``sigmoid(A * logit + B)`` a un array de logits crudos.

        Args:
            logits: Logits crudos emitidos por la cabeza lineal de la 1D-CNN
                (antes de cualquier sigmoide).

        Returns:
            Array de ``numpy`` con las probabilidades calibradas, mismo shape
            que ``logits``.
        """
        logits_arr = np.asarray(logits, dtype=np.float64)
        z = self.coef_a * logits_arr + self.intercept_b
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, logits: Sequence[float], labels: Sequence[int]) -> "PlattScaler":
        """Ajusta ``A`` y ``B`` sobre pares ``(logit, etiqueta)`` de un hold-out.

        Args:
            logits: Logits crudos del hold-out de calibracion (no visto en
                backprop).
            labels: Etiquetas binarias (0/1) correspondientes a ``logits``.

        Returns:
            Instancia de :class:`PlattScaler` ajustada.

        Raises:
            ValueError: Si ``labels`` no contiene ambas clases (0 y 1); la
                regresion logistica de Platt no esta definida sobre una unica
                clase.
        """
        logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        labels_arr = np.asarray(labels, dtype=np.int64)

        if len(set(labels_arr.tolist())) < 2:
            raise ValueError(
                "El hold-out de calibracion debe contener ambas clases (positiva "
                "y negativa) para ajustar la regresion logistica de Platt."
            )

        classifier = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        classifier.fit(logits_arr, labels_arr)
        return cls(
            coef_a=float(classifier.coef_.ravel()[0]),
            intercept_b=float(classifier.intercept_[0]),
        )

    def save(self, path: Path) -> None:
        """Persiste los coeficientes como artefacto pickle.

        Args:
            path: Ruta destino. Se crean los directorios padre si hace falta.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"coef_a": self.coef_a, "intercept_b": self.intercept_b}, handle)

    @classmethod
    def load(cls, path: Path) -> "PlattScaler":
        """Carga un :class:`PlattScaler` desde un artefacto pickle previamente guardado.

        Args:
            path: Ruta al artefacto ``.pkl``.

        Returns:
            Instancia de :class:`PlattScaler` reconstruida.

        Raises:
            FileNotFoundError: Si ``path`` no existe.
            KeyError: Si el artefacto no contiene las claves esperadas.
        """
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(coef_a=float(payload["coef_a"]), intercept_b=float(payload["intercept_b"]))
