"""Interfaz abstracta comun para los motores computacionales del pipeline."""

from abc import ABC, abstractmethod
from typing import Generic, List, Sequence, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class BaseEngine(ABC, Generic[TIn, TOut]):
    """Contrato minimo que debe cumplir cualquier motor de inferencia del pipeline.

    Cada fase del pipeline (antigenicidad, epitopos) implementa esta interfaz
    para permitir composicion y sustitucion transparente de motores (patron
    Strategy), sin acoplar el orquestador (``main.py``) a una implementacion
    concreta.
    """

    @abstractmethod
    def run(self, items: Sequence[TIn]) -> List[TOut]:
        """Ejecuta inferencia sobre un lote de datos de entrada.

        Args:
            items: Secuencia homogenea de elementos de entrada para este motor.

        Returns:
            Lista de resultados, en el mismo orden logico que ``items`` salvo
            que el motor documente explicitamente lo contrario.
        """
        raise NotImplementedError
