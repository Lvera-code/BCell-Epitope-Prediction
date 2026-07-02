"""Interfaz abstracta para los motores computacionales del pipeline."""

from abc import ABC, abstractmethod
from typing import Sequence, Any


class BaseEngine(ABC):
    @abstractmethod
    def run(self, items: Sequence[Any]) -> list[Any]:
        """Ejecuta inferencia sobre un lote de datos."""
        pass