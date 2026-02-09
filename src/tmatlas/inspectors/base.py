"""Abstract class for base inspectors"""

from abc import ABC, abstractmethod
from typing import List

from tmatlas.utils.types import Clause, ModelInfo, ModelMetadata


class BaseInspector(ABC):
    """
    Base class the inspectors.
    """

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return high-level model information."""

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return training metadata / hyper-parameters."""

    @abstractmethod
    def get_clauses(self) -> List[Clause]:
        """Extract every clause with its literals and weights."""
