"""Abstract base for exporters."""

from abc import ABC, abstractmethod
from typing import Any

from tmatlas.inspectors.base import BaseInspector


class BaseExporter(ABC):
    """
    Abstract for the exporter class.
    """

    def __init__(self, inspector: BaseInspector):
        self.inspector = inspector

    @abstractmethod
    def export(self) -> Any:
        """Produce the exporter representation."""
