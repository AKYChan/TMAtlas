"""JSON model exporter."""

from typing import Any, Dict

from tmatlas.exporters.base import BaseExporter
from tmatlas.inspectors.base import BaseInspector
from tmatlas.inspectors.features import FeatureInspector


class JsonExporter(BaseExporter):
    """
    Exports a trained TM model to a JSON-serialisable dictionary.

    Args:
        inspector: Any BaseInspector implementation has been wired into
                    a trained model.
        feature_inspector: A FeatureInspector for resolving feature definitions.
    """

    def __init__(
        self,
        inspector: BaseInspector,
        feature_inspector: FeatureInspector,
    ):
        super().__init__(inspector)
        self.feature_inspector = feature_inspector

    def export(self) -> Dict[str, Any]:
        """
        Return the full model as a JSON-ready dictionary
        """
        feature_defs = self.feature_inspector.extract()
        clauses = self.inspector.get_clauses()

        return {
            "model": self.inspector.get_model_info().to_dict(),
            "metadata": self.inspector.get_metadata().to_dict(),
            "features": [f.to_dict() for f in feature_defs],
            "clauses": [c.to_dict() for c in clauses],
        }
