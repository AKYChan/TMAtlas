"""Implementation of inpsector for TMU models."""

import re
from datetime import datetime, timezone
from typing import Any, List, Dict, Union
from numpy.typing import NDArray

from tmatlas.inspectors.base import BaseInspector
from tmatlas.inspectors.features import FeatureInspector
from tmatlas.utils.types import Clause, ClauseWeight, Literal, ModelInfo, ModelMetadata

# To match binarizer labels to create for example "x1 ≥ 45.3"
_LITERAL_RE = re.compile(r"^(.+?)\s*≥\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")


class TMUInspector(BaseInspector):
    """
    Extracts model structure from a trained TMU model.

    Args:
        model: A trained TMRegressor / TMClassifier with a clause_bank, number_of_clauses,
                and weight_bank attributes.
        feature_inspector: A pre-built FeatureInspector for boolean labels.
        classes: Class names for classification models. If none are provided, falls back
                to model.classes_, then assumes regression.
    """

    def __init__(
        self,
        model: Any,
        feature_inspector: FeatureInspector,
        classes: List[str] | None = None,
    ):
        if not hasattr(model, "clause_bank"):
            raise AttributeError("Model must have a clause_bank attribute.")

        self.model = model
        self.feature_inspector = feature_inspector
        self.classes = classes or getattr(model, "classes_", [])
        self.is_classification = bool(self.classes)

    def get_model_info(self) -> ModelInfo:
        if self.is_classification:
            task = "multiclass" if len(self.classes) > 2 else "binary"
            return ModelInfo(type="classification", task=task, classes=self.classes)
        return ModelInfo(type="regression", task="regression")

    def get_metadata(self) -> ModelMetadata:
        meta = ModelMetadata(
            num_clauses=self.model.number_of_clauses,
            T=getattr(self.model, "T", None),
            s=getattr(self.model, "s", None),
            weighted_clauses=getattr(self.model, "weighted_clauses", False),
            created=datetime.now(timezone.utc).isoformat(),
        )
        if not self.is_classification:
            meta.min_y = float(self.model.min_y)
            meta.max_y = float(self.model.max_y)
        return meta

    def get_clauses(self) -> List[Clause]:
        n_literals_total = self.model.clause_bank.number_of_literals
        n_features = n_literals_total // 2
        boolean_labels = self.feature_inspector.boolean_labels

        regression_weights = None
        if not self.is_classification:
            regression_weights = self.model.weight_bank.get_weights()

        clauses: List[Clause] = []
        for cid in range(self.model.number_of_clauses):
            literals = self._extract_literals(
                cid, n_features, n_literals_total, boolean_labels
            )
            weights = self._extract_weights(cid, regression_weights)
            clauses.append(Clause(id=cid, literals=literals, weights=weights))
        return clauses

    # Private helper functions

    def _extract_literals(
        self,
        clause_id: int,
        n_features: int,
        n_literals_total: int,
        boolean_labels: NDArray[str],
    ) -> List[Literal]:
        """
        Extracts the literal information for trained TMU model.

        Args:
            clause_id: The current clause that is being inspected.
            n_features: Total number of (pre-binarized) features.
            n_literals_total: Total number of binarized features used in the TM.
            boolean_labels: Labels for the bins for each feature.
        """
        literals: List[Literal] = []
        for lit_idx in range(n_literals_total):
            if self.model.get_ta_action(clause_id, lit_idx) != 1:
                continue

            is_negated = lit_idx >= n_features
            feat_idx = lit_idx - n_features if is_negated else lit_idx
            desc = boolean_labels[feat_idx]

            match = _LITERAL_RE.match(desc)
            if not match:
                continue

            feature_name, val_str = match.groups()
            literals.append(
                Literal(
                    feature=feature_name.strip(),
                    operator="<" if is_negated else "≥",
                    threshold=float(val_str),
                )
            )
        return literals

    def _extract_weights(
        self, clause_id: int, regression_weights: NDArray | None
    ) -> Union[ClauseWeight, Dict[str, ClauseWeight]]:
        if self.is_classification:
            # Coalesced model so single clause bank but each class has independent weight bank.
            return {
                cls: ClauseWeight(
                    value=float(self.model.get_weight(i, clause_id)),
                    polarity="positive"
                    if self.model.get_weight(i, clause_id) >= 0
                    else "negative",
                )
                for i, cls in enumerate(self.classes)
            }
        # Regression TM has constant positive polarity
        if regression_weights is None:
            raise ValueError(
                f"Regression weight bank returned None; cannot extract weight for clause {clause_id}."
            )
        w = float(regression_weights[clause_id])
        return ClauseWeight(value=w, polarity="positive")
