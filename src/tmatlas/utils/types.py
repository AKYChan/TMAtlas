"""Data structures for the TM export pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, Union


@dataclass
class FeatureClass:
    """Class for the definition of a single original (pre-binarized) feature."""

    name: str
    type: str  # "binary" | "continuous"
    range: tuple[float, float]
    thresholds: list[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "range": list(self.range),
            "thresholds": self.thresholds,
        }


@dataclass
class Literal:
    """A single literal (boolean condition) inside a clause."""

    feature: str
    operator: str  # "≥" or "<"
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "operator": self.operator,
            "threshold": self.threshold,
        }


@dataclass
class ClauseWeight:
    """A clause's weight and derived polarity for a single class (or regression)."""

    value: float
    polarity: str  # "positive" | "negative"

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "polarity": self.polarity}


@dataclass
class Clause:
    """A fully translated clause with its literals and weight."""

    id: int
    literals: list[Literal]
    weights: Union[ClauseWeight, Dict[str, ClauseWeight]]

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.weights, ClauseWeight):
            weights_out = self.weights.to_dict()
        else:
            weights_out = {cls: cw.to_dict() for cls, cw in self.weights.items()}

        return {
            "id": self.id,
            "literals": [lit.to_dict() for lit in self.literals],
            "weights": weights_out,
        }


@dataclass
class ModelInfo:
    """High-level model information (type, task, classes)."""

    type: str  # "classification" | "regression"
    task: str  # "binary" | "multiclass" | "regression"
    classes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "task": self.task,
            "classes": self.classes,
        }


@dataclass
class ModelMetadata:
    """Training hyper-parameters and training data retention."""

    num_clauses: int
    T: float | None = None
    s: float | None = None
    weighted_clauses: bool = False
    created: str | None = None
    min_y: float | None = None
    max_y: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "numClauses": self.num_clauses,
            "T": self.T,
            "s": self.s,
            "weightedClauses": self.weighted_clauses,
            "created": self.created,
        }

        if self.min_y is not None:
            d["min_y"] = self.min_y
        if self.max_y is not None:
            d["max_y"] = self.max_y
        return d
