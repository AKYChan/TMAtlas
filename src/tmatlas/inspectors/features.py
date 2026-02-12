"""Feature and binarizer inspector."""

import numpy as np
from typing import Any, List
from numpy.typing import NDArray

from tmatlas.utils.types import FeatureClass


class FeatureInspector:
    """
    Builds the `FeatureClass` objects from a fitted binarizer and the raw (pre-binarized)
    training data.

    Args:
        binarizer: A fitted binarizxer with `get_features_names_out()` and a `unique_values` attribute.
        X: The original training data as a 2D array-like of shape (n_samples, n_features)
        feature_names: An optional humen-readable column of names. Falls back `x0`, `x1` if not given.
    Raises:
        ValueError: On dimension mismatches.
        AttributeError: If the binarizer lacks required attributes.
    """

    def __init__(
        self,
        binarizer: Any,
        X: Any,
        feature_names: List[str] | None,
    ):
        if X is None:
            raise ValueError("X is empty / None.")
        if not hasattr(binarizer, "unique_values"):
            raise AttributeError("Binarizer must have a `unique_values` attribute.")
        if not hasattr(binarizer, "get_feature_names_out"):
            raise AttributeError(
                "Binarizer must have a `get_feature_names_out` method."
            )

        self.binarizer = binarizer
        self.X = np.asarray(X, dtype=np.float32)
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]

        if len(self.feature_names) != X.shape[1]:
            raise ValueError("Length of feature_names must match columns in X.")

        self.boolean_labels: NDArray[str] = self.binarizer.get_feature_names_out(
            self.feature_names
        )

    def extract(self) -> List[FeatureClass]:
        """Return a FeatureClass for every original feature column."""
        defs: List[FeatureClass] = []
        for i, name in enumerate(self.feature_names):
            col = self.X[:, i]
            vmin, vmax = float(col.min()), float(col.max())
            unique = np.unique(col)

            ftype = (
                "binary"
                if np.array_equal(unique, np.array([0, 1]))
                or np.array_equal(unique, np.array([1, 0]))
                else "continuous"
            )

            thresholds = [float(v) for v in self.binarizer.unique_values[i]]
            defs.append(
                FeatureClass(
                    name=name,
                    type=ftype,
                    range=(vmin, vmax),
                    thresholds=thresholds,
                )
            )
        return defs
