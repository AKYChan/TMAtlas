import numpy as np
from typing import Any, List
from numpy.typing import NDArray


class Thermometer:
    """The standard TM binarizer as detailed in https://arxiv.org/pdf/1905.04199.pdf, Section 3.3,
    and is the same booleanisation used in the TMU (https://github.com/cair/tmu/tree/main) repo.

    Args:
        max_bits_per_feature: Number of bits / bins to allocate to each feature.
    """

    number_of_features: int
    max_bits_per_feature: int
    unique_values: List[np.ndarray]

    def __init__(self, max_bits_per_feature: int = 25):
        self.max_bits_per_feature = max_bits_per_feature
        return

    def fit(self, X: Any) -> None:
        """Determines the thermometers bins for the passed data.

        Args:
            X: 2D array-like of shape (n_samples, n_features).
        """
        X = np.asarray(X, dtype=np.float32)

        self.number_of_features = 0
        self.unique_values = []
        for i in range(X.shape[1]):
            uv = np.unique(X[:, i])[1:]

            if uv.size > self.max_bits_per_feature:
                unique_values = np.empty(0)

                step_size = 1.0 * uv.size / self.max_bits_per_feature
                pos = 0.0
                while (
                    int(pos) < uv.size
                    and unique_values.size < self.max_bits_per_feature
                ):
                    unique_values = np.append(unique_values, np.array(uv[int(pos)]))
                    pos += step_size
            else:
                unique_values = uv

            self.unique_values.append(unique_values)
            self.number_of_features += self.unique_values[-1].size
        return

    def transform(self, X: Any) -> NDArray[np.uint32]:
        """Transforms the input data to binary values according to the
        bins determined during fit.

        Args:
            X: 2D array-like of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=np.float32)
        X_transformed = np.zeros((X.shape[0], self.number_of_features), dtype=np.uint32)

        pos = 0
        for i in range(X.shape[1]):
            for j in range(self.unique_values[i].size):
                X_transformed[:, pos] = X[:, i] >= self.unique_values[i][j]
                pos += 1

        return X_transformed

    def fit_transform(self, X: Any) -> NDArray[np.uint32]:
        """Performs both the fit and transform methods in a single function

        Args:
            X: 2D array-like of shape (n_samples, n_features)
        """
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(
        self, feature_names: List[str] | None = None, style: str = "threshold"
    ) -> NDArray[str]:
        """
        Using the transformed boolean bins, names are assigned to each column to make the data easily sortable and readable.

        Args:
            feature_names: Optional custom names for the original input columns. If None, defaults to x0, x1, ...
            style: {"threshold", "range"}
                * "threshold" -> labels look like "x3 ≥ 7"
                * "range"     -> labels look like "x3 ∈ (-∞–7]".
                (still overlaps, because the underlying bits are cumulative)

        Returns:
            Numpy array which returns the feature name for each column of data.
        """
        if not hasattr(self, "unique_values"):
            raise RuntimeError("Call fit() before requesting feature names.")

        if style != "threshold" and style != "range":
            raise RuntimeError(
                f"{style} is not a supported style. Use 'threshold' or 'range'."
            )

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(self.unique_values))]

        names: list[str] = []
        for fname, thr in zip(feature_names, self.unique_values):
            thr_sorted = np.sort(thr)  # just to be sure
            if style == "threshold":
                # one-to-one match with transform() output
                names.extend([f"{fname} ≥ {t:g}" for t in thr_sorted])
            elif style == "range":
                # shows explicit intervals but note: still overlapping
                prev = "-∞"
                for t in thr_sorted:
                    names.append(f"{fname} ∈ ({prev}–{t:g}]")
                    prev = f"{t:g}"
            else:
                raise ValueError("style must be 'threshold' or 'range'")

        return np.asarray(names, dtype=str)

    def get_bits_per_feature(
        self,
        feature_names: List[str] | None = None,
    ) -> list[tuple[str, int]]:
        """
        Outputs a the numbner of bins for each feature as shown below:
            ┌────────────┬──────────────┐
            │ feature    │ n_bits       │
            ├────────────┼──────────────┤
            │ "x0"       │ 5            │
            │ "x1"       │ 4            │
            │    …       │ …            │
            └────────────┴──────────────┘
        Args:
            feature_names: Optional variable for providing the names of the features used. MUST be in the same order as the input data X.
        Returns:
            A 2D list of tuples of ordered features names with the number of bins for each feature.
        Note:
            The order of rows matches the order of columns in the original data X.
        """
        if not hasattr(self, "unique_values"):
            raise RuntimeError("Call fit() before requesting bit counts.")

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(self.unique_values))]

        counts = [len(thr) for thr in self.unique_values]  # k thresholds → k bits
        return list(zip(feature_names, counts))
