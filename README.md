# TMAtlas — An Interactive Visual Analytics Framework for Explaining Tsetlin Machine Outputs

A Python package for exporting trained Tsetlin Machine models into structured, serialisable formats. It separates **model introspection** (extracting clauses, weights, features) from **export** (JSON, CSV), making it straightforward to add support for new model types or output formats.

## Installation

`pip install -e tmatlas`

Your trained model should come from the [`tmu`](https://github.com/cair/tmu) library.

The thermometer binarizer provided in this repo, has been adapted from the StandardBinarizer inside of [`tmu`](https://github.com/cair/tmu).

To add your own binarizer the following functions are needed to allow for seamless integration into TMAtlas.

- `get_feature_names_out(feature_names)` — returns a list of boolean literal descriptions (e.g. `"temperature ≥ 25.0"`)
- `unique_values` — a list of arrays, one per feature, containing the binarization thresholds

## Package Structure

```
src/
└── tmatlas/
    ├── booleanisation/
    │   └── thermometer.py          # Adapted thermometer encoding binarizer for TMAtlas
    ├── utils/
    │   └── types.py                # Shared dataclasses (Clause, Literal, FeatureClass, etc.)
    ├── inspectors/
    │   ├── base.py                 # Abstract base — subclass to support new model types
    │   ├── features.py             # Feature/binarizer extraction (model-agnostic)
    │   └── tmu_inspector.py        # Concrete introspector for tmu models
    └── exporters/
        ├── base.py                 # Abstract base — subclass to add new output formats
        ├── json.py                 # Exports full model to a JSON-ready dict
        └── csv.py                  # Exports predictions + clause activations to CSV
```

## Quick Start

### Model support

**VERY IMPORTANT**
TMAtlas currently only supports the Coalesced Classification model from TMU and the Regression model from TMU.

### 1. JSON Model Export

This is the primary use case: exporting a trained model's full structure (features, clauses, weights, metadata) to a JSON-serialisable dictionary.

```python
import json
import numpy as np
from tmatlas.inspectors.features import FeatureInspector
from tmatlas.inspectors.tmu_inspector import TMUInspector
from tmatlas.exporters.json import JsonExporter

# --- Prerequisites ---
# model:         a trained tmu TMRegressor or TMClassifier
# binarizer:     a fitted binarizer (e.g. StandardBinarizer)
# X_train:       the original (non-binarized) training data as a numpy array
# feature_names: list of column names matching X_train's columns

# Step 1: Build the feature inspector
feat_ext = FeatureInspector(binarizer, X_train, feature_names)

# Step 2: Create the inspector
#   For regression — omit `classes`:
inspector = TMUInspector(model, feat_ext)

#   For classification — pass class names:
inspector = TMUInspector(model, feat_ext, classes=["cat", "dog"])

# Step 3: Export
data = JsonExporter(inspector, feat_ext).export()

# Step 4: Save to file
with open("model_export.json", "w") as f:
    json.dump(data, f, indent=2)
```

#### What the JSON contains

| Key        | Description                                                                |
| ---------- | -------------------------------------------------------------------------- |
| `model`    | Model type (`"regression"` / `"classification"`), task, and class names    |
| `metadata` | Hyperparameters (`T`, `s`, `numClauses`), timestamp, and y-range           |
| `features` | Per-feature definitions: name, type, range, and binarization thresholds    |
| `clauses`  | Every clause with its literals, thresholds, and per-class weights/polarity |

#### Clause weight structure

Weights include polarity derived from the sign, which matters for coalesced classification models where the same clause can vote positively for one class and negatively for another:

```json
// Classification — polarity varies per class
"weights": {
  "cat": {"value": 3.0, "polarity": "positive"},
  "dog": {"value": -2.0, "polarity": "negative"}
}

// Regression — weights are always positive
"weights": {"value": 5.0, "polarity": "positive"}
```

### 2. CSV Prediction Export

Export input features alongside actual values, predictions, and which clauses activated for each sample. This is useful for debugging or downstream analysis.

```python
import numpy as np
import pandas as pd
from tmatlas.exporters.csv import export_data_to_csv, format_clause_activations

# --- Requirements ---
# X_test must be a pandas DataFrame (column names become CSV headers)
# y_actual and y_predicted must be numpy arrays
# clause_activations is the output of model.transform(X_test_binarized)

X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Convert the raw activation matrix to a list-of-dicts format
activations = model.transform(X_test_binarized)
clause_list = format_clause_activations(activations)

# Write CSV
export_data_to_csv(
    output_filename="predictions.csv",
    X_data=X_test_df,          # must be a DataFrame
    y_actual=y_actual,          # numpy array
    y_predicted=y_predicted,    # numpy array
    activated_clauses_list=clause_list,
)
```

The resulting CSV has one row per sample with columns: `Sample`, all feature columns, `Actual`, `Predicted`, and `Activated_Clauses` (semicolon-separated clause IDs).

## Extending the Package

### Adding a new model type

Subclass `BaseInspector` and implement three methods. The exporters don't need to change.

```python
from tmatlas.inspectors.base import BaseInspector
from tmatlas.utils.types import Clause, ModelInfo, ModelMetadata

class MyCustomInspector(BaseInspector):
    def get_model_info(self) -> ModelInfo: ...
    def get_metadata(self) -> ModelMetadata: ...
    def get_clauses(self) -> list[Clause]: ...
```

Then use it with any exporter:

```python
inspector = MyCustomInspector(...)
data = JsonExporter(inspector, feat_ext).export()
```

### Adding a new export format

Subclass `BaseExporter` and implement `export()`. The inspection layer doesn't need to change.

```python
from tmatlas.exporters.base import BaseExporter

class YamlExporter(BaseExporter):
    def export(self):
        clauses = self.introspector.get_clauses()
        # ... format as YAML ...
```

## Common Pitfalls

- **`X` must be a raw numpy array** (pre-binarization) when passed to `FeatureExtractor`. Don't pass the binarized version — feature ranges and types are derived from the original data.
- **`X_data` must be a pandas DataFrame** for `export_data_to_csv`. Column names are used as CSV headers. You'll get a `TypeError` if you pass a numpy array.
- **`feature_names` length must match `X.shape[1]`**. If omitted, defaults to `x0`, `x1`, … which works but is less readable.
- **Classification models need `classes`**. Pass them explicitly or ensure your model has a `classes_` attribute (scikit-learn convention). Without classes, the introspector assumes regression.
- **Binarizer compatibility**. Your binarizer must have both `get_feature_names_out()` and `unique_values`. The `Thermometer` from `booleanisation/` satisfies this.
