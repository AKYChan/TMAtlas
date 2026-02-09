# Examples

Here are a few examples for running and using the TMAtlas tools. Currently we only have a single classification example with more to be added in the future.

## Classification

A simple iris example that will train a TM model with Optuna and return a JSON and CSV with the best performing explained model.

### Requirements

`optuna
scikit-learn
tmu`

### Quick Start

I recommend using uv to run everything as it makes installation simple, otherwise:

`python3 iris_classification.py`
