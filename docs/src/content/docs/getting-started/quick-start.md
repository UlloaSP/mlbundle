---
title: Quick start
description: Export a scikit-learn classifier to a .mlbundle file, inspect its metadata, load the verified model and predict.
---

This walkthrough uses scikit-learn. Install `mlbundle[sklearn,schema,lock]` on Python 3.14 as described
in [Installation](/getting-started/installation/).

## 1. Export on the training machine

```python title="train.py"
from sklearn.tree import DecisionTreeClassifier

from mlbundle import export

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = DecisionTreeClassifier(random_state=7).fit(rows, ["cold", "warm"] * 2)

view = export(model, "trained.mlbundle", feature_names=["temperature", "humidity"])
print([member.value for member in view.members])
# ['manifest.json', 'model.skops', 'schema.json', 'structure.json', 'pylock.toml']
```

`export` serializes the model with skops, loads it back, recomputes its claims and structure, and
only then publishes the file. It also locks the installed scikit-learn runtime as `pylock.toml` for
Linux and Windows serving, through uv and your package index. A model with types outside mlbundle's allowlist
fails here, on the training machine, not after upload. An existing destination needs `overwrite=True`.

The model was fitted on plain lists, so it has no native feature names: the names you pass are
recorded as `declared`. No training data was given, so the schema comes from one synthetic row.

## 2. Inspect without loading the model

```python
from mlbundle import inspect

view = inspect("trained.mlbundle")
print(view.manifest.model.library, view.manifest.model.estimator_class)
# sklearn DecisionTreeClassifier
print(view.manifest.features.names, view.manifest.features.origin)
# ['temperature', 'humidity'] declared
print(view.schema["reports"][0]["labels"])  # ['cold', 'warm']
```

`inspect` validates the archive, the manifest, every non-manifest member's hash, the lock and the schema, without importing
scikit-learn or deserializing anything. It therefore **cannot** tell you whether the metadata
describes the model. The command line does the same:

```bash
mlbundle inspect trained.mlbundle
```

## 3. Load and predict

```python
from mlbundle import load

bundle = load("trained.mlbundle")
print(bundle.kind, bundle.class_name, bundle.class_labels)
# classifier DecisionTreeClassifier [['cold', 'warm']]
print(bundle.predict_proba([[0.0, 1.0]]))  # [[1.0, 0.0]]
```

`load` performs every `inspect` check, deserializes through the scikit-learn adapter, recomputes
the hyperparameters, counts, labels and tree, and raises `ClaimMismatchError` if any of them
differs from the manifest or `structure.json`. Classifiers use `predict_proba`, regressors
`predict`; both return plain lists. Probabilities are output-major: one matrix per output, which
for a single output is the matrix itself.

`bundle.native_model` is the real `DecisionTreeClassifier`, for native inference or explanations.
Mutating it invalidates what was verified.

## 4. Predict from a record

Forms and APIs send values keyed by name. Records use the schema's mappings: positions for this
model, because its names were declared rather than native.

```python
result = bundle.predict_record({"0": "0.0", "1": "true"})[0]
print(result.labels, result.probabilities, result.predicted_label)
# ('cold', 'warm') (1.0, 0.0) cold
```

Strings are parsed strictly, every feature is required, and `predicted_label` is the model's own
class decision. See [Records and predictions](/guides/records/).

## 5. Verify from the command line

```bash
mlbundle verify trained.mlbundle
```

```text
Valid bundle: sklearn "DecisionTreeClassifier"; verified format, integrity, schema/samples and native model claims/structure.
```

## Next steps

- Use your real training frame for dtypes and categories: [Schema and samples](/guides/schema-and-samples/).
- Bundle XGBoost, CatBoost, ONNX or a pipeline: [Supported models](/adapters/overview/).
- Understand what each check proves: [What is verified](/guides/trust/).
