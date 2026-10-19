---
title: Records and predictions
description: Predict from arrays or from form-style records keyed by the bundle's schema, and read the frozen per-output results.
---

A loaded bundle predicts from two kinds of input: arrays in feature order, and single records keyed
by the schema. Both go through the same verified predictor.

## Arrays and DataFrames

```python
from mlbundle import load

bundle = load("trained.mlbundle")
print(bundle.predict_proba([[0.0, 1.0], [1.0, 0.0]]))  # [[1.0, 0.0], [0.0, 1.0]]
```

- Classifiers use `predict_proba`; results are **output-major**: one matrix per output, or the
  matrix itself when there is one output.
- Regressors use `predict`; results are row-major values.
- For models without native feature names, pass a 2-D NumPy array or nested numeric rows in
  manifest feature order. Declared names do not enable name-based selection.
- Adapter-supported DataFrames and sparse arrays keep their native restrictions.

## Records

`predict_record` takes one mapping, as a form or JSON API would send it, and returns one frozen
`OutputPrediction` per output.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlbundle import export, load

frame = pd.DataFrame(
    {"temperature": [0.0, 1.0, 0.1, 0.9], "humidity": [1.0, 0.0, 0.9, 0.1]}
)
model = DecisionTreeClassifier(random_state=7).fit(frame, ["cold", "warm"] * 2)
export(model, "named.mlbundle", training_data=frame)

bundle = load("named.mlbundle")
(result,) = bundle.predict_record(
    {"temperature": "0.05", "humidity": 0.95, "note": "ignored"}
)
print(result.labels, result.probabilities, result.predicted_label)
# ('cold', 'warm') (1.0, 0.0) cold
```

### Keys

| Feature origin | Keys |
| --- | --- |
| `model` (native names) | Feature names. |
| `dataframe`, `declared`, `generated` | Zero-based positions, as integers (`0`) or canonical decimal strings (`"0"`). Supplying both forms for one position is an error. |

Every feature is required. Extra keys are ignored **without iterating the record**, so the work
depends on the feature count, not the size of what a client sent.

### Values

- Finite Python `int`, `float` and `bool` values.
- Strings, stripped of ASCII whitespace and at most 64 characters: `true`/`false` in any case, or
  decimals such as `42`, `-1.5`, `.5`, `1.` and `2e-3`.
- `None` becomes NaN only when the schema field has `required: false`; for one-hot options the
  group's requirement applies.

Category fields still need the numeric value the model was trained on. Records convert strings to
numbers; they do not encode category names.

`row_from_record` returns the validated numeric row without predicting, for example to feed an
explainer with the same input:

```python
print(bundle.row_from_record({"temperature": 0.2, "humidity": " 1e0 "}))  # [0.2, 1.0]
```

### Errors

Invalid records raise `PredictionInputError`, which is also a `ValueError`. Messages name the key,
escaped and bounded, but **never echo the value**:

```text
Prediction record must be a mapping.
Missing features: "humidity".
Feature "temperature": a required feature cannot be None.
Feature "temperature": number must be finite.
Feature "temperature": expected a decimal or true/false string.
Feature 0: both integer and string keys are present.
```

Missing-key errors list at most five keys and then a count of the rest.

## `OutputPrediction`

```python
@dataclass(frozen=True)
class OutputPrediction:
    output: int  # report position, in output order
    kind: ModelKind  # classifier or regressor
    labels: tuple[str, ...] | None  # report labels, in probability order
    probabilities: tuple[float, ...] | None
    value: float | None  # regression value
    predicted_label: str | None
```

`predicted_label` is the report label of the **model's own class decision**: scikit-learn
`predict`, the XGBoost wrapper's `predict`, CatBoost's class prediction including its threshold,
or the ONNX graph's label output. It can differ from the label with the largest probability, and
mlbundle never substitutes an argmax. It is `None` for regressors and for ONNX classifiers without
a label output.

```python
regression = load("reg.mlbundle")  # a LinearRegression bundle
print(regression.predict_record({"temperature": 0.5, "humidity": 0.5}))
# (OutputPrediction(output=0, kind=<ModelKind.REGRESSOR: 'regressor'>, labels=None,
#   probabilities=None, value=2.5, predicted_label=None),)
```

## Pipelines that select columns by name

scikit-learn pipelines whose fitted column selectors use names receive a one-row pandas DataFrame
from `predict_record`; this needs pandas, which the `schema` extra installs. Other models receive a
numeric row, preserving their native categorical handling.

The normative record rules are in [Runtime record input](/format/schema/#runtime-record-input).
