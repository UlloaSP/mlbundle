---
title: Schema and samples
description: How mlbundle infers schema.json locally with mlschema, what it reveals about the training data, and how to include sample rows explicitly.
---

Every bundle carries `schema.json`: the input fields a form or API needs, plus one report per
model output. It is inferred **on the training machine** by
[mlschema](https://docs.mlschema.eshkra.dev/). The training DataFrame itself never enters the
bundle.

## Infer from your training frame

Pass the frame you trained on. With `target_names`, those columns are excluded from inference,
dtypes and samples; without it, every column must be a feature.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlbundle import export

training = pd.DataFrame(
    {
        "age": [34, 51, 29, 46],
        "plan": pd.Categorical(
            ["basic", "pro", "basic", "team"],
            categories=["basic", "pro", "team", "free"],
        ),
        "color__red": [1, 0, 0, 1],
        "color__blue": [0, 1, 1, 0],
        "income": [52000.0, None, 61000.0, 48000.0],
        "churned": ["no", "yes", "no", "yes"],
    }
)
features = training.drop(columns="churned")
encoded = features.assign(
    plan=features["plan"].cat.codes, income=features["income"].fillna(0)
)
model = DecisionTreeClassifier(random_state=0).fit(
    encoded.to_numpy(), training["churned"]
)

view = export(model, "churn.mlbundle", training_data=training, target_names=["churned"])
print(view.manifest.features.origin)  # dataframe
print(
    view.manifest.features.dtypes
)  # ['int64', 'category', 'int64', 'int64', 'float64']
```

The resulting `schema.json`, keys sorted:

```json
{
  "schema_version": "2",
  "source": "training_data",
  "onehot_separator": "__",
  "fields": [
    { "kind": "number", "label": "age", "mappedTo": 0, "required": true, "step": 1.0 },
    {
      "kind": "category",
      "label": "plan",
      "mappedTo": 1,
      "options": ["basic", "pro", "team", "free"],
      "required": true
    },
    {
      "kind": "onehot-category",
      "label": "color",
      "options": [
        { "label": "red", "mappedTo": 2, "value": "red" },
        { "label": "blue", "mappedTo": 3, "value": "blue" }
      ],
      "required": true
    },
    { "kind": "number", "label": "income", "mappedTo": 4, "required": false, "step": 0.1 }
  ],
  "reports": [
    {
      "kind": "classifier",
      "label": "Predicted class",
      "labels": ["no", "yes"],
      "mappedTo": 0,
      "showClassProbabilities": true
    }
  ]
}
```

What happened:

- `color__red` and `color__blue` are binary columns sharing a prefix before the separator, so they
  became one `onehot-category` field. Change the separator with `onehot_separator=` (1–16
  characters); it is stored in the schema.
- `income` has a missing value, so it is `required: false`.
- `plan` keeps its categorical order, **including the unused `free`**.
- The model was fitted on an array, so mappings are zero-based positions. A model with native
  feature names maps by name instead.
- The report labels are the string form of the model's classes, in probability-column order.

Frame columns must be unique strings, in model input order, and must equal native names when the
model has them. `feature_names=` cannot rename or reorder a supplied frame.

## Without training data

Without `training_data`, mlschema infers from one synthetic row of `1.0` values, one column per
feature. Fields become required `number` fields with `step: 0.1`, except that named columns sharing
a prefix before the separator can still be grouped as one-hot. `source` is `"default"` and the
training dtypes are `null`: a synthetic row is not evidence about your data.

## What leaves the training machine

| Included | Not included |
| --- | --- |
| Feature names and labels | Training rows (unless you request samples) |
| Field kinds and nullability | Numeric or date minima, maxima, means |
| Category values, including unused ones | Row counts |
| One-hot names and suffixes | Example values, defaults, free-text values |
| Dtype strings and dtype-based numeric steps | The DataFrame index |
| Pair-axis names for series fields | |
| Class labels | |

These declarations can still be sensitive: column and category names reveal what you trained on.
The model, its learned structure and your `metadata=` can independently reveal training
information. Leaving out samples does not prevent model memorization.

## Explicit samples

Samples are **real user data**, so they are only ever included when you ask. `sample_rows=N`
includes exactly the first N feature rows of the frame, without target columns or the index.

```python
from mlbundle import inspect

export(model, "sampled.mlbundle", training_data=features, sample_rows=2)
print(inspect("sampled.mlbundle").samples)
# [{'age': 34, 'color__blue': 0, 'color__red': 1, 'income': 52000.0, 'plan': 'basic'},
#  {'age': 51, 'color__blue': 1, 'color__red': 0, 'income': None, 'plan': 'pro'}]
```

- N must be a positive integer within the frame; `0`, `True` and larger values raise
  `InvalidSamplesError`, as does `sample_rows` without `training_data`.
- There is no random sampling or clipping. Shuffle the frame yourself if the first rows are not
  representative.
- Missing values become `null`, timestamps ISO 8601 strings, categorical and NumPy scalars plain
  JSON values. Bytes, complex values, timedeltas and arbitrary objects are rejected.
- Schema and samples are limited to 16 MiB each and JSON depth 32.

## Reading schemas

`inspect` and `load` validate `schema.json` without mlschema or pandas: every feature is covered
exactly once, mappings match the feature origin, and reports match the manifest's class labels.
`load` additionally verifies those labels against the model. The schema's `source`, separator and
nullability remain declarations; nothing can attest how the training data looked.

Bundles written by 0.1.0a1 without a schema, with schema version `1` or, since 0.3.0, without
`pylock.toml` must be re-exported.

The full rules, including series fields and cell conversions, are in
[Schema and samples](/format/schema/).
