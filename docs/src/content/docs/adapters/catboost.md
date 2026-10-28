---
title: CatBoost
description: Bundle numeric, single-target symmetric CatBoost models as native CBM with complete, verified tree structure.
---

Install `mlbundle[catboost,schema,lock]` (CatBoost 1.2.10 or newer; it brings matplotlib and plotly).
Models are stored as native CBM and loaded in memory.

```python
from catboost import CatBoostClassifier

from mlbundle import export, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = CatBoostClassifier(
    iterations=3, depth=2, verbose=False, allow_writing_files=False
)
model.fit(rows, ["cold", "warm"] * 2)
export(model, "catboost.mlbundle")

bundle = load("catboost.mlbundle")
print(bundle.class_labels)  # [['cold', 'warm']]
assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
```

CatBoost owns its labels: `classes_` is verified on load and `class_labels=` is rejected, even when
equal.

## Supported

- Exact fitted `CatBoostClassifier` and `CatBoostRegressor`, one target, numeric features.
- `SymmetricTree` (oblivious trees) only.
- Classifier losses `Logloss`, `CrossEntropy` and `MultiClass`.
- Regressor losses `RMSE`, `MAE`, `Quantile`, `Huber`, `Poisson` and `MAPE`, including their
  native parameter strings.

Complete trees, feature borders, NaN handling, scale and bias, and feature importances are carried
in `structure.json` and verified on load. `predicted_label` uses CatBoost's own class prediction,
including its threshold.

## Rejected

`class_labels=`; categorical, text and embedding features; `Depthwise` and `Lossguide` growth;
`MultiClassOneVsAll` and other losses; multi-target and uncertainty models; rankers, bare
`CatBoost`, subclasses; custom Python objectives or metrics; unknown wrapper attributes.

Categorical features round-trip natively in CBM, but their CTR storage has an unchecked native
loader, so mlbundle rejects them before export and again before native load. Encode categories
yourself, or use a scikit-learn prefix in a [composite pipeline](/adapters/pipelines/).

The wrapper recovery, native-parser limits and in-memory serializer are specified in
[CatBoost mapping](/format/catboost/).
