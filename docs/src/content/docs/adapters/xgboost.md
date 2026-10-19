---
title: XGBoost
description: Bundle XGBClassifier and XGBRegressor as native UBJSON, bind original class labels and know which settings survive.
---

Install `mlbundle[xgboost,schema]`. Models are stored as native in-memory UBJSON and loaded with
`load_model`, never pickled.

```python
from xgboost import XGBClassifier

from mlbundle import export, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(rows, [0, 1, 0, 1])
export(model, "xgboost.mlbundle", class_labels=[["cold", "warm"]])

bundle = load("xgboost.mlbundle")
assert isinstance(bundle.native_model, XGBClassifier)
assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
print(bundle.class_labels)  # [['cold', 'warm']]
```

## Class labels

XGBoost trains on encoded indices. Supply the original labels in probability-column order, nested
even for one output. mlbundle binds them, with the prediction settings, to an attribute on a
**copy** of the booster; your model object is not changed. Load verifies that the binding is
consistent, not how the model was trained.

- Without `class_labels=`, encoded indices are used, or an existing binding.
- The native model's `predict` keeps returning encoded indices; `predicted_label` in
  [records](/guides/records/) maps the decision to your label.
- Regressors reject `class_labels=`.

## Supported

- Exact `XGBClassifier` and `XGBRegressor`, one target.
- Native `gbtree` storage, including XGBoost 3.4.1's `dart` option, which uses `gbtree` storage.
- Classifier objectives `binary:logistic` and `multi:softprob`.
- Regression objectives `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`,
  `reg:squaredlogerror`, `reg:logistic`, `count:poisson`, `reg:gamma` and `reg:tweedie`.
- At most 100,000 trees and depth 14.

Structure is the complete list of JSON tree dumps with statistics, plus the native model record.

## Rejected

Raw `Booster` objects, `XGBRFClassifier`/`XGBRFRegressor`, subclasses, ranking, survival and custom
objectives, softmax-only objectives, `gblinear`, legacy `dart` storage and multiple targets.

## What survives

Only configuration that native XGBoost I/O preserves is recorded and verified: the learner model
parameters, objective, booster, the bound prediction settings (`missing`, `importance_type`,
`enable_categorical`) and best-iteration attributes. Learning rate, max depth, seed and thread count
do not survive native I/O and are neither recorded nor reconstructed.

The exact mapping is in [Implemented XGBoost mapping](/format/adapters/#implemented-xgboost-mapping).
