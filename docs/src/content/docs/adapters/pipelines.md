---
title: Composite pipelines
description: Bundle a fitted scikit-learn Pipeline that ends in XGBoost or CatBoost, storing the prefix with skops and the final estimator natively.
---

A fitted `sklearn.pipeline.Pipeline` whose final step is an exact XGBoost or CatBoost wrapper is
exported as it is. The transformer prefix is stored with skops, the final estimator with its native
adapter, and load rebuilds a real `Pipeline` after verifying both parts. Install
`mlbundle[sklearn,xgboost,schema]` or `mlbundle[sklearn,catboost,schema]`.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mlbundle import export, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = make_pipeline(StandardScaler(), XGBClassifier(n_estimators=3, n_jobs=1))
model.fit(rows, [0, 1, 0, 1])
export(model, "pipeline.mlbundle", class_labels=[["cold", "warm"]])

bundle = load("pipeline.mlbundle")
assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
print([member.value for member in bundle.members])
# ['manifest.json', 'model.skops', 'model.ubj', 'schema.json', 'structure.json']
```

## How it is stored

| Part | Member | Verified on load |
| --- | --- | --- |
| Transformer prefix | `model.skops` | Step names, order, options and a fitted-state commitment |
| Final XGBoost estimator | `model.ubj` | The full XGBoost claims and tree structure |
| Final CatBoost estimator | `model.cbm` | The full CatBoost claims and tree structure |

A single-step pipeline stores only the native leaf. The manifest reports the library as `sklearn`
with serialization `pipeline`, and `model.pipeline` records the steps, the final estimator's claims
and the **transformed-feature boundary** between them. All-scikit-learn pipelines are not composite:
they use one skops member and have no structure.

## Labels, features and structure

- Labels follow the final estimator: XGBoost uses the labels you bind with `class_labels=`, and its
  native `classes_`/`predict` stay encoded; CatBoost keeps its native labels.
- Manifest features, schema, training dtypes and samples describe the **original inputs**.
- Trees address the **transformed features**, without attribution back to the original columns.
  `mlbundle inspect` prints the boundary's width and names.
- Six members are enough for manifest, prefix, leaf, structure, schema and explicit samples.

## Supported prefixes

Scalers, `PCA`, `SimpleImputer` (with indicators and empty columns), `KBinsDiscretizer`,
`PolynomialFeatures`, univariate selectors, `OneHotEncoder`, `OrdinalEncoder`, identity or closed
NumPy `FunctionTransformer`, nested native-free `Pipeline` and `ColumnTransformer`. Each needs a
fitted, data-free proof of its output width.

## Rejected

Transformers whose width cannot be proven, `FeatureUnion`, `KNNImputer`, slice, callable or scalar
column selectors, pipeline caching (`memory`), metadata routing, and native estimators anywhere
other than the final step.

Step names and outer options are reconstruction declarations; no name from the manifest ever
selects loading code. The full representation is in [Composite pipelines](/format/pipeline/).
