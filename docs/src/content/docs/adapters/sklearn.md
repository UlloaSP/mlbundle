---
title: scikit-learn
description: Which scikit-learn estimators and transformers mlbundle can bundle with skops, what structure they carry and what is rejected.
---

Install `mlbundle[sklearn,schema]`. Models are stored with `skops.io.dumps` and loaded with
`skops.io.loads` against a trusted-types allowlist that lives inside mlbundle. The allowlist is
never read from a bundle, and you cannot extend it.

```python
from sklearn.ensemble import RandomForestRegressor

from mlbundle import export, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = RandomForestRegressor(n_estimators=3, random_state=0).fit(
    rows, [1.0, 2.0, 1.1, 1.9]
)
export(model, "forest.mlbundle", feature_names=["temperature", "humidity"])

bundle = load("forest.mlbundle")
assert bundle.predict(rows) == model.predict(rows).tolist()
structure = bundle.structure
print(structure["family"], structure["estimator_shape"])  # tree_ensemble [3]
```

## Supported estimators

Fitted classifiers that expose `predict_proba`, and regressors whose feature and output counts can
be recovered. Tested families include:

- decision trees, random forests, extra trees and gradient boosting;
- linear, regularized, robust and generalized linear models;
- SVMs, `PLSRegression`, naive Bayes, LDA and QDA;
- dense brute-force neighbors and `NearestCentroid`;
- `MLPClassifier`/`MLPRegressor`;
- `MultiOutputRegressor`/`MultiOutputClassifier` and `RegressorChain`.

Transformers that round-trip, including inside nested `ColumnTransformer` pipelines:
`SimpleImputer`, `KNNImputer`, scalers, `KBinsDiscretizer`, categorical encoders, univariate
selectors and `FunctionTransformer` with closed NumPy functions.

The exact class, attribute and function lists are in [Prediction state](/format/state/); the
tested configurations are in
[`tests/test_sklearn_families.py`](https://github.com/UlloaSP/mlbundle/blob/main/tests/test_sklearn_families.py).

## Structure

| Model | `structure.json` family |
| --- | --- |
| Single tree (`tree_`) | `decision_tree` |
| Forests, extra trees, gradient boosting | `tree_ensemble` |
| Models with `coef_`/`intercept_` | `linear` |
| Models with only `feature_importances_` | `feature_importances` |
| All-scikit-learn `Pipeline`, other families | `null` |

scikit-learn structures also carry a `prediction_state` commitment and `state_sha256`, so the
fitted state used for prediction is bound to the bundle. An all-scikit-learn pipeline has no
structure, because version 1 has no mapping from its final step back to the input features, but
its claims and state commitment still cover every step.

## Rejected

Export fails with `UnsupportedModelError`, listing the untrusted types where skops finds them:

- custom estimators and subclasses;
- functions and types outside the closed lists, including lambdas and your own functions;
- compound fitted dtypes;
- classifiers without `predict_proba`;
- multilabel and `ClassifierChain` layouts;
- untrusted nested storage, such as fitted sparse neighbor arrays;
- `HistGradientBoostingClassifier`/`Regressor`, whose `TreePredictor` storage is not on the
  allowlist.

You cannot opt in to trusting a rejected type. Rewrite the model with supported components, or keep
it out of the bundle.

## Versions

The allowlist is audited and tested against **scikit-learn 1.9.1 with skops 0.16.0**, not every
version the `>=1.6` range allows. Unknown fitted state fails. Use the same scikit-learn version for
export and load; a failed load under a different declared version raises
`EnvironmentMismatchError` to say so.
