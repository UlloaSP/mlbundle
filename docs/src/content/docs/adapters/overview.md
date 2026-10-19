---
title: Supported models
description: The four model libraries mlbundle supports, how each is stored, what structure it carries and where the support boundary lies.
---

mlbundle supports four libraries through closed adapters, plus scikit-learn pipelines that end in
XGBoost or CatBoost. There is no conversion between formats and no generic fallback: a model
outside an adapter's boundary fails on export, on the training machine.

| Library | Extra | Stored as | Bundle member | Structure | Class labels |
| --- | --- | --- | --- | --- | --- |
| [scikit-learn](/adapters/sklearn/) | `sklearn` | skops, closed allowlist | `model.skops` | Trees, ensembles, linear, importances; `null` for other families | Native `classes_` |
| [XGBoost](/adapters/xgboost/) | `xgboost` | Native UBJSON | `model.ubj` | Complete JSON tree dumps | Supplied labels bound to a native copy, or encoded indices |
| [CatBoost](/adapters/catboost/) | `catboost` | Native CBM | `model.cbm` | Complete symmetric trees and borders | Native `classes_` |
| [ONNX](/adapters/onnx/) | `onnx` | Your file, byte for byte | `model.onnx` | `null` | Graph labels, or count-checked declarations |
| [Pipelines](/adapters/pipelines/) | `sklearn` + native | skops prefix + native leaf | `model.skops` + `model.ubj`/`model.cbm` | Leaf structure with the transformed boundary | From the final estimator |

Every adapter owns recognition, export preparation, serialization, loading, claim and structure
extraction, and prediction. The same extraction code runs on export and on load, so the format is
defined once.

## Shared limits

- XGBoost, CatBoost and ONNX models have exactly one output. scikit-learn supports up to 1,024.
- At most 100,000 features and 100,000 classes per output.
- Classifiers must produce probabilities; there is no decision-only classifier support.

## Explicitly out of scope

mlbundle does not support:

- other libraries, including LightGBM and PyTorch;
- converting scikit-learn models to ONNX, or any other format conversion;
- per-model environments and bundle signing;
- pickle, joblib or cloudpickle, even as an option.

The normative mappings are in [Adapters and structure](/format/adapters/),
[Prediction state](/format/state/), [CatBoost mapping](/format/catboost/),
[ONNX mapping](/format/onnx/) and [Composite pipelines](/format/pipeline/).
