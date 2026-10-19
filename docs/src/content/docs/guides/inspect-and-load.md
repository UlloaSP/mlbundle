---
title: Inspect and load
description: Read a bundle's metadata without touching the model, or load it through a closed adapter that recomputes every model-derived claim.
---

mlbundle has two readers with different guarantees. Choose by what you need to trust.

| | `inspect` | `load` |
| --- | --- | --- |
| Archive layout, members, resource limits | Validated | Validated |
| Strict manifest and format version | Validated | Validated |
| SHA-256 and size of every non-manifest member | Verified | Verified |
| Schema and samples against manifest claims | Validated | Validated |
| Imports a model library | **Never** | The matching adapter only |
| Deserializes the model | **Never** | Through the closed adapter |
| Hyperparameters, counts, labels, dtypes, structure | Read as claims | **Recomputed and compared** |
| Returns | `MetadataView` | `LoadedBundle` with model and predictor |

Both accept a path, `bytes` or a binary stream, and an optional lowered
[`ArchiveLimits`](/reference/limits/). The ZIP is read in memory by allowed member names and never
extracted to disk.

## `inspect`: metadata only

```python
from mlbundle import inspect

view = inspect("trained.mlbundle")
manifest = view.manifest
print(manifest.format_version, manifest.writer_version)  # 1 0.2.1
print(manifest.model.n_features, manifest.model.n_outputs)  # 2 1
print([member.value for member in view.members])
# ['manifest.json', 'model.skops', 'schema.json', 'structure.json']
print(view.samples)  # None
```

`MetadataView` has `manifest`, `members`, `schema` and `samples`. Use it to list, index or review
bundles on a machine without model libraries.

:::caution[Inspection does not verify the model]
A bundle's author can write any manifest and recompute its hashes. `inspect` checks that the file is
well formed and internally consistent, not that the metadata, labels or structure describe the
model inside. Only `load` (or `mlbundle verify`) establishes that agreement.
:::

## `load`: a verified model

```python
from pathlib import Path

from mlbundle import load

bundle = load(Path("trained.mlbundle").read_bytes())  # paths, bytes or streams
print(bundle.library, bundle.class_name, bundle.kind)
# sklearn DecisionTreeClassifier classifier
print(
    bundle.feature_names, bundle.feature_origin
)  # ('temperature', 'humidity') declared
print(bundle.structure["family"])  # decision_tree
```

`load` runs every `inspect` check, then:

1. resolves the adapter from a closed list (`sklearn`, `xgboost`, `catboost`, `onnx`), never from a
   class name in the manifest;
2. deserializes with skops against mlbundle's own trusted types, native UBJSON or CBM in memory, or
   an ONNX CPU session after graph validation;
3. extracts claims and structure with the same code the writer used and compares them as canonical
   JSON bytes, raising `ClaimMismatchError` on any difference.

### `LoadedBundle`

| Member | Contents |
| --- | --- |
| `manifest`, `members`, `schema`, `samples` | As in `MetadataView`, validated. |
| `structure` | Decoded `structure.json`, or `None` for families without extractable structure. |
| `native_model` | The native object: estimator, `Pipeline`, XGBoost or CatBoost wrapper, or an ONNX model holding `data` and `session`. |
| `kind`, `library`, `class_name` | From the verified manifest. |
| `feature_names`, `feature_origin` | Ordered names and their origin. |
| `class_labels` | Output-major labels, or `None` for regressors. |
| `predict_proba(data)` | Classifiers only. One probability matrix per output; the matrix itself for one output. |
| `predict(data)` | Regressors only. Row-major values. |
| `row_from_record(record)`, `predict_record(record)` | See [Records and predictions](/guides/records/). |

Calling `predict_proba` on a regressor, or `predict` on a classifier, raises `ValueError`. Results
are plain Python lists. `native_model` stays available for native inference and explanations;
mutating it invalidates what `load` verified.

## Structure

`structure.json` describes the model a reviewer would visualise: decision trees, tree ensembles,
linear coefficients, feature importances, XGBoost tree dumps or CatBoost's symmetric trees. It is
required whenever the adapter can extract it and is compared on load, so a reviewer never sees a
tree that differs from the model used for prediction. All-scikit-learn pipelines and ONNX have
`None`. The families and fields are in [Adapters and structure](/format/adapters/#structurejson).

## Version diagnostics

Some failures add a hint without adding a compatibility gate:

- `EnvironmentMismatchError` when a skops load fails and the bundle declares a different
  scikit-learn version than the one installed;
- `InvalidManifestError` mentioning a different writer version when a manifest fails validation.

Format 1 is not frozen; incompatible changes may require re-export. Bundles written by 0.1.0a1, or
with schema version 1, must be re-exported.

## Treat every bundle as untrusted

Validation does not sandbox native parsers or inference. A runtime that accepts uploads still owns
authentication, process isolation and memory, CPU and row limits. See
[What is verified](/guides/trust/) and the [security policy](/security/).
