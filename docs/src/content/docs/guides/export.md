---
title: Export a bundle
description: Every export argument, how feature names and class labels are resolved, what the manifest records, and how the file is published.
---

`export` runs on the training machine. It rejects unsupported models there, before any file
exists, and publishes a bundle only after it has been read back and verified.

```python
def export(
    model: object,
    destination: str | PathLike[str],
    *,
    training_data: object | None = None,
    feature_names: Sequence[str] | None = None,
    class_labels: Sequence[Sequence[Label]] | None = None,
    target_names: Sequence[str] | None = None,
    sample_rows: int | None = None,
    onehot_separator: str = "__",
    metadata: Mapping[str, JsonValue] | None = None,
    environment_lock: bytes | str | PathLike[str] | None = None,
    overwrite: bool = False,
    limits: ArchiveLimits | None = None,
) -> MetadataView
```

Export always needs the `schema` extra, even without `training_data`, and the `lock` extra unless
you pass `environment_lock=`.

## Arguments

| Argument | Meaning |
| --- | --- |
| `model` | A fitted, supported model: see [Supported models](/adapters/overview/). For ONNX, a path, bytes or a `ModelProto`. |
| `destination` | Path of the `.mlbundle` file to write. |
| `training_data` | Optional nonempty pandas DataFrame, used locally for schema inference and dtypes. Never stored. |
| `feature_names` | Names for models without native names. Ignored in favour of native names; must match the frame's columns when a frame is given. |
| `class_labels` | Output-major original labels, nested even for one output. For XGBoost, ONNX (verified against graph labels when present, count-checked otherwise) and composites ending in XGBoost. Rejected by scikit-learn, CatBoost and regressors. |
| `target_names` | One unique name per output, in output order. With a frame, these columns are excluded. A bare string is rejected. |
| `sample_rows` | Positive integer: include the first N feature rows as `samples.json`. Requires a frame. See [samples](/guides/schema-and-samples/#explicit-samples). |
| `onehot_separator` | Separator for one-hot groups, 1–16 characters, stored in the schema. |
| `metadata` | Free JSON object recorded as `user_metadata`. |
| `environment_lock` | Canonical `pylock.toml` bytes or a path to use instead of generating one, such as another bundle's `environment_lock.to_bytes()`. See [the runtime lock](#the-runtime-lock). |
| `overwrite` | Replace an existing destination. Without it, an existing file raises `FileExistsError`. |
| `limits` | Lower [`ArchiveLimits`](/reference/limits/) for this bundle. Ceilings cannot be raised. |

`export` returns the same [`MetadataView`](/reference/api/#metadataview) that `inspect` would
produce for the new file.

## Feature names and origin

Every bundle records an ordered list of feature names and where they came from:

| Origin | When | Names | Record keys and schema mappings |
| --- | --- | --- | --- |
| `model` | The model exposes native names, such as `feature_names_in_` after fitting on a DataFrame. | Native names | Names |
| `dataframe` | No native names, `training_data` supplied. | Frame columns | Zero-based positions |
| `declared` | No native names, `feature_names=` supplied. | Your names | Zero-based positions |
| `generated` | None of the above. | `feature_0` … `feature_{N-1}` | Zero-based positions |

Native names always win. Declared names are labels: they do not add name-based prediction or
reordering. For models without native names, predict with a 2-D NumPy array or nested numeric rows
in manifest feature order.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlbundle import export

frame = pd.DataFrame(
    {"temperature": [0.0, 1.0, 0.1, 0.9], "humidity": [1.0, 0.0, 0.9, 0.1]}
)
model = LogisticRegression().fit(frame, ["cold", "warm"] * 2)

view = export(model, "named.mlbundle", training_data=frame, overwrite=True)
print(view.manifest.features.origin, view.manifest.features.names)
# model ['temperature', 'humidity']
print(view.manifest.features.dtypes)  # ['float64', 'float64']
```

## Class labels and targets

Class labels are **output-major**: a list per output, in probability-column order. Where they come
from depends on the adapter:

- **scikit-learn and CatBoost** own their `classes_`; `class_labels=` is rejected.
- **XGBoost** is trained on encoded indices. Supply the original labels with
  `class_labels=[["cold", "warm"]]`; they are bound to a native copy of the model and verified on
  load. Without them, encoded indices are used. See [XGBoost](/adapters/xgboost/).
- **ONNX** labels from the graph are verified; supplied labels must agree. A graph without labels
  accepts count-checked declarations. See [ONNX](/adapters/onnx/).
- **Composites** delegate labels to the final native estimator.

Regressors reject `class_labels=`. `target_names=` is a separate declaration that names each output,
and with a training frame tells mlbundle which columns to exclude from inference, dtypes and samples.

## What the manifest records

`manifest.json` holds:

- identity: format name and version, a random `bundle_id`, `created_at`, `writer_version`;
- model: library, class, kind, hyperparameters, feature/output/class counts and serialization;
- features: ordered names, origin, training dtypes and verified model dtypes;
- target: output-major class labels and target names;
- environment: Python version, implementation and free-threading, OS and version, and
  architecture;
- integrity: the SHA-256 and size of every other member;
- `user_metadata`: your `metadata=`, at most 1,024 top-level keys, 64 KiB and depth 8.

Model-derived values, including model-exposed `model_dtypes`, are claims that `load` recomputes.
The environment, the lock, training `dtypes` and user metadata are declarations: recorded, never authenticated. The exact fields are in the
[format 1 manifest](/format/#manifest).

```python
view = export(
    model,
    "described.mlbundle",
    training_data=frame,
    metadata={"name": "weather", "metrics": {"accuracy": 1.0}},
    overwrite=True,
)
print(view.manifest.user_metadata["name"])  # weather
```

## The runtime lock

Every bundle contains `pylock.toml`, a PEP 751 lock of what serving the model needs: pydantic, the
model libraries, pandas for scikit-learn pipelines whose `ColumnTransformer` selects columns by name,
and their dependencies, for CPython with the GIL on Linux `x86_64` and `aarch64`
(glibc 2.28 or newer) and on Windows `amd64`. The manifest's `writer_version` pins mlbundle itself.

Export pins those distributions at their installed versions, constrains every other installed
distribution to its installed version and asks uv for one wheel per package and target. Every
locked package installed on the training machine must keep its installed version; packages needed
on other targets, such as NCCL for XGBoost on Linux, come from uv's resolution. Export therefore needs uv (the
`lock` extra) and access to the package index uv is configured to use.

```python
view = export(model, "locked.mlbundle", training_data=frame, overwrite=True)
lock = view.environment_lock
print(lock.python_version, lock.version("scikit-learn"))
print(lock.environment_id)  # SHA-256 identity of Python and every wheel
```

Offline machines, or machines whose interpreter cannot run uv, can pass a lock generated for the same
installed versions elsewhere: `export(..., environment_lock="pylock.toml")`. The lock fixes versions
and wheel files, not the binaries that trained the model. The rules are in
[the lock contract](/format/environment/).

## Publication

The writer builds a deterministic ZIP in memory, validates it with the reader, then publishes it
atomically with a hard link or replacement. Where hard links are unavailable it falls back to
exclusive creation, which removes a partial file on failure but exposes the destination while
copying.

## Export failures

Validation failures happen before anything is published (with the exclusive-creation fallback, a
failed copy is removed but was briefly visible):

| Error | Typical cause |
| --- | --- |
| `UnsupportedModelError` | The model class, objective, layout or a contained type is outside the adapter's closed boundary. For scikit-learn the message lists the untrusted types. |
| `InvalidSchemaError` | Frame columns do not match native names, a target column is missing, or report labels collide. |
| `InvalidSamplesError` | `sample_rows` is zero, too large, not an integer, or a cell cannot be converted. |
| `MissingOptionalDependencyError` | The `schema` or `lock` extra, or the model's extra, is not installed. |
| `LockGenerationError` | uv found no single wheel for a target and locked version, only an sdist, or could not reach the index. |
| `InvalidEnvironmentLockError` | A supplied lock is not canonical or disagrees with the installed versions or Python. |
| `LimitExceededError` | A member or the archive exceeds the selected limits. |
| `FileExistsError` | The destination exists and `overwrite` is false. |

The full list is in [Errors](/reference/errors/).
