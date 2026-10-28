---
title: Installation
description: Install mlbundle 0.3.1 with the extras for your model library, from PyPI or a local checkout, and check the Python and dependency ranges.
---

mlbundle requires **Python 3.14 or newer**. Its only core dependency is Pydantic v2; every model
library is an optional extra that is imported lazily, only when a bundle needs it.

These pages document mlbundle 0.3.1.

```bash
python -m pip install "mlbundle[sklearn,schema,lock]==0.3.1"
```

Replace `sklearn` with the extra for your model library; the extras are listed
[below](#choose-the-extras).

## From a local checkout

To work from source, or before 0.3.1 is available from your package index, install a checkout of
the repository:

```bash
git clone https://github.com/UlloaSP/mlbundle.git
cd mlbundle
python -m pip install ".[sklearn,schema,lock]"
```

Inside a uv project, add the checkout as a path dependency instead (adjust the path to where you
cloned it):

```bash
uv add ../mlbundle --extra sklearn --extra schema --extra lock
```

## Choose the extras

| Extra | Installs | Needed for |
| --- | --- | --- |
| `sklearn` | `scikit-learn>=1.6`, `skops>=0.16.0` | Exporting and loading scikit-learn models and pipeline prefixes. |
| `xgboost` | `xgboost>=3.4.1`, `scikit-learn>=1.2` | `XGBClassifier` and `XGBRegressor`. |
| `catboost` | `catboost>=1.2.10` | CatBoost classifiers and regressors. CatBoost brings matplotlib and plotly. |
| `onnx` | `onnx>=1.23.0`, `onnxruntime>=1.30.0` | Existing ONNX files. |
| `schema` | `mlschema>=0.3.0,<0.4` on Python 3.14 only | **Every export.** Also installs pandas. |
| `lock` | `uv>=0.13.5` | **Every export** that generates its runtime lock; not needed with `environment_lock=`. |
| `all` | All of the above | Development and mixed workloads. |

Pick by what each machine does:

- **Training machine:** the model extra plus `schema` and `lock`, for example
  `mlbundle[xgboost,schema,lock]`. Export always infers a schema, even without training data, and
  resolves `pylock.toml` through your package index.
- **Serving or verifying machine:** only the matching model extra, for example `mlbundle[xgboost]`.
  Loading validates the schema without pandas or mlschema. A runtime that serves bundles installs
  each bundle's `pylock.toml` plus mlbundle at its `writer_version` instead.
- **Metadata only:** no extra. `inspect` and `mlbundle inspect` never import a model library.

Composite pipelines need both parts: `[sklearn,xgboost]` or `[sklearn,catboost]`.

:::caution[Schema inference is Python 3.14 only]
The `schema` extra is limited to Python 3.14 because mlschema supports only that version. Export
therefore runs on 3.14. Inspecting and loading validate the schema without mlschema.
:::

## Verify the installation

```python
from importlib.metadata import version

import mlbundle

print(version("mlbundle"))  # 0.3.1
print(mlbundle.FORMAT_NAME, mlbundle.FORMAT_VERSION)  # mlbundle 1
```

The console script is installed with the package:

```bash
mlbundle --help
```

## Troubleshooting

| Symptom | Cause | Resolution |
| --- | --- | --- |
| `MissingOptionalDependencyError` naming `schema` | Export runs without pandas or mlschema. | Install `mlbundle[schema]` on Python 3.14. |
| `MissingOptionalDependencyError` naming `lock` | Export cannot find uv to generate the lock. | Install `mlbundle[lock]`, or pass `environment_lock=`. |
| `LockGenerationError` | uv found no single wheel for a Linux or Windows target, only an sdist, or could not reach the index. | Install versions with wheels for every target, check index access, or pass a lock generated elsewhere. |
| `InvalidEnvironmentLockError` on export | A supplied lock disagrees with the installed versions or Python. | Use a lock generated from this environment. |
| `MissingOptionalDependencyError` naming a library | Load or `verify` needs the model's extra. | Install the extra named in the message. |
| `mlbundle: missing optional dependency; install mlbundle[xgboost].` | Same, from the CLI. | Install that extra; `inspect` works without it. |
| `EnvironmentMismatchError` on load | A skops load failed and the bundle locks a different scikit-learn version. | Install the bundle's `pylock.toml`. |
| `No matching distribution found` | Python is older than 3.14. | Check `python --version`. |
