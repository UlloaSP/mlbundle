# mlbundle

mlbundle is a Python library for packaging a trained model, a rich manifest and
its internal structure in one compressed `.mlbundle` file. The same package
writes bundles on the training machine and reads them with validation in a runtime.
Local schema inference will be added in a later phase.

**Status: pre-alpha, Phase 3.** The [version 1 format contract](docs/format.md),
strict manifest models, canonical JSON, integrity helpers, environment capture,
adapter/result contracts, secure container reading/writing and public `inspect`
and public sklearn `export`/`load` are implemented and tested. XGBoost and ONNX
adapters, schema inference and the CLI are still missing. Nothing has been
published to PyPI.

## Why it exists

[MLSuite](https://github.com/UlloaSP/mlsuite) turns trained models into reviewable
applications: it generates forms, runs predictions and stores feedback. Its
current workflow uploads separate joblib files for a model and its training
DataFrame. Loading those files can execute arbitrary code, the artifacts do not
record the training environment, and training data leaves the user's machine
just to infer a schema.

mlbundle will compute the schema locally with
[mlschema](https://github.com/UlloaSP/mlschema), record the model and training
environment, and carry everything the runtime needs in one file. Training rows
will be excluded unless the user explicitly asks to include samples.

The first version will support scikit-learn through skops, XGBoost through its
native UBJSON format, and existing ONNX files without conversion. Model-library
dependencies are optional extras (`sklearn`, `xgboost`, `onnx`, `all`), with lazy
adapter resolution. The design takes ideas from MLflow's versioned manifest and
native model formats; it copies no MLflow code. mlbundle is MIT licensed.

## Security model

Bundles are untrusted input. The container reader rejects unsafe ZIP members,
enforces resource limits, validates the manifest and format version, and verifies
member hashes. Sklearn loading uses skops with a fixed trusted-types list owned
by mlbundle, with no pickle, joblib or cloudpickle path. Model-derived metadata
and visualisation structures are claims: `load` recomputes them from the native
model and rejects mismatches. `export` checks that they survive a native round
trip before writing. A metadata-only `inspect`
operation works without deserializing the model or importing model libraries.

Hashes detect corruption; they do not authenticate the author or prevent
deliberate tampering. Signing is outside the project's scope. These guarantees
cover container validation and sklearn claim recomputation; native parsers and
prediction still require a consuming runtime's isolation and resource limits.
See [SECURITY.md](SECURITY.md) for the threat model and vulnerability reporting.

```python
from sklearn.tree import DecisionTreeClassifier
from mlbundle import export, inspect, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
labels = ["cold", "warm", "cold", "warm"]
model = DecisionTreeClassifier(random_state=7).fit(rows, labels)
export(model, "trained.mlbundle", feature_names=["temperature", "humidity"])

bundle = load("trained.mlbundle")
assert isinstance(bundle.native_model, DecisionTreeClassifier)
assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
print(bundle.class_labels)  # [["cold", "warm"]]
print(bundle.predict_proba([[0.0, 1.0]]))  # [[1.0, 0.0]]
print(inspect("trained.mlbundle").manifest.model.library)  # sklearn
```

Inspection validates format and integrity. It cannot attest that model-derived
claims or structure match the model. `load` supplies that guarantee and returns
the native model, predictor and metadata. Model-exposed feature names take
precedence over `feature_names=`; otherwise those names are declarations with
origin `dataframe`, or omitted names become `feature_1` through `feature_N`.
`target_name=` and `metadata=` record author declarations. Existing destinations
require `overwrite=True`. Paths, bytes and binary streams are accepted by `load`.

Supported sklearn models must be fitted classifiers with `predict_proba` or
regressors, with all stored types trusted and all claims surviving skops.
DecisionTree, RandomForest, ExtraTrees, GradientBoosting, LogisticRegression,
LinearRegression, Ridge and scaler/classifier pipelines are covered by round-trip
tests. Multi-output labels and probabilities retain output-major order. Pipeline
structure is null because v1 does not map transformed features. Custom types,
user functions, HistGradientBoosting and XGBoost steps inside sklearn pipelines
are rejected. Export currently includes no training DataFrame, schema or samples.

## Development

Use Python 3.14 and [uv](https://docs.astral.sh/uv/). The Python minimum follows
mlschema 0.2.1; source code must avoid syntax that requires Python 3.14 so the
supported range can be widened later. Pydantic v2 is the sole core runtime
dependency, for strict manifest validation. Model libraries are optional;
mlschema integration comes in Phase 5.

```shell
uv python install 3.14
uv sync --locked --all-extras
uv run pre-commit install
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

The default dev group installs all model extras. Base consumers can use
`uv sync --no-dev` without installing model libraries. The committed `uv.lock`
fixes the development versions of scikit-learn, skops, XGBoost, ONNX and ONNX
Runtime to the target MLSuite runtime baseline; this is not yet evidence of
compatibility for the future adapters; sklearn compatibility is exercised by
the round-trip tests. Pandas is a development dependency for DataFrame tests,
not a core runtime dependency. `uv run pytest`
produces terminal and XML coverage reports and enforces an 80% minimum. CI runs
lint, type checks, and Python 3.14 tests on Ubuntu and Windows. `uv build` creates
local wheel and source archives; it does not publish them.

Read [AGENTS.md](AGENTS.md) before contributing. Publishing, releases, version
tags, sklearn-to-ONNX conversion, per-model environments, signing and additional
model libraries are outside this project's scope.
