# mlbundle

mlbundle is a Python library being built to package a trained model, a rich
manifest, its input schema and its internal structure in one compressed
`.mlbundle` file. The same package will write bundles on the training machine
and read them with validation in a runtime.

**Status: pre-alpha, Phase 2.** The [version 1 format contract](docs/format.md),
strict manifest models, canonical JSON, integrity helpers, environment capture,
adapter/result contracts, secure container reading/writing and public `inspect`
are implemented and tested. No model adapter, public export/load API, schema
inference or CLI exists yet, and nothing has been published to PyPI.

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
member hashes. Future model loading must use a closed set of model types and a
trusted-types list owned by mlbundle, with no pickle, joblib or cloudpickle path. Model-derived
metadata and visualisation structures are claims: the reader must recompute
them from the loaded model and reject mismatches. A metadata-only `inspect`
operation works without deserializing the model or importing model libraries.

Hashes detect corruption; they do not authenticate the author or prevent
deliberate tampering. Signing is outside the project's scope. These guarantees
cover today's container validation; model loading and claim recomputation remain
requirements for future adapters.
See [SECURITY.md](SECURITY.md) for the threat model and vulnerability reporting.

```python
from mlbundle import inspect

metadata = inspect("trained.mlbundle")
print(metadata.manifest.model.library)
print(metadata.members)
```

Inspection validates format and integrity. It cannot attest that model-derived
claims or structure match the model. The internal container functions accept
already serialized bytes; public model export and loading arrive in later phases.

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
adapter compatibility. `uv run pytest`
produces terminal and XML coverage reports and enforces an 80% minimum. CI runs
lint, type checks, and Python 3.14 tests on Ubuntu and Windows. `uv build` creates
local wheel and source archives; it does not publish them.

Read [AGENTS.md](AGENTS.md) before contributing. Publishing, releases, version
tags, sklearn-to-ONNX conversion, per-model environments, signing and additional
model libraries are outside this project's scope.
