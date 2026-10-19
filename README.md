<p align="center">
  <img src="https://raw.githubusercontent.com/UlloaSP/mlbundle/main/brand/mlbundle.png" alt="" height="112" />
</p>

<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/UlloaSP/mlbundle/main/brand/wordmark-dark.svg" />
    <img src="https://raw.githubusercontent.com/UlloaSP/mlbundle/main/brand/wordmark-light.svg" alt="MLbundle" height="52" />
  </picture>
</h1>

<p align="center">Your trained model, schema and verified structure. One file.</p>

<p align="center">
  <a href="https://mlbundle.eshkra.dev"><img src="https://img.shields.io/badge/website-303039?style=flat&amp;logo=astro&amp;logoColor=ff6a1a" alt="Website" /></a>
  <a href="https://docs.mlbundle.eshkra.dev"><img src="https://img.shields.io/badge/docs-303039?style=flat&amp;logo=readthedocs&amp;logoColor=ff6a1a" alt="Documentation" /></a>
  <a href="https://playground.mlbundle.eshkra.dev"><img src="https://img.shields.io/badge/playground-303039?style=flat&amp;logo=stackblitz&amp;logoColor=ff6a1a" alt="Playground" /></a>
  <a href="https://pypi.org/project/mlbundle/"><img src="https://img.shields.io/badge/version-0.2.1-b73d00?style=flat&amp;logo=pypi&amp;logoColor=white&amp;labelColor=303039" alt="Version 0.2.1" /></a>
  <a href="https://github.com/UlloaSP/mlbundle"><img src="https://img.shields.io/badge/source-303039?style=flat&amp;logo=github&amp;logoColor=white" alt="Source on GitHub" /></a>
  <a href="https://github.com/UlloaSP/mlbundle/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/UlloaSP/mlbundle/ci.yml?branch=main&amp;style=flat&amp;label=CI&amp;labelColor=303039" alt="CI status" /></a>
  <a href="https://github.com/UlloaSP/mlbundle/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-b73d00?style=flat&amp;labelColor=303039" alt="MIT license" /></a>
  <a href="https://eshkra.dev"><img src="https://img.shields.io/badge/part_of-eshkra-b73d00?style=flat&amp;labelColor=303039" alt="Part of eshkra" /></a>
</p>

MLbundle packages a trained model with its manifest, locally inferred input schema
and extractable structure in one compressed `.mlbundle` file. The same library
writes the bundle and validates it when loaded, preserving feature order, class
labels and output meaning across the training/runtime boundary.

Use it to hand a model to [MLSuite](https://github.com/UlloaSP/mlsuite), inspect
its metadata or build your own consumer around the validated native model.

## Why bundle a model?

A model file alone leaves the next application to recover its inputs, labels,
environment and supporting metadata. Separate descriptions can drift from the
model they are supposed to describe.

MLbundle keeps them together and checks model-derived claims against the loaded
model. Schema inference runs locally through [MLschema](https://github.com/UlloaSP/mlschema);
real training rows are included only when explicitly requested. Native formats
preserve the supported model without conversion.

## Installation

MLbundle **0.2.1** requires Python **3.14** for export with the schema extra.
The installation command for this release is:

```bash
python -m pip install "mlbundle[sklearn,schema]==0.2.1"
```

To use a local checkout, run `python -m pip install -e ".[sklearn,schema]"`
from its root. The base library requires Python >=3.14; schema inference uses
mlschema `>=0.3.0,<0.4` on Python 3.14 only.

Choose `xgboost`, `catboost` or `onnx` instead of `sklearn`, or install `[all]`.
Export always needs `schema`. Metadata-only inspection needs only the base
package; load and CLI verification need the matching model extra.

## Quick start

Train a model, export its schema without including training rows, then inspect
and load the bundle:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlbundle import export, inspect, load

training = pd.DataFrame(
    {
        "temperature": [0.0, 1.0, 0.1, 0.9],
        "humidity": [1.0, 0.0, 0.9, 0.1],
        "target": ["cold", "warm", "cold", "warm"],
    }
)
model = DecisionTreeClassifier(random_state=7).fit(
    training.drop(columns="target"), training["target"]
)

export(
    model,
    "trained.mlbundle",
    training_data=training,
    target_names=["target"],
)

metadata = inspect("trained.mlbundle")  # No model deserialization.
print(metadata.manifest.features.names)  # ["temperature", "humidity"]

bundle = load("trained.mlbundle")  # Recomputes native claims and structure.
result = bundle.predict_record({"temperature": 0.0, "humidity": 1.0})[0]
print(result.predicted_label)  # cold
print(result.probabilities)  # (1.0, 0.0)
```

The target column is excluded from the input schema. Add `sample_rows=N` only
when you intend to include the first N real feature rows. Without
`training_data=`, export infers a default schema from synthetic values.

Existing destinations require `overwrite=True`. Native feature names take
precedence; otherwise mappings use zero-based positions. `load` and `inspect`
accept paths, bytes or streams. The native model remains available as
`bundle.native_model`; mutating it invalidates the verified metadata.

## Choose your operation

| Operation | Purpose |
| --- | --- |
| `export(model, destination, ...)` | Validate a native round trip, infer the schema locally and write a bundle. |
| `inspect(source)` | Validate the container, manifest, hashes and declarations without loading a model. |
| `load(source)` | Load through a closed adapter and verify model-derived claims and structure. |
| `bundle.predict_proba(data)` | Classifier probabilities as plain lists. |
| `bundle.predict(data)` | Regressor predictions as plain lists. |
| `bundle.predict_record(record)` | Map one record through the schema and return frozen output-major results. |
| `bundle.row_from_record(record)` | Validate and order record values for native inference or explanations. |

The CLI exposes metadata inspection and full verification:

```shell
mlbundle inspect trained.mlbundle
mlbundle inspect trained.mlbundle --json
mlbundle verify trained.mlbundle
```

Inspection does not prove that metadata describes the model. Full verification
loads the model and compares its claims. Neither operation establishes authorship
or training history.

## Supported models

| Library | Storage | Support boundary |
| --- | --- | --- |
| scikit-learn | skops | Closed fitted classifier/regressor families and supported transformers. |
| XGBoost | Native UBJSON | Exact single-target tree wrappers and closed objectives. |
| CatBoost | Native CBM | Exact numeric, single-target symmetric classifiers/regressors. |
| ONNX | Original ONNX bytes | Self-contained tabular models with supported signatures; no conversion. |
| Composite pipelines | skops + UBJSON or CBM | Supported sklearn preprocessing with an XGBoost/CatBoost final estimator. |

Custom types fail at export. Sklearn and CatBoost own their class labels;
XGBoost can bind declared labels to a native copy. ONNX has no visualization
structure, and all-sklearn Pipeline structure is null. The
[adapter contracts](https://docs.mlbundle.eshkra.dev/format/adapters/) define the exact support limits.

## Some notes

Package versions and wire-format versions are independent. Format 1 is not frozen;
incompatible schema shapes require re-export. The security design has not had an
independent review; do not accept bundles from untrusted parties.

The reader bounds ZIP/JSON data, verifies integrity and uses closed native loaders.
Hashes detect corruption, not deliberate tampering. Native parsers and inference
still need process isolation and application-owned resource limits.
Read [SECURITY.md](https://docs.mlbundle.eshkra.dev/security/) for guarantees, limits and private reporting.

Training rows stay local by default, but feature names, categories, labels and
learned structure can reveal information. Schemas and environment records remain
author declarations. Model training, hosting, signing and conversion are outside
the library's scope.

## Documentation

The [documentation site](https://docs.mlbundle.eshkra.dev) includes installation, adapter guides, record
prediction and API reference. It renders the normative contracts from this
repository unchanged:

- [Bundle format and verification](https://docs.mlbundle.eshkra.dev/format/)
- [Schema, mappings and explicit samples](https://docs.mlbundle.eshkra.dev/format/schema/)
- [Supported adapters and structure](https://docs.mlbundle.eshkra.dev/format/adapters/)
- [Sklearn state and trusted types](https://docs.mlbundle.eshkra.dev/format/state/)
- [Composite pipelines](https://docs.mlbundle.eshkra.dev/format/pipeline/)
- [ONNX layouts and bounds](https://docs.mlbundle.eshkra.dev/format/onnx/)

The [playground](https://playground.mlbundle.eshkra.dev) provides editable examples and inspection of the
generated `.mlbundle` file. It runs scikit-learn examples in the browser and
inspects uploaded files without loading their models.

## Contributing

The Python library uses [uv](https://docs.astral.sh/uv/) and Python 3.14:

```bash
uv python install 3.14
uv sync --locked --all-extras
uv run pre-commit install
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

Tests use the installed package and enforce 80% branch-inclusive coverage.
CI validates the built wheel on Windows and Linux. Read [AGENTS.md](https://github.com/UlloaSP/mlbundle/blob/main/AGENTS.md)
for contribution rules and [publishing](https://github.com/UlloaSP/mlbundle/blob/main/docs/publishing.md) for the automatic
release workflow.

The landing, docs and playground use [Vite+](https://viteplus.dev/guide/) and a
pnpm workspace. With `vp` installed, run from the repository root:

```bash
vp install
vp run check
vp run build
```

Start a site with `vp run dev:landing`, `vp run dev:docs` or
`vp run dev:playground`. Their local ports are 4340, 4341 and 10021 respectively.
Each site also accepts `vp run dev`, `vp run build` and `vp run preview` from its
directory. Astro sites run their framework CLI through `vp exec`; the playground
uses Vite+ directly. `vp run check` runs frontend lint, landing formatting, all
three sites' type checks, and the docs and playground unit tests. Runtime and
browser tests live in `playground/`.

Actions runs Python and frontend checks on pushes and PRs to `main` and `develop`.
After all checks pass on `main`, it deploys the three tested site builds to
Cloudflare. Manual **CI Pipeline** runs on `main` use the same gates and can
redeploy without publishing Python. A change to `project.version` on `main`
automatically starts the PyPI workflow after its version and branch preflight.
See [website deployment](https://github.com/UlloaSP/mlbundle/blob/main/docs/deployment.md) and [publishing](https://github.com/UlloaSP/mlbundle/blob/main/docs/publishing.md).

## License

MIT. The design takes ideas from MLflow's versioned manifest and native storage,
not its code.
