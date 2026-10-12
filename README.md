# mlbundle

Package a trained model, manifest, locally inferred schema and structure in one `.mlbundle` file.
Export on the training machine and validate/load in the consuming application with the same Python library.

mlbundle replaces three gaps in separate model/DataFrame uploads:

- **Pickle code execution:** closed adapters use skops, native XGBoost UBJSON,
  CatBoost CBM or an existing ONNX file, without pickle/joblib/cloudpickle.
- **Missing training versions:** the manifest records Python, OS, architecture
  and installed package versions alongside model and feature metadata.
- **Training data leaving the machine:** schema inference runs locally;
  training rows are included only when explicitly requested as samples.

**Pre-release v0.1.0; format version 1 is not yet frozen.** Nothing is published
on PyPI. Requires **Python >=3.14**. The tested baseline is CPython 3.14 on
Windows and Linux; the schema extra currently works only on Python 3.14.

## Install and try scikit-learn

Install from Git with
`python -m pip install "mlbundle[sklearn] @ git+https://github.com/UlloaSP/mlbundle.git@develop"`.
Replace `[sklearn]` with `[xgboost]`, `[catboost]`, `[onnx]`, `[schema]` or `[all]` for the
corresponding dependencies. Combine extras as `[sklearn,schema]` for local
DataFrame inference. Omit the brackets for a base install that can inspect
every bundle without model libraries. Load/verify needs the matching model extra.

```python
from sklearn.tree import DecisionTreeClassifier
from mlbundle import export, inspect, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = DecisionTreeClassifier(random_state=7).fit(
    rows, ["cold", "warm", "cold", "warm"]
)
export(model, "trained.mlbundle", feature_names=["temperature", "humidity"])

bundle = load("trained.mlbundle")
assert isinstance(bundle.native_model, DecisionTreeClassifier)
assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
assert bundle.class_labels == [["cold", "warm"]]
print(bundle.predict_proba([[0.0, 1.0]]))  # [[1.0, 0.0]]
assert inspect("trained.mlbundle").manifest.model.library == "sklearn"
```

`export` returns metadata and requires `overwrite=True` for existing destinations.
`load` accepts paths, bytes or streams. Classifiers use `bundle.predict_proba`;
regressors use `bundle.predict`. Results are plain lists: output-major classifier
labels/probabilities, row-major regression values. `native_model` supports native
inference/explanations; mutating it invalidates the metadata verification.

Native names win; otherwise origin is `declared`, `dataframe`, or `generated` (`feature_1` through `feature_N`).
`target_name=` and `metadata=` remain author declarations.

## XGBoost

Install the `xgboost` extra. Train on encoded indices and supply the original
labels in probability-column order, nested even for one output:

```python
from xgboost import XGBClassifier
from mlbundle import export, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(rows, [0, 1, 0, 1])
export(
    model,
    "xgboost.mlbundle",
    feature_names=["temperature", "humidity"],
    class_labels=[["cold", "warm"]],
)
bundle = load("xgboost.mlbundle")
assert isinstance(bundle.native_model, XGBClassifier)
assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
assert bundle.class_labels == [["cold", "warm"]]
```

The adapter binds labels/settings to a native **copy**, leaving your model
unchanged. Load verifies that binding, not historical truth. Omitted labels use
encoded indices or an existing binding; native `predict` still returns indices.
Sklearn, CatBoost and regressors reject `class_labels=`.

Only recoverable native configuration is recorded; learning rate, max depth,
seed and thread count do not survive native XGBoost I/O and are not reconstructed.

## CatBoost

Install the `catboost` extra (>=1.2.10; includes matplotlib/plotly). Numeric,
single-target symmetric classifiers/regressors use native CBM and native labels:

```python
from catboost import CatBoostClassifier
from mlbundle import export, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = CatBoostClassifier(
    iterations=3, depth=2, verbose=False, allow_writing_files=False
).fit(rows, ["cold", "warm"] * 2)
export(model, "catboost.mlbundle")
bundle = load("catboost.mlbundle")
assert bundle.class_labels == [["cold", "warm"]]
assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
```

Export a pipeline's estimator as `export(pipeline.named_steps["est"], ...)`.
CatBoost pipelines, `class_labels=`, categorical/text/embedding features,
Depthwise/Lossguide, custom Python objectives/metrics and other losses are rejected.
Classifier losses: Logloss, CrossEntropy, MultiClass. Regressors: RMSE, MAE,
Quantile, Huber, Poisson, MAPE. Complete trees/borders/NaN handling/importances
are verified on load. [The CatBoost contract](docs/format-catboost.md) details
wrapper recovery, native-parser limits and the private in-memory serializer.

## Existing ONNX files

Install the `onnx` extra. This example assumes your existing `trained.onnx` is
a self-contained tabular classifier with two numeric features and two probability
columns. Labels on a label-free graph are count-checked declarations:

```python
from pathlib import Path
from mlbundle import export, load

original = Path("trained.onnx").read_bytes()
export("trained.onnx", "onnx.mlbundle", class_labels=[["cold", "warm"]])
bundle = load("onnx.mlbundle")
assert bundle.native_model.data == original
assert bundle.native_model.session.get_providers() == ["CPUExecutionProvider"]
assert bundle.class_labels == [["cold", "warm"]]
print(bundle.predict_proba([[0.0, 0.0]]))
```

Paths and bytes are preserved unchanged; a ModelProto is serialized once.
There is no conversion. `native_model` contains the original `data` and CPU
`session`. Native graph labels are verified; supplied labels must agree with
them. Without native/supplied labels, encoded indices are generated. ONNX has
no visualization structure. See [the exact ONNX contract](docs/format-onnx.md).

## Local schema and explicit samples

Install a model extra plus `schema` on Python 3.14. Supply a nonempty pandas
DataFrame with string feature names in model input order. A named target must
exist and is excluded; without `target_name`, the frame contains features only.

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
features = training.drop(columns="target")
model = DecisionTreeClassifier(random_state=7).fit(features, training["target"])
export(model, "schema.mlbundle", training_data=training, target_name="target")
view = inspect("schema.mlbundle")
assert view.manifest.features.dtypes == ["float64", "float64"]
assert view.schema["reports"][0]["labels"] == ["cold", "warm"]
assert view.samples is None
assert load("schema.mlbundle").schema == view.schema

# Samples are real data: explicitly include the first two feature rows.
export(
    model,
    "sampled.mlbundle",
    training_data=training,
    target_name="target",
    sample_rows=2,
)
assert inspect("sampled.mlbundle").samples == features.head(2).to_dict("records")
```

`schema.json` contains a schema version, locally inferred fields and one report
per target output. `color__red`/`color__blue` binary columns can become one
category field using mlschema's `__` convention. Frame names must match native
names in order; `feature_names=` cannot rename/reorder a supplied frame.
Training dtypes are declarations, separate from verified model dtypes. Inspect/load
validate schema mappings, reports and sample keys without pandas or mlschema.

**Data leaving the machine:** names/dtypes, field kinds/nullability, dtype-based
steps, categories (including unused), one-hot suffixes, pair-axis metadata and
class labels. No inferred ranges, means, counts or arbitrary text/example rows.
Names/categories and learned models/structure can reveal training information.
Explicit `sample_rows=N` includes the first N feature rows, without target/index;
N must be positive and within the frame. See [schema and samples](docs/format-schema.md).

## Inspect, load and the CLI

`inspect` validates archive layout, limits, manifest, hashes and schema/sample
declarations **without importing any model library**. It cannot attest that
metadata or structure matches the model. `load` additionally deserializes through
the closed adapter and recomputes native claims and structure before returning
the model and predictor. Export checks the same native round trip before writing.

After the sklearn example above, these commands work:

```shell
mlbundle inspect trained.mlbundle
mlbundle inspect trained.mlbundle --json
mlbundle verify trained.mlbundle
```

Inspection prints metadata, members/hashes and declarations. Human values escape
terminal controls and truncate at 512 characters; `--json` is lossless ASCII-escaped
manifest JSON. `verify` requires the model extra. Errors are one bounded stderr
line without bundle content/traceback. Exit codes: **0** success, **1** bundle or
dependency rejection (`MLBundleError`), **2** usage/I/O failure.

## Supported models and limits

Support requires a faithful closed-adapter round trip; a catalog entry does not guarantee every fitted configuration exports.

- **Sklearn:** fitted classifiers exposing `predict_proba` and regressors with
  recoverable feature/output counts. Tested families include trees/ensembles,
  linear/regularized/robust/GLM models, SVMs, PLSRegression, naive Bayes, LDA/QDA,
  dense brute-force neighbors, NearestCentroid, MLP and MultiOutput/RegressorChain.
  MLPClassifier/Regressor (adam, sgd, lbfgs) retain optimizer state, alone/in pipelines, with null structure and complete commitments; see [MLP tests](tests/test_sklearn_mlp.py).
  SimpleImputer/KNNImputer, scalers, KBinsDiscretizer, categorical encoders,
  univariate selectors and closed NumPy FunctionTransformer functions round-trip,
  including nested ColumnTransformer pipelines; see [the exact lists](docs/format-state.md).
  Pipeline structure stays null because format 1 does not map transformed features. See
  [tested configurations](tests/test_sklearn_families.py) and [core round trips/rejections](tests/test_sklearn.py).
- **Sklearn rejections:** custom estimators/subclasses, functions/types outside the
  closed lists (including lambdas and user functions), compound fitted dtypes, classifiers without
  probabilities, multilabel/ClassifierChain layouts, XGBoost/CatBoost pipeline steps,
  and untrusted nested storage such as fitted sparse neighbor arrays.
  HistGradientBoosting uses untrusted `TreePredictor` storage and is rejected.
- **Sklearn versions:** the closed class/attribute catalog is audited/tested
  against **1.9.1 with skops 0.16.0**, not every version allowed by `>=1.6`.
  Unknown classes/attributes fail export/load. The committed sklearn version
  must match at export/load; other matching-version configurations can work
  when they satisfy the catalog. No permissive fallback or advertised broader
  range. See [state ownership](docs/format-state.md).
- **XGBoost:** exact XGBClassifier/XGBRegressor, one target, native `gbtree`
  storage (including XGBoost 3.4.1 `dart` dropout normalized to gbtree).
  Classifiers: `binary:logistic`, `multi:softprob`. Regressors:
  `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`,
  `reg:squaredlogerror`, `reg:logistic`, `count:poisson`, `reg:gamma`,
  `reg:tweedie`. Raw boosters, RF wrappers, subclasses, gblinear, legacy dart
  storage, ranking/survival/custom objectives and multiple targets are rejected.
  At most 100,000 trees and depth 14; exact native state and complete dumps are
  required. See [the XGBoost mapping](docs/format-adapters.md#implemented-xgboost-mapping).
- **ONNX:** rank-two numeric matrix or width-one inputs, fixed width,
  float32/float64/int32/int64; dynamic or one-row batches. One target: float
  regression `[batch]`/`[batch,1]` or tensor/ZipMap probabilities, optional labels.
  Classification requires native graph evidence. Ambiguous/image/string layouts,
  larger fixed batches, multiple targets, external tensors, custom domains/local
  functions and training/runtime configuration fail before CPU sessions.
  Tested with ONNX 1.23.0/Runtime 1.30.0; operators/opsets must pass both.
  Large unpacked ensembles/ZipMap and multiline producer versions work. One wire
  validation per export/load; field/scalar ceilings follow 512 MiB, producer strings
  allow 65,536 characters. See [ONNX bounds](docs/format-onnx.md).

Archive ceilings: **5 members**, **512 MiB** archive/compressed/member bytes,
**1 GiB** aggregate uncompressed bytes, ratio **200**, JSON depth **32**.
Manifest: **8 MiB**; structure: **256 MiB**; schema/samples: **16 MiB each**.
Features/classes: at most **100,000**; outputs: **1,024** (XGBoost/CatBoost/ONNX: one).
Callers may lower `ArchiveLimits`, never raise these ceilings. Native work has
additional [state](docs/format-state.md) and [ONNX](docs/format-onnx.md) bounds;
these byte budgets do not cap total process memory, prediction rows or CPU.

## Bundle contents and security

Every bundle contains `manifest.json` and exactly one of `model.skops`,
`model.ubj`, `model.cbm`, `model.onnx`. Optional members are `schema.json`, `structure.json`
(required when the adapter extracts structure) and explicitly requested
`samples.json`. No training DataFrame or arbitrary files are included.
The manifest records identity/time/writer, model parameters/counts/state,
ordered features/dtypes/origin, target labels/name, environment, member
SHA-256/sizes, references and free-form user metadata. The normative
[format specification](docs/format.md) owns these records and verification rules.

Treat bundles as untrusted. The reader validates ZIP layout, allowed members,
limits, strict JSON/manifest and hashes; adapters use closed trust/loading paths
and recompute claims. **Hashes detect corruption, not deliberate tampering**;
there is no signing or proof of authorship, training history, quality or fairness.
Schema, samples and training-environment declarations are not authenticated.
Native parsers/inference require process isolation, memory/CPU/row limits and
maintained dependencies. See [SECURITY.md](SECURITY.md) for guarantees and reporting.

## Development

From the root: `uv python install 3.14`, `uv sync --locked --all-extras`,
`uv run pre-commit install`. Default dev installs every extra; base: `uv sync --no-dev`.
Pydantic v2 is the sole core dependency; model/schema imports are lazy. Schema uses
mlschema `>=0.2.1,<0.3` on 3.14 only (0.2.1 excludes 3.15). `uv.lock` fixes tested
versions; dependency edits require `uv sync` and committing the updated lockfile.

Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`,
`uv run pytest` and `uv build`. Tests execute every fenced README example and
write terminal/XML coverage with an 80% branch-inclusive minimum.
CI preserves lint/types and Ubuntu/Windows tests and checks a clean base wheel
installation on both OSes. `uv build` creates a local wheel and source archive.
Read [AGENTS.md](AGENTS.md) before contributing. Publishing, releases, version
tags, conversions, per-model environments, signing and additional model libraries
remain outside scope. The project is MIT licensed.
