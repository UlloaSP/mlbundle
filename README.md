# mlbundle

Package a trained model, its manifest, locally inferred schema and model
structure in one compressed `.mlbundle` file. Use the same Python library to
export on the training machine and validate/load in the consuming application.

mlbundle replaces three gaps in separate model/DataFrame uploads:

- **Pickle code execution:** closed adapters use skops, native XGBoost UBJSON
  or an existing ONNX file, with no pickle/joblib/cloudpickle loading path.
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
Replace `[sklearn]` with `[xgboost]`, `[onnx]`, `[schema]` or `[all]` for the
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

`export` returns a metadata view and refuses an existing destination unless
`overwrite=True`. `load` accepts a path, bytes or a binary stream. Classifiers
use `bundle.predict_proba`; regressors use `bundle.predict`. Results are plain
lists. For multiple targets, classifier labels/probabilities are output-major,
while regression values are row-major. `native_model` exposes the loaded native
object for explanations and native inference. Do not mutate it and continue
treating the original metadata as verified.

Native feature names take precedence over `feature_names=`. Otherwise supplied
names have origin `declared`; omitted names become `feature_1` through
`feature_N` with origin `generated`. DataFrame names have origin `dataframe`
when the model has none. `target_name=` and `metadata=` are author declarations.

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

The adapter binds labels and prediction settings to a native **copy**, leaving
your model unchanged. Load checks agreement with that binding, not the historical
truth of the label mapping. Without `class_labels=`, labels are encoded indices
or an existing mlbundle binding. Native `predict` still returns encoded indices.
Sklearn and regressors reject `class_labels=`.

Only recoverable native configuration is recorded. Training settings such as
learning rate, max depth, seed and thread count do not survive native XGBoost
I/O and are not reconstructed from the manifest.

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
Training dtypes are declarations, separate from verified model-exposed dtypes.
Inspect/load validate schema mappings, report kinds/labels and sample row keys
without pandas or mlschema.

**What data-derived information leaves the machine:** column names, dtypes,
field kinds, nullability, dtype-based numeric steps, categorical values (including
unused categories), one-hot suffixes, pair-axis metadata and class labels.
Inference adds no numeric/date ranges, means, counts, arbitrary text row values
or example rows. Categories and names can be sensitive. Models and structure
can themselves retain training information; omitting samples does not prevent
memorization. Explicit `sample_rows=N` adds exactly the first N feature rows,
without target/index; N must be positive and no larger than the frame.
See [schema and samples](docs/format-schema.md) for conversions and limits.

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

Inspection prints identity, model/counts, feature origin/names, output-major
labels, member sizes/hashes, training environment and optional-member presence.
The manifest has no self-integrity record. Human display values are escaped and
truncated to 512 characters; `--json` prints the complete validated manifest with
ASCII JSON escaping and the format's size limits. This escapes terminal controls
without changing the decoded JSON. `verify` prints one valid-bundle line or an
error, and requires the matching extra. Errors use one bounded stderr line
without bundle content or traceback. Exit codes are **0** success, **1** bundle
or dependency rejection (`MLBundleError`), **2** usage or I/O failure.

## Supported models and limits

Support requires a faithful round trip through the closed adapter. A catalog
entry alone does not guarantee that every fitted configuration exports.

- **Sklearn:** fitted classifiers exposing `predict_proba` and regressors with
  recoverable feature/output counts. Tested families include DecisionTree,
  ExtraTree, RandomForest, ExtraTrees, GradientBoosting, AdaBoost and Bagging;
  LogisticRegression, LinearRegression, Ridge, Bayesian/ARD, ElasticNet/Lasso,
  Huber, Poisson/Gamma, SGD; SVC with probabilities, SVR/NuSVR/LinearSVR;
  PLSRegression, naive Bayes, LDA/QDA, dense brute-force neighbors,
  NearestCentroid, MLPClassifier/MLPRegressor (adam, sgd, lbfgs), MultiOutput
  wrappers and RegressorChain. MLPs retain optimizer state, work alone/in pipelines
  and have null structure with complete commitments; see [MLP tests](tests/test_sklearn_mlp.py). Scaler and
  dense OneHotEncoder pipelines round-trip; pipeline structure stays null
  because format 1 does not map transformed features. See the
  [tested configurations](tests/test_sklearn_families.py) and
  [core round trips/rejections](tests/test_sklearn.py).
- **Sklearn rejections:** custom estimators/subclasses, callable/type parameters
  (including FunctionTransformer user functions), classifiers without
  probabilities, multilabel/ClassifierChain layouts, XGBoost pipeline steps,
  and untrusted nested storage such as fitted sparse neighbor arrays.
  HistGradientBoosting uses untrusted `TreePredictor` storage and is rejected.
- **Sklearn versions:** the closed class/attribute catalog is audited/tested
  against **1.9.1 with skops 0.16.0**, not every version allowed by `>=1.6`.
  Unknown classes/attributes fail export/load. The embedded sklearn version is
  committed too: loading with a different sklearn version fails native state
  comparison. Use the same sklearn version for export and load. Other versions
  may work when both ends match and their state satisfies the catalog, but are
  not an advertised compatibility range. There is no manifest-driven restoration
  or permissive fallback. See [state ownership](docs/format-state.md).
- **XGBoost:** exact XGBClassifier/XGBRegressor, one target, native `gbtree`
  storage (including XGBoost 3.4.1 `dart` dropout normalized to gbtree).
  Classifiers: `binary:logistic`, `multi:softprob`. Regressors:
  `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`,
  `reg:squaredlogerror`, `reg:logistic`, `count:poisson`, `reg:gamma`,
  `reg:tweedie`. Raw boosters, RF wrappers, subclasses, gblinear, legacy dart
  storage, ranking/survival/custom objectives and multiple targets are rejected.
  At most 100,000 trees and depth 14; exact native state and complete dumps are
  required. See [the XGBoost mapping](docs/format-adapters.md#implemented-xgboost-mapping).
- **ONNX:** one numeric matrix input, or multiple width-one numeric inputs;
  rank two, fixed feature width, float32/float64/int32/int64. Batch is dynamic
  or one. One target: float regression `[batch]`/`[batch,1]`, or tensor/ZipMap
  probabilities with an optional int64/string label output. Classification must
  be inferable from supported graph evidence, including terminal Softmax or
  native classifier labels. Ambiguous shapes, images/string inputs, fixed
  batches above one, multiple targets, external tensors, custom domains/local
  functions and training/runtime configuration are rejected before a CPU session.
  Tested with ONNX 1.23.0/Runtime 1.30.0; actual operators/opsets must pass both.
  Large unpacked TreeEnsembleClassifier/ZipMap exports and multiline producer
  versions work. Wire validation runs once per export/load; the field/scalar
  ceiling follows the 512 MiB member bound, producer strings allow 65,536 characters.
  See [the ONNX bounds](docs/format-onnx.md).

Archive ceilings: **5 members**, **512 MiB** archive/compressed/member bytes,
**1 GiB** aggregate uncompressed bytes, ratio **200**, JSON depth **32**.
Manifest: **8 MiB**; structure: **256 MiB**; schema/samples: **16 MiB each**.
Features/classes: at most **100,000**; outputs: **1,024** (XGBoost/ONNX: one).
Callers may lower `ArchiveLimits`, never raise these ceilings. Native work has
additional [state](docs/format-state.md) and [ONNX](docs/format-onnx.md) bounds;
these byte budgets do not cap total process memory, prediction rows or CPU.

## Bundle contents and security

Every bundle contains `manifest.json` and exactly one of `model.skops`,
`model.ubj`, `model.onnx`. Optional members are `schema.json`, `structure.json`
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
Native parsers and inference remain dependency boundaries. Consumers should
isolate the loading/prediction process, enforce memory/CPU/row limits and maintain
dependencies. See [SECURITY.md](SECURITY.md) for guarantees, limits and reporting.

## Development

From the repository root, use `uv python install 3.14`,
`uv sync --locked --all-extras` and `uv run pre-commit install`.
The default dev group installs every extra; base users can use `uv sync --no-dev`.
Pydantic v2 is the sole core runtime dependency; model/schema imports stay lazy.
The schema extra admits mlschema `>=0.2.1,<0.3` for the 0.2 inferred vocabulary,
with a Python 3.14 marker because 0.2.1 excludes 3.15. `uv.lock` fixes the tested
development versions. Dependency edits require `uv sync` and the updated lockfile.

Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`,
`uv run pytest` and `uv build`. Tests execute every fenced README example,
write terminal/XML coverage and enforce 80% branch-inclusive coverage.
CI preserves lint/types and Ubuntu/Windows tests and checks a clean base wheel
installation on both OSes. `uv build` creates a local wheel and source archive.
Read [AGENTS.md](AGENTS.md) before contributing. Publishing, releases, version
tags, conversions, per-model environments, signing and other model libraries
remain outside scope. The project is MIT licensed.
