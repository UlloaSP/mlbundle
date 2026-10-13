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

**Pre-alpha v0.1.0a1, published to reserve the PyPI name.** Format version 1 is
not frozen; bundles may need re-export. The security design has not had an
independent review: do not accept bundles from untrusted parties.
Requires **Python >=3.14**; tested on Windows/Linux. Schema requires Python 3.14.

## Install and try scikit-learn

Install the pre-release with
`python -m pip install --pre "mlbundle[sklearn]"`.
Replace `[sklearn]` with `[xgboost]`, `[catboost]`, `[onnx]`, `[schema]` or `[all]` for the
corresponding dependencies. Combine extras as `[sklearn,schema]` for local
DataFrame inference. Omit the brackets for a base install that can inspect
every bundle without model libraries. Load/verify needs the matching model extra.

```python
from sklearn.tree import DecisionTreeClassifier
from mlbundle import export, inspect, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = DecisionTreeClassifier(random_state=7).fit(rows, ["cold", "warm"] * 2)
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
export(model, "xgboost.mlbundle", class_labels=[["cold", "warm"]])
bundle = load("xgboost.mlbundle")
assert isinstance(bundle.native_model, XGBClassifier)
assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
assert bundle.class_labels == [["cold", "warm"]]
```

Labels/settings bind to a native **copy**. Load verifies consistency, not history.
Omitted labels use encoded indices/an existing binding; native predict stays encoded.
Standalone sklearn, CatBoost and regressors reject `class_labels=`; composites
delegate labels to the final native adapter.

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

`class_labels=`, categorical/text/embedding features,
Depthwise/Lossguide, custom Python objectives/metrics and other losses are rejected.
Classifier losses: Logloss, CrossEntropy, MultiClass. Regressors: RMSE, MAE,
Quantile, Huber, Poisson, MAPE. Complete trees/borders/NaN handling/importances
are verified on load. [The CatBoost contract](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-catboost.md) details
wrapper recovery, native-parser limits and the private in-memory serializer.

## Composite pipelines

Install `[sklearn,xgboost]` or `[sklearn,catboost]`. Export the fitted Pipeline
as it is; transformers use skops and the final estimator uses its native adapter:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from mlbundle import export, load

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = make_pipeline(StandardScaler(), XGBClassifier(n_estimators=3, n_jobs=1))
model.fit(rows, [0, 1, 0, 1])
export(model, "pipeline.mlbundle", class_labels=[["cold", "warm"]])
bundle = load("pipeline.mlbundle")
assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
assert bundle.class_labels == [["cold", "warm"]]
```

Load rebuilds a real Pipeline after verifying both parts. Bundle probability labels
use XGBoost's bound originals; native classes/predict stay encoded. CatBoost keeps
native labels. Prefixes include scalers, PCA, SimpleImputer (indicators/empty columns),
KBinsDiscretizer, PolynomialFeatures, univariate selectors, OneHotEncoder/OrdinalEncoder,
identity/closed NumPy FunctionTransformer, nested native-free Pipeline and ColumnTransformer.
Each needs a fitted, data-free width proof. Unknown widths, FeatureUnion/KNNImputer,
slice/callable/scalar selectors, caching, metadata routing and other native placements
remain rejected. Step names/outer options are declarations; trees address transformed
features, without original-feature attribution. Schema/dtypes/samples describe original
inputs; six members permit schema plus explicit samples. See [the pipeline contract](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-pipeline.md).

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
no visualization structure. See [the exact ONNX contract](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-onnx.md).

## Local schema and explicit samples

Install a model extra plus `schema` on Python 3.14. Supply a nonempty pandas
DataFrame with string feature names in model input order. A named target must
exist and is excluded; without `target_name`, the frame contains features only.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from mlbundle import export, inspect, load

training = pd.DataFrame(
    [[0.0, 1.0, "cold"], [1.0, 0.0, "warm"]] * 2,
    columns=["temperature", "humidity", "target"],
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
export(model, "sampled.mlbundle", training_data=features, sample_rows=2)
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
N must be positive and within the frame. See [schema and samples](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-schema.md).

## Inspect, load and the CLI

`inspect` validates archive layout, limits, manifest, hashes and schema/sample
declarations **without importing any model library**. It cannot attest that
metadata or structure matches the model. `load` additionally deserializes through
the closed adapter and recomputes native claims and structure before returning
the model and predictor. Export checks the same native round trip before writing.

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

- **Sklearn:** fitted classifiers exposing `predict_proba` and regressors with
  recoverable feature/output counts. Tested families include trees/ensembles,
  linear/regularized/robust/GLM models, SVMs, PLSRegression, naive Bayes, LDA/QDA,
  dense brute-force neighbors, NearestCentroid, MLP and MultiOutput/RegressorChain.
  SimpleImputer/KNNImputer, scalers, KBinsDiscretizer, categorical encoders,
  univariate selectors and closed NumPy FunctionTransformer functions round-trip,
  including nested ColumnTransformer pipelines; see [the exact lists](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-state.md).
  All-sklearn Pipeline structure stays null; composite trees expose the inner boundary. See
  [tested configurations](https://github.com/UlloaSP/mlbundle/blob/main/tests/test_sklearn_families.py) and [core round trips/rejections](https://github.com/UlloaSP/mlbundle/blob/main/tests/test_sklearn.py).
- **Sklearn rejections:** custom estimators/subclasses, functions/types outside the
  closed lists (including lambdas and user functions), compound fitted dtypes, classifiers without
  probabilities, multilabel/ClassifierChain layouts, unsupported native placements,
  and untrusted nested storage such as fitted sparse neighbor arrays.
  HistGradientBoosting uses untrusted `TreePredictor` storage and is rejected.
- **Sklearn versions:** the closed class/attribute catalog is audited/tested
  against **1.9.1 with skops 0.16.0**, not every version allowed by `>=1.6`.
  Unknown state fails; export/load sklearn versions must match. Configurations
  must satisfy the catalog and round trip; see [state ownership](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-state.md).
- **XGBoost:** exact XGBClassifier/XGBRegressor, one target, native gbtree
  (including XGBoost 3.4.1 dart normalized to gbtree). Closed probability/regression
  objectives; raw boosters, RF wrappers, subclasses, ranking/survival/custom
  objectives and unsupported storage are rejected. At most 100,000 trees/depth 14,
  with complete state/dumps; see [the mapping](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-adapters.md#implemented-xgboost-mapping).
- **ONNX:** fixed-width rank-two numeric inputs; one regression/probability target,
  optional native labels and dynamic/one-row batches. Unsupported layouts,
  external data/custom domains/functions and runtime/training configuration fail
  before CPU sessions. Tested with ONNX 1.23.0/Runtime 1.30.0; see [ONNX bounds](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-onnx.md).

Archive ceilings: **6 members**, **512 MiB** archive/compressed/member bytes,
**1 GiB** aggregate uncompressed bytes, ratio **200**, JSON depth **32**.
Manifest: **8 MiB**; structure: **256 MiB**; schema/samples: **16 MiB each**.
Features/classes: at most **100,000**; outputs: **1,024** (XGBoost/CatBoost/ONNX: one).
Callers may only lower `ArchiveLimits`. Native [state](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-state.md)/[ONNX](https://github.com/UlloaSP/mlbundle/blob/main/docs/format-onnx.md)
bounds do not cap process memory, prediction rows or CPU.

## Bundle contents and security

Every bundle contains `manifest.json` and one native member, or the declared
`model.skops` + `model.ubj`/`model.cbm` composite pair. Single-step composites use only the leaf. Optional members are `schema.json`, `structure.json`
(required when the adapter extracts structure) and explicitly requested
`samples.json`. No training DataFrame or arbitrary files are included.
The manifest records identity/time/writer, model parameters/counts/state,
ordered features/dtypes/origin, target labels/name, environment, member
SHA-256/sizes, references and free-form user metadata. The normative
[format specification](https://github.com/UlloaSP/mlbundle/blob/main/docs/format.md) owns these records and verification rules.

Treat bundles as untrusted. The reader validates ZIP layout, allowed members,
limits, strict JSON/manifest and hashes; adapters use closed trust/loading paths
and recompute claims. **Hashes detect corruption, not deliberate tampering**;
there is no signing or proof of authorship, training history, quality or fairness.
Schema, samples and training-environment declarations are not authenticated.
Native parsers/inference require process isolation, memory/CPU/row limits and
maintained dependencies. See [SECURITY.md](https://github.com/UlloaSP/mlbundle/blob/main/SECURITY.md) for guarantees and reporting.

## Development

From the root: `uv python install 3.14`, `uv sync --locked --all-extras`,
`uv run pre-commit install`. Default dev installs every extra; base: `uv sync --no-dev`.
Pydantic v2 is the sole core dependency; model/schema imports are lazy. Schema uses
mlschema `>=0.2.1,<0.3` on 3.14 only (0.2.1 excludes 3.15). `uv.lock` fixes tested
versions; dependency edits require `uv sync` and committing the updated lockfile.

Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`,
`uv run pytest` and `uv build`. Tests execute every fenced README example and
write terminal/XML coverage with an 80% branch-inclusive minimum.
CI checks lint/types and tests the same built wheel on Ubuntu/Windows, including
clean base installs. The owner, or an agent on the owner's explicit request for
that version, runs manual [publishing](https://github.com/UlloaSP/mlbundle/blob/main/docs/publishing.md) from `main` with `pypi_env` approval. Read [AGENTS.md](https://github.com/UlloaSP/mlbundle/blob/main/AGENTS.md).
Agents never run or approve it on their own initiative or add publish paths, tokens or secrets.
Releases, version tags, conversions, per-model environments, signing and additional
model libraries remain outside scope. The project is MIT licensed.
