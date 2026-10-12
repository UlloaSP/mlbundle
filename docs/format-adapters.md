# Version 1 structure, adapters and results

Normative sections of [the format contract](format.md).

## `structure.json`

Structure is never embedded in the manifest. It is a JSON object with required
`family` and `feature_importances` fields, plus exactly the family fields below.
`feature_importances` is null or an ordered numeric array of length `n_features`.
Numbers use the canonical encoding, including non-finite tags. Arrays and tree
lists preserve native order; no pruning, rounding, flattening of output/class
dimensions or display-oriented simplification is permitted. Sklearn structures
add required `state_sha256` and `prediction_state` fields; XGBoost adds required
`native_model`. See [prediction state](format-state.md) for their exact rules.

| `family` | Additional required fields |
| --- | --- |
| `decision_tree` | `tree`: one tree object. |
| `tree_ensemble` | `estimator_shape`: original shape of the ensemble's estimator collection; `trees`: row-major flattened list of tree objects. Product of shape equals list length. A one-dimensional collection uses `[N]`. |
| `linear` | `coef`, `intercept`: native numeric arrays/scalars converted without reshaping. |
| `xgboost` | `trees`: ordered list of parsed JSON tree dumps, each from `get_dump(dump_format="json", with_stats=True)`. Retain all native fields, including node IDs, split/leaf values, missing-child IDs, gain/cover and children. Do not embed dump strings. |
| `catboost` | `native_model`: complete numeric symmetric-tree arrays, feature borders/NaN routing, scale/bias and repacked splits defined in [CatBoost](format-catboost.md). |
| `feature_importances` | No additional fields; the importance array must be non-null. |

A scikit-learn tree object has exactly `node_count` (positive integer),
`max_depth` (nonnegative integer, verified from reachability),
`n_features` (positive integer), `n_outputs` (positive integer), `n_classes`
(native per-output class-count array), and these native `tree_` arrays:
`children_left`, `children_right`, `feature`, `threshold`, `value`, `impurity`,
`n_node_samples`, `weighted_n_node_samples`, `missing_go_to_left`. Except `value`, each array's length
is `node_count`; `value` retains its native `[node, output, class/value]` shape.
Preserve leaf and undefined-feature sentinel values, and sklearn's native value
normalization for the installed version. Do not reinterpret `value` as counts.
Trees use each constituent tree's own dimensions, not an ensemble-wide guess.

Select family in this precedence: a single `tree_`, a tree ensemble, linear
coefficients/intercept, then feature importances only. An ensemble exposing
unsupported non-tree constituents must not masquerade as a supported tree
ensemble. Unsupported or oversized mandatory structure fails export rather than
being silently omitted. ONNX graph visualizations and additional structure
families require a deliberate future contract change.

Sklearn `Pipeline` returns null structure in version 1. Its final estimator
operates on transformed features; version 1 has no input transformation mapping,
so no final-step tree or coefficient member is emitted. Claims, names and
predictions still describe the complete pipeline; `model.state_sha256` also
binds every fitted step and transform. Standalone DecisionTree,
RandomForest, ExtraTrees, GradientBoosting, LogisticRegression, LinearRegression
and Ridge retain their native structures as specified above. HistGradientBoosting
is currently rejected: its `TreePredictor` storage is not on mlbundle's trust
allowlist. No approximate trees are emitted for it.

## Adapter and result contracts

The internal `Adapter` protocol in `mlbundle.adapters` owns recognition,
export preparation, serialization, loading, claim extraction, structure
extraction and a `Predictor`. `prepare_export(model, ExportDeclarations)` returns
the object to serialize and extract, leaving the caller's model unchanged.
`ExportDeclarations` is a strict frozen record with `class_labels` only. Adapters
bind or reject declarations according to their native contracts; neither
`deserialize` nor extraction receives manifest fields. The shared pipeline
resolves labels only when extraction returns null labels for a classifier.
The registry imports only the four constant mlbundle module paths `_sklearn`,
`_xgboost`, `_catboost`, `_onnx`, and constructs their module's `Adapter` class. Library
imports must be lazy, with dependency failures naming the extra to install.
Missing adapter modules raise `AdapterNotImplementedError` on load; export skips
unimplemented adapters while looking for one that recognizes its input. A missing
optional dependency reports `pip install "mlbundle[EXTRA]"`. An unrecognized
object raises `UnsupportedModelError`.

`export(model, destination, *, training_data=None, feature_names=None, class_labels=None,
target_name=None, sample_rows=None, metadata=None, overwrite=False, limits=None)`
returns a `MetadataView` of the written
bundle. It serializes, extracts, reloads and compares both claims and structure
before writing. Model-exposed names take precedence over supplied names. It
captures distribution metadata, including `claims.onnx` in the environment for
the ONNX adapter. [Local schema export](format-schema.md) owns frame alignment,
training dtypes, schema/report declarations and explicit sample selection.
`load(source, *, limits=None)` validates the container, resolves its closed enum,
deserializes, recomputes claims/structure and compares canonical bytes before
constructing a `LoadedBundle`. `ClaimMismatchError` identifies the claim without
echoing values; absent/present structure disagreements are checked separately,
including a present JSON null. Malformed native payloads raise `InvalidModelError`,
an `MLBundleError` subclass, without native exception text.

The sklearn adapter uses only `skops.io.dumps/loads`. Its fixed supplemental
trusted list is exactly `["sklearn.tree._tree.Tree"]`, for numeric node storage
in the supported tree families; it adds no callable, arbitrary class or I/O type.
Skops' own defaults remain part of the dependency trust boundary. Dumped bytes
are audited with `get_untrusted_types`; every type outside the list is rejected
at export and independently at load. XGBoost estimators inside sklearn pipelines
are rejected (`xgboost.core.Booster`, `xgboost.sklearn.XGBClassifier` are untrusted).
CatBoost pipeline steps are also rejected, with unsupported types listed and
an instruction to export `pipeline.named_steps["est"]` directly. Native adapters
own standalone XGBoost and CatBoost loading; neither is implicitly unwrapped.
The nested skops archive is bounded before parsing; every schema-referenced
numeric array header is checked regardless of filename suffix. Headers must
match the stored byte sizes and cannot use object-array loading or zero-itemsize
dtypes. Tree constructor dimensions/class counts and state-array shapes are
bounded before native construction. Loaded tree dimensions, child/feature indices,
reachability and depth are checked before native feature-importance traversal or
prediction, and input/output/class counts must agree with owning estimators.
Closed state, shared-reference work and sparse-array storage rules are specified
in [prediction state](format-state.md) and apply identically to export and load.
Ensemble feature mappings must address the parent input width. Every loaded root object is
checked again for classifier/regressor identity, fitted state and probability
support. Multilabel classifiers with independent positive-class columns and
classifier chains are rejected because their probabilities cannot satisfy the
output-major contract; ordinary multi-output probability-matrix lists are supported.

### Implemented XGBoost mapping

Only exact `XGBClassifier` and `XGBRegressor` classes are supported. Subclasses,
ranking, survival, custom objectives, raw `Booster`, and the deprecated
`XGBRFClassifier`/`XGBRFRegressor` wrappers are rejected. Raw boosters do not
carry enough wrapper identity or custom-objective history for this narrow
training-side boundary; RF wrappers add no required v0.1 contract. Supported
classifier objectives are `binary:logistic` and `multi:softprob`. Supported
regression objectives are `reg:squarederror`, `reg:absoluteerror`,
`reg:pseudohubererror`, `reg:squaredlogerror`, `reg:logistic`, `count:poisson`,
`reg:gamma`, and `reg:tweedie`. Only one target is supported. Binary classifiers
return two probability columns; softmax-only objectives are rejected.

Only native `gbtree` storage is admitted. In XGBoost 3.4.1, the deprecated `dart`
option uses `gbtree` storage with dropout; that representation is supported and
all native dump statistics are retained. Legacy native `dart` storage is rejected.
`gblinear` is rejected: its multi-class coefficient/importance dimensions and
deprecated format are outside the version 1 tree contract. There is no silently
omitted structure for an admitted XGBoost model.

Serialization is exclusively `bytes(booster.save_raw(raw_format="ubj"))` and
`load_model(bytearray(data))`, in memory. Class selection uses the closed adapter
map and the `mlbundle` booster attribute, never the manifest. That attribute is
exactly `{ "estimator_class": CLASS, "class_labels": LABELS_OR_NULL,
"prediction": SETTINGS }` encoded canonically. `SETTINGS` has exactly `missing`
(a finite binary64 value or the canonical NaN tag), `importance_type` (one of
`gain`, `weight`, `cover`, `total_gain`, `total_cover`; omitted native option
normalizes to `gain`), and `enable_categorical` (boolean). These settings survive
loading and are compared with the wrapper. The only other allowed booster keys
are `best_iteration`, `best_score`, `scikit_learn`; they are bounded and included
in verified hyperparameters. Prediction respects the verified best iteration.

`hyperparameters` has exactly these five keys:

```json
{
  "learner_model_param": {"base_score":"[5E-1]","boost_from_average":"1","num_class":"0","num_feature":"4","num_target":"1"},
  "objective": {"name":"binary:logistic","reg_loss_param":{"scale_pos_weight":"1"}},
  "gradient_booster": {"name":"gbtree","gbtree_model_param":{"num_parallel_tree":"1","num_trees":"3"}},
  "prediction": {"missing":{"$float":"nan"},"importance_type":"gain","enable_categorical":true},
  "attributes": {}
}
```

The first three records are a projection of the parsed native `save_config()`:
retain every field in `learner_model_param`, `objective`, and
`gradient_booster.gbtree_model_param`, plus the booster name, with native value
types/spellings unchanged. `prediction` is the bound wrapper settings record.
`attributes` contains the allowed native attributes other than `mlbundle`.
Neither `get_params()` nor the whole config survives native model I/O. Training
settings such as learning rate, max depth, seed, threads and metrics are therefore
not claimed or reconstructed. Export compares this projection and the complete
structure before and after making the native copy, and again through the shared
serialization round trip. Unsupported unrecoverable claims fail publication.

Booster feature names and feature types become verified claims; absent values
are null. Structure is exactly the `xgboost` family above, with the wrapper's
native float32-normalized importance vector for the bound importance type and
the complete `native_model` record, including routing and dropout state.
Prediction values and structure scalars promote float32 exactly to binary64.
Claim/structure comparisons use exact canonical bytes; probability row-sum
validation allows `rtol=1e-6`, `atol=1e-7` for float32 arithmetic.

Before native parsing, the adapter bounds bytes, recognizes the native UBJSON
prefix, validates the complete bounded document, and checks feature/class/target,
tree and node-array dimensions, output-group indices, iteration pointers and
child/feature indices. Numeric arrays remain
byte views. At most 100,000 trees are supported; reachable tree depth is at most
14 so the complete nested dump fits the existing JSON depth ceiling of 32.
Cycles, aliases and unreachable nodes are rejected. Native dump generation is
count-bounded; total dump bytes and canonical structure bytes must also fit
256 MiB and caller-selected lower limits. No truncation is permitted. These
checks do not sandbox the native parser or cap its internal allocations.

Predictors accept adapter-supported tabular objects without a core pandas/numpy
dependency. Regressors return `list[float]` for a single output or row-major
`list[list[float]]` for multiple outputs. Classifiers return row-major
`list[list[float]]` for one output or an output-major list of those matrices for
multiple outputs, aligned with class labels. Validate probability dimensions,
finite probabilities, range and row sums in the adapter. Classifier prediction
uses `predict_proba`; regressor prediction uses `predict`. Unsupported probability
or regression contracts fail rather than synthesizing outputs. Inputs must have
rank two and the native feature width. Numeric inputs reject strings, complex
values and infinities; sklearn/XGBoost permit NaN for native missing-value handling.
A sklearn Pipeline with a fitted OneHotEncoder/OrdinalEncoder also admits strings;
categorical XGBoost DataFrame columns retain native recoding. ONNX requires finite
numeric cells as specified separately. Input and remaining native failures become
bounded `MLBundleError` subclasses without cell contents. Shape errors additionally
retain `ValueError` compatibility through `PredictionInputError`.

`MetadataView` carries `manifest`, a tuple of `MemberName` values including
the manifest, and optional `schema` and `samples` JSON values; it has no model.
`LoadedBundle` additionally carries the exact
native object, a uniform predictor, optional schema/structure/sample values, and
properties for kind, library, class name, ordered feature names and origin, and
output-major original class labels. Calling the prediction method for the wrong
kind raises a bounded `ValueError`. Result records and typed models prevent field
reassignment, but nested JSON values and the native model are not deeply frozen.
Construction itself performs no model validation; `load` supplies the loaded-result
guarantee. Consumers must not mutate a validated
model and continue treating its original metadata as verified.
