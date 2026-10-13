# Version 1 CatBoost mapping

Normative section of [the format contract](format.md). Audited and tested with
CatBoost 1.2.10 on Python 3.14. `catboost>=1.2.10` is an optional extra included
in `all`; its floor supplies the tested CBM/API baseline and Python 3.14 wheels.
The floor does not promise compatibility with every later native schema.
Base import and metadata-only inspect do not import CatBoost.

## Closed support boundary

Only exact fitted `catboost.CatBoostClassifier` and `CatBoostRegressor` are
supported, with one target and numeric features. Classifier losses are
`Logloss`, `CrossEntropy`, `MultiClass`. Regressor losses are `RMSE`, `MAE`,
`Quantile`, `Huber`, `Poisson`, `MAPE`, including their native parameter strings.
`MultiClassOneVsAll` does not guarantee probability row sums of one and is
rejected. Multi-target/uncertainty, ranking, other losses, bare CatBoost,
rankers, subclasses and custom Python objectives/metrics are unsupported.
The fitted wrapper attribute set and native `_object` type are closed; overrides, unknown
attributes and prediction-changing parameter caches are rejected. Export never
silently normalizes a blob-loaded Poisson wrapper that originally predicts raw values.

Only `SymmetricTree` (oblivious trees) is admitted. `Depthwise` and `Lossguide`
are rejected. Categorical, text and embedding features are rejected at export
and independently before native load. Categorical CBM round trips work natively,
but CTR storage has an unchecked native loader and needs its own preflight.
There is no implicit input encoding or model conversion in this adapter.

An exact outer sklearn Pipeline may end in an admitted CatBoost wrapper via the
[composite representation](format-pipeline.md). The prefix uses checked skops bytes;
CBM passes through this adapter's unchanged preflight/claims/structure checks.
Other native placements are rejected; CatBoost remains outside skops trust.
`class_labels=` is rejected even if equal: native `classes_` owns the labels.
Class arrays retain native scalar types and probability-column order and are
output-major, including `[[0, 1]]` for binary classification. Int and string
binary/multiclass labels have tested exact round trips.

## Native I/O and wrapper recovery

Library/serialization/member pairing is `catboost` / `cbm` / `model.cbm`.
Standalone bundles have one native member; composites may also have a skops prefix.
CBM never goes through skops; there is no pickle fallback.
Public `save_model(path, format="cbm")` writes in an export-only temporary
directory that is removed before publication. Public saving has no in-memory
API. Public `load_model(blob=data)` consumes bytes without reader-side disk I/O.
The blob path ignores its `format` argument and supports CBM, not native JSON.

CatBoost's audited private `_serialize_model()` returns CBM bytes through its
C++ `SerializeModel`; it is used only to recompute current native state for
extraction/verification. It performs no pickle serialization and writes no files.
CBM layout/metadata order may change when reserialized, so verification compares
semantic snapshots, not a SHA of the entire serialized CBM buffer.

Blob loading preserves `get_all_params`, feature names, class labels, best
iteration, stored tree truncation, scale/bias and probability threshold.
It loses constructor `get_params` and returns zero for `n_features_in_`.
The reader chooses a class from the fixed classifier/regressor map using the
validated native loss and sets its feature-width cache from the native table.
It constructs that wrapper with the validated loss. This is prediction-critical
for Poisson: a default regressor loaded via blob otherwise predicts raw values
instead of exponentiated values. No manifest setting restores state.

Constructor parameters are not claimed or reconstructed wholesale. The
recoverable effective parameters from `get_all_params()` are claimed. Native
training/evaluation metadata is committed verbatim, but API evaluation-history
projections are not separately claimed: native parsing may change their float
spelling/projection without changing predictions. Native labels/borders and
numeric CBM probability/regression predictions survived tested round trips
exactly. JSON file export/load produced float noise and is not the chosen format.

## Claims and complete structure

`hyperparameters` has exactly `parameters` (the recoverable `get_all_params()`
JSON projection), `tree_count` (actual stored trees), `best_iteration` (native
value or null), `scale_and_bias` (native pair), and `probability_threshold`
(native classifier threshold, null for a regressor). CatBoost predicts all stored
trees by default; best iteration is descriptive, and shrinking changes the
actual native tree arrays. Feature names are always model-origin: CatBoost
generates numeric string names when fitting unnamed arrays. `model_dtypes` is
null; numerical acceptance/quantization does not establish training dtypes.

Counts, kind, class identity, ordered names, native labels, hyperparameters and
`model.state_sha256` are extracted by the same code at export/load and compared
by canonical JSON bytes. The commitment covers every admitted native feature,
tree, repacked split and all native metadata, including class mapping, params,
best-iteration/training metadata and the threshold. It does not authenticate
training history. Loaded results contain the real CatBoost object. Mutating it
invalidates the verified relationship with metadata.

Mandatory structure is exactly `family="catboost"`, `feature_importances`
and `native_model`. Importances are freshly computed with
`get_feature_importance(type="PredictionValuesChange", thread_count=1)` on a
native public `copy()`, so computation cannot overwrite the caller's importance cache;
not copied from the fitted wrapper's cached `feature_importances_`.
`native_model` has exactly these fields:

| Field | Native representation |
| --- | --- |
| `approx_dimension` | One for binary/regression; class count for MultiClass. |
| `tree_depths` | Depth per tree, in native order. |
| `tree_split_offsets` | Start offset per tree into `tree_splits`; cumulative depths. |
| `tree_splits` | Native bin-feature indices, in tree/level order. |
| `leaf_values` | Native flat binary64 array in tree, leaf, approximation-dimension order. Tree `t` has `2**tree_depths[t] * approx_dimension` values; retain every dimension. |
| `leaf_weights` | Native flat binary64 weights in tree/leaf order. |
| `float_features` | Native ordered records with `index`, `flat_index`, `name`, `has_nans`, `nan_value_treatment`, `borders`. Borders promote binary32 exactly to binary64. NaN treatment is native enum 0=AsIs, 1=AsFalse, 2=AsTrue. |
| `scale`, `bias` | Native binary64 scale and dimension-length bias. Legacy scalar bias normalizes to that vector. |
| `repacked_bins` | Native little-endian uint32 words: low 16 bits feature bucket, next byte XOR mask, high byte split index; zero or one per stored split. |

The split lookup flattens float-feature borders in feature/border order. Bit
`level` in a leaf index corresponds to that level's split decision. Splits use
CatBoost's native greater-than-border test after native NaN handling. No pruning,
rounding or manufactured node gains. Full canonical structure comparison rejects
missing, altered or additional records. The existing 256 MiB structure ceiling
and caller-lowered limits apply; oversize structure is rejected, never truncated.

### Native state commitment

Root is exactly `{"native_model":NATIVE,"metadata":INFO_MAP}` using the snapshot
above before JSON conversion. Every dictionary replaces children by digests
then hashes `b"dict:" + canonical_dumps(record)`. Lists hash
`b"list:" + canonical_dumps(child_digests)`. Scalars hash
`b"scalar:" + canonical_dumps(value)`. Each numeric array hashes
`b"array:" + canonical_dumps([dtype.str, shape]) + contiguous_bytes`.
Arrays use their native little-endian `<i4`, `<u4`, `<f4`, `<f8` dtypes and flat
shapes. Floats must be finite; signed zero and exact bits are retained. Digests
are lowercase SHA-256 hexadecimal. Dictionary key order and CBM buffer layout
do not affect the commitment. Unknown CBM table fields fail preflight.
The payload commitment must survive native loading/recomputation before claims
are compared with the manifest; ignored or altered state is rejected.

## Bounded work and predictor

Before native load, require exact `CBM1` prefix, uint32 core length equal to
the remaining member bytes, core version `FlabuffersModel_v1` (native spelling),
closed tables, checked field/indirect offsets and stored vector/string extents.
Unknown model parts and nonempty categorical/CTR, non-symmetric, text,
embedding/estimated-feature arrays are rejected without native traversal.
Numeric vectors remain byte views. Table visits are capped at 200,000;
features/classes at 100,000, trees at 100,000, depth at 16 and aggregate borders
at 1,600,000. Feature indices/order, split offsets/indices, leaf dimensions,
finite values, nonnegative weights and ordered borders must agree. Metadata
has at most 4,096 distinct keys and 8 MiB aggregate UTF-8 bytes. Native params
and class JSON use 1 MiB; params nesting 16, other metadata JSON nesting 32.
The native model member remains capped at 512 MiB with caller-lowered bounds.

Predictors use shared rank-two numeric checks and the native feature width.
NaN is admitted for native handling; strings, complex values and infinity are
rejected. DataFrame columns must match native names/order; arrays/lists are
positional. Empty batches fail before CatBoost. Prediction uses one CPU thread;
regressors return a finite row vector, classifiers a finite row/class matrix in
`[0,1]` with row sums one (`rtol=1e-6`, `atol=1e-7`). Wrong-kind calls raise
bounded ValueError. Shape/name errors use PredictionInputError; remaining
native/model errors become bounded MLBundleError without native text or cells.

These checks do not sandbox CatBoost or bound internal native allocations/CPU,
serializer/importance work or process crashes. Windows corrupt-byte subprocess
probes included CatBoostError, UnicodeDecodeError and accepted truncation; no
crash or timeout was observed. Full verification still needs host isolation and
process/row limits. Full bundle load also parses and exactly compares expanded
structure JSON; it has substantially more work than native CBM loading alone.

Before native parsing or Python list conversion, a conservative structure
allocation budget also fits 256 MiB: 4,096 bytes overhead, 25 bytes per floating
array element, 12 per integer element, and 256 bytes plus six times UTF-8 name
bytes per feature (plus 25 per border). It can reject a model whose actual compact
JSON fits; it avoids allocating a huge Python list merely to discover an
oversized structure. Numeric commitments hash buffer views without a full copy.
