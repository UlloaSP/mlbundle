# Version 1 prediction state and bounded native work

Normative section of [the format contract](format.md). Phase 6 completes the
unreleased version 1 contract; it does not increment the format version. Earlier
unreleased bundles lacking the new state records fail full claim comparison.

## Sklearn state ownership

Skops' type audit is necessary but does not define which attributes belong to an
estimator. mlbundle additionally owns a closed class/attribute catalog in
`_sklearn/state_fields.json`, audited against scikit-learn 1.9.1. It covers public
estimator constructor parameters, inherited and fitted attributes, shared fit
helpers and the admitted loss/link internals. It is package data, never supplied
by a bundle and never used to import or construct a class. Unknown object types,
attributes, nested estimator attributes, node keys, loaders and unreferenced files
are rejected. The supplemental skops trusted list remains Tree only. A catalog
entry does not extend skops' trust or guarantee that every configuration exports.

Object state must be a dictionary with one string key type per attribute; skops
may otherwise truncate attributes silently. Stateless admitted objects may use
the native null state. Tree constructor state has exactly `node_count`,
`max_depth`, `nodes`, `values`. Constructor dimensions, buffer shapes and loaded
tree reachability/indices are checked before native traversal. Tree capacity must
equal node count; maximum depth is recomputed, and maximum class width is derived
from the per-output class counts. Missing directions are zero or one.
Estimator classes must be the exact installed native class, rather than a subclass
impersonating its module and name. This rule also applies before export.

Every admitted state field participates in a complete state commitment, including
fields omitted from visualization and optional training-only state. Deserialization
also compares the payload commitment with a fresh skops serialization of the
loaded object. Ignored declarations or state that changes during construction
are invalid models. Export applies the same rules before publication.

### Commitment encoding

`state_sha256` is a lowercase SHA-256 digest. A sklearn structure object requires
this field. Sklearn models with null structure, including Pipeline, instead put
the digest in `model.state_sha256`; other models use null there. This optional
manifest field defaults to null, and is part of the verified claim projection.
Missing commitments on an opaque sklearn model fail full load comparison.

The commitment is a Merkle digest of the bounded skops schema and its buffers.
For each dictionary, resolve CachedNode to its definition, omit `__id__`, replace
each child with its digest, then hash `b"dict:" + canonical_dumps(record)`.
Lists hash `b"list:" + canonical_dumps(child_digest_list)`; scalars hash
`b"scalar:" + canonical_dumps(value)`. A `file` child uses
`SHA256(b"file:" + canonical_dumps(buffer_digest))` instead of a filename.
Shared definitions and files are memoized; identifiers and ZIP timestamps do not
affect the result. Metadata, loader/class identity and array dimensions do.

A numeric buffer hashes canonical `[dtype.str, shape]` followed by contiguous
row-major element bytes. Floating NaNs normalize to NumPy's NaN before hashing;
finite bits, signed zero, dtype and shape are preserved. Structured arrays hash
the canonical dictionary of named-field numeric digests; unnamed padding is
excluded because native Tree storage leaves it uninitialized. Sparse buffers
hash the canonical dictionary of inner member names and their numeric digests.
These records bind state; they do not authenticate authorship or training history.

### Readable inference composition

Every sklearn structure also requires `prediction_state`, an object with exactly
the applicable fields below. Missing fields are omitted rather than invented.
Native arrays keep their shape and canonical numeric representation.

| Native state | Record |
| --- | --- |
| `estimators_` | `estimator_labels`: row-major constituent `classes_` arrays, or null for a constituent without labels. |
| `estimator_weights_`, `estimators_features_` | The corresponding arrays under their native names. |
| `_x_mean`, `_x_std` | Native centering/scaling vectors under their native names. |
| `_loss` or `_base_loss` | `loss`: exactly `class`, `link` (class names), `n_classes`. |
| `init_` | `initializer`: literal native value, or exactly `class`, `parameters`, `state_sha256` for an estimator. Dummy initializers additionally expose `values`, with each present `classes_`, `class_prior_`, `constant_`, `n_classes_`, `n_outputs_`. `learning_rate` is also present. |
| `_probA`, `_probB` | `probability_calibration`: exactly `A`, `B` native arrays. |

A custom supported initializer's commitment binds all fitted state; it does not
manufacture a visualization for that initializer. Linear coefficients are native
decision/regression coefficients. Centered models and calibrated classifiers
require the composition record to interpret the complete prediction.

### Prediction-state coverage

The commitment covers **all** allowed attributes and nested buffers, not a
selection inferred from attribute suffixes. The explicit checks supplement it:

| Family | Prediction-relevant state and coverage |
| --- | --- |
| DecisionTree / ExtraTree | All eight node record fields, including missing routing, plus value arrays and native dimensions are in the tree record. Depth, capacity and class width are derived/checked. Parameters, names, owning output counts and labels are claims. |
| RandomForest / ExtraTrees | Constituent trees and order are structure; parent/child output and class counts agree. Parent counts, labels and parameters are claims; remaining state is committed. |
| GradientBoosting | Stage arrangement/counts, constituent trees, initializer, learning rate and loss/link state are structure/composition. Stage width agrees with classes; fitted loss/link class and loss class count agree with the native parameter-derived loss. Initializer feature width, dummy priors and dummy output dimensions agree with fitted state. |
| AdaBoost / Bagging / tree meta-estimators | Trees, native order/shape, weights, feature subsets and constituent labels are structure/composition. Feature subsets address parent input widths. All other state, including output mappings, is committed. |
| LinearRegression / Ridge / Bayesian / sparse-penalty / robust / SGD / LinearSVR / LDA | Native coefficients/intercepts are structure, parameters and labels/counts are claims, all fitted state is committed. Native probability transforms remain library-owned. |
| PLS / CCA | Coefficients/intercepts plus centering/scaling are structure/composition; fitted output-shape flags and remaining state are committed. |
| GLMs | Coefficients/intercepts plus loss/link are structure/composition. Fitted loss/link class and class count must agree with the parameter-derived loss. |
| SVC / NuSVC / SVR / NuSVR | Kernel parameters, support data, native/private coefficients, sparse flag, effective gamma and calibration are committed. Linear coefficients and calibration are structure/composition when available. Public and inference dual coefficients/intercepts must agree, including sklearn's binary sign reversal. Support counts/indices, coefficient/calibration dimensions and sparse storage agree. |
| Naive Bayes / QDA / neighbors / nearest centroid / MLP / semi-supervised / opaque meta-estimators | Their complete fitted statistics, data, class/output mappings, distance/activation state and nested models are committed in the manifest when structure is null. Statistics/likelihood/prior dimensions agree with classes/features; neighbor sample counts agree with fitted arrays. MLP layer sizes, output counts/activation and label-binarizer classes agree with public parameters/labels. Existing probability/count/trust constraints still apply. |
| Pipeline | All fitted steps, transformations, final estimator, order, names, classes and parameters are committed/claimed. Structure stays null; no transformed-feature visualization is inferred. |

Unsupported custom types, callable/type parameters, HistGradientBoosting storage,
multilabel/chain probability layouts and untrusted nested internals remain
rejections. For example, actual fitted `csr_matrix` neighbor state is outside the
unchanged trust list; merely declaring its loader as the trusted base `spmatrix`
does not make its state survive the native construction comparison.

## Work and allocation ceilings

Before skops' type audit or construction: at most 100,000 schema nodes, 100,000
CachedNode references, 100,000 expanded dictionary/list visits, and nesting 32.
Expanded work is calculated once per shared definition, with capped sums; cycles,
missing references and conflicting numeric/scalar definitions are rejected.
Sklearn parameter projection separately caps visits at 100,000 and ancestry at
16, within the existing 1 MiB/depth-16 hyperparameter contract.

Nested ZIP directories are count-checked before ZipFile creates entries: at most
100,000 outer skops entries and nine inner sparse NPZ entries. Numeric headers
have rank at most 32, positive item size, no object storage, dimensions at most
512 MiB, and exact `product(shape) * itemsize` bytes, checked against both ZIP
metadata and actual streamed bytes before NumPy loads anything. Inner sparse
archives have a closed member set, ratio at most 200 and checked format, shape,
index/pointer/block/coordinate/diagonal consistency. Sparse declared dense bytes
are at most 512 MiB. Aggregate decoded buffers and aggregate materialized buffer
references are each at most 512 MiB; repeating a file under new identities cannot
bypass this bound. All these are fixed ceilings in addition to archive limits.

Canonical JSON preflight calculates expanded minimum size/depth of shared native
containers with memoization before normalization, rejecting over-budget expansion
and cycles. Native arrays are size-checked before `tolist`. Wire JSON has no
references and remains bounded by its stored bytes/depth.

## XGBoost and ONNX

XGBoost's structure additionally requires `native_model`, the complete native
gradient-booster model returned by bounded UBJSON decoding of `save_raw`.
Numeric buffers become arrays without decimal rounding. It includes every tree
field, `tree_info`, iteration pointers, parallel-tree counts, category encodings,
categorical splits/default directions and optional dropout weights. Learner base
score and objective fields remain verified hyperparameters. Tree dumps alone are
insufficient to describe this prediction state.

Booster attribute keys are closed: `mlbundle`, `best_iteration`, `best_score`,
`scikit_learn`. The last three are preserved in `hyperparameters.attributes`;
the binding is independently checked. Best iteration is a decimal integer with
at most six digits, in `[0, native_round_count)`. Best score is finite and at most
256 characters; sklearn identity has only the native classifier/regressor key.
Unknown wrapper attributes and unknown/missing native record fields fail. Native
wrapper objective and supplied feature types must agree with native state. The
wrapper booster is null or a supported tree alias (`gbtree`, or `dart`, which
XGBoost 3.4 normalizes to gbtree). Wrapper declarations cannot reinterpret the
native predictor. Native
iteration increments equal `num_parallel_tree * max(1,num_class)`. Objective
and learner class counts agree; dropout weights have native tree count, are
finite and nonnegative. Category indices are at most 1,000,000; split segments
are checked and aggregate declared categorical bitsets are capped at 512 MiB.
Category dictionary segments and sorted-index permutations agree with stored
values; string offsets must address the actual stored byte vector.
UBJSON has nesting 32, at most 100,000 object collection items and 1,000,000
non-buffer values; numeric buffer counts require actual stored bytes.

ONNX has no additional mutable fitted state. The CPU session is created from the
exact original bytes; signatures, metadata, kind, labels and graph/environment
claims are recomputed from those bytes. No initializer override, custom function,
domain or model-provided session configuration is admitted. Its descriptor scan
has no shared-reference mechanism, bounds fields/depth before construction, and
tensor dimensions require actual stored values before a session. See
[the ONNX contract](format-onnx.md) for execution limits.

These checks are not a native-code sandbox. Runtime-computed allocations,
dependency memory safety and prediction row/CPU/process limits remain the host's
responsibility. No claim of exhaustive native-parser memory safety is made.
