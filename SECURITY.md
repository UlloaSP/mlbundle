# Security policy

MLbundle 0.2.1 has not had an independent security review;
do not accept bundles from untrusted parties. Format version 1 is not frozen.
It implements strict manifest validation, secure container reading/writing, metadata-only
`inspect`, export/load, sklearn, XGBoost, CatBoost and ONNX adapters, local schema inference,
explicit samples, prediction-state verification and the inspect/verify CLI.
The container validates ZIP/ZIP64 layout, bounds sources and actual decompression,
checks CRC and member integrity, and rejects unsafe members and polyglot wrappers.
The writer publishes complete validated files atomically where hard links work;
the exclusive-creation fallback cleans up on failure but exposes the destination
while copying and cannot guarantee crash-atomic publication. The version 1 contract
is defined in [docs/format.md](docs/format.md).

## Threat model

A runtime such as MLSuite will receive `.mlbundle` files uploaded by users.
Treat the entire file and every manifest field as attacker-controlled, even
when the archive appears to come from a trusted training workflow. An attacker
may try to execute code during loading, select arbitrary classes, exhaust
memory or CPU with an archive, expose training data, or supply metadata and
visualisations that do not describe the actual model.

The writer runs on the training machine and must reject unsupported models
there. The reader must independently validate uploads; writer checks do not
make a received file trustworthy.

## Required reader guarantees

- Never use pickle, joblib or cloudpickle, even as an option or fallback.
- Never import or instantiate from strings in the manifest. Classes must come
  from a closed list inside mlbundle.
- Keep the skops trusted-types list inside mlbundle. Never accept a list from
  the bundle or let its author expand the reader's trust boundary.
- Export must fail on the user's machine when a model contains custom types,
  listing unsupported types from `skops.io.get_untrusted_types`. This must be
  discovered before upload, not at runtime.
- Read zip content in memory by allowed member names; never extract to disk.
  Reject unknown and duplicate members, names with path separators or `..`,
  symlinks and encrypted entries. Limit member count, compressed size,
  uncompressed size and compression ratio before allocating or loading content.
- Validate local and central records and any data descriptors; require exact
  decompressed sizes and CRC. Reject comments, non-ZIP64 extra fields and bytes
  outside the contiguous ZIP layout. ZIP64 and validated streaming descriptors
  are accepted within the same resource ceilings.
- Verify each member's SHA-256 against the manifest and validate member sizes.
- Validate the manifest strictly and reject unknown format versions.
- Provide a metadata-only `inspect` operation that does not deserialize a model.
  It must not present model-derived claims as verified against the model.
- Treat every model-derived value as a claim. Recompute it from the loaded model
  using the same extraction code as the writer, and reject mismatches, including
  disagreements in visualisation structure.

The supported storage formats are skops for scikit-learn, native UBJSON for
XGBoost, native CBM for CatBoost and the user's original ONNX file. There is no conversion or fallback
serialization path. Adapters must use lazy imports for optional dependencies.

## Implemented sklearn trust boundary

Export and load call skops' `get_untrusted_types` and reject all types outside
mlbundle's fixed supplemental list: native Tree, AdamOptimizer/SGDOptimizer,
`numpy.dtype`, six sklearn univariate score functions and nine NumPy elementwise
functions. Load passes exactly that list to `skops.io.loads`, alongside skops'
built-in trusted defaults. Exact names and scalar types are in
[the state contract](docs/format-state.md). The bundle cannot add trusted types;
XGBoost and CatBoost add none to this list and use their own native adapters.

Skops 0.16.0 FunctionNode resolves `getattr(import_module(module), name)` without
calling the function. mlbundle checks exact module/name pairs and permits only
FunctionNode/CachedNode for these names, rejecting arbitrary functions and
alternative constructor loaders before skops. TypeNode resolves a NumPy scalar
class without constructing it; parameter extraction also checks native identity.
DTypeNode reads an empty array's dtype with `allow_pickle=False`; mlbundle bounds
and validates that representation before loading. Only closed plain scalar dtypes
and empty JSON object-dtype arrays are admitted; compound dtypes and hidden state
fail. Every new reference/value participates in the complete state commitment.

The optimizer path was checked against skops 0.16.0/sklearn 1.9.1: ObjectNode
uses ordinary `object.__new__` and `__dict__.update`, bypassing constructors.
Neither optimizer defines allocation, setstate or reduce hooks. Only ObjectNode
and shared CachedNode references are admitted for them; alternative constructor
loaders fail before skops. Their closed attributes and numeric buffers use the
existing pre-load resource bounds and full state commitment. MLP weights and all
fitted state remain committed, with no MLP visualization member.

Skops 0.16.0 deliberately distrusts tree storage because unchecked child and
feature indices can crash native inference. mlbundle checks loaded tree dimensions,
indices, reachability, depth and owning-estimator dimensions before computing
importances or predicting. Ensemble feature mappings must stay within parent
input widths. It bounds tree constructor dimensions and class counts before
native construction, and checks state-array shapes against those dimensions.
It also bounds the inner skops ZIP and validates every schema-referenced numeric
array header, including renamed files, before skops reads it. Zero-itemsize
arrays are rejected; object arrays never use pickle loading. Malformed native
payloads are wrapped in bounded `InvalidModelError` messages. Closed state catalogs and native construction comparisons reject extra or ignored
state and unreferenced files. Complete fitted-state commitments additionally bind
opaque pipelines and inference state outside tree/linear visualization. Shared-node
audit visits and materialized buffers are bounded before skops. Sparse NPZ headers
are checked against actual storage before NumPy allocation. These checks do not
make the native parser a sandbox.

The installed skops load path builds nodes, audits them and only then constructs
objects; NumPy buffers use `allow_pickle=False`. No pickle/joblib loading fallback
is used. Skops imports `pickle` only to recognize an unrelated save-side exception;
this does not deserialize pickle bytes. Custom estimators/functions and unsupported
native placements are rejected. Exact outer Pipelines ending in an admitted native
wrapper split transformer skops bytes from UBJSON/CBM, retaining each adapter's
preflight, trust, state/claims and structure checks; no booster type enters skops.
Closed fitted-state width proofs never use data. Shared bounded names/parameters/
commitments and inner tree coordinates are recomputed before closed Pipeline
reconstruction is returned. Cache/metadata routing/unknown outer state fail;
step names and outer options are declarations, not authenticated training history.
See [the composite contract](docs/format-pipeline.md).
HistGradientBoosting's separate
`TreePredictor` storage is also excluded. Broadening the supplemental list requires
source review and hostile-payload tests, never author approval or bundle metadata.

Both export and load recompute the same model claims and structures and compare
canonical bytes, preserving class-label types and order. Export additionally
compares the original and reloaded native models before publication. Loaded results
contain the native model for runtime explanations; mutating it invalidates the
relationship to the verified metadata. All-sklearn Pipeline structure stays null;
composites expose checked native trees in transformed feature coordinates.

The closed sklearn class/attribute catalog is audited against 1.9.1 with skops
0.16.0. Other versions/configurations can be rejected for unknown state. The
embedded sklearn version is committed too; different export/load sklearn
versions fail native state comparison. Dependency version floors are not a
cross-version compatibility guarantee. See [state ownership](docs/format-state.md).

## Implemented XGBoost dependency boundary

XGBoost uses native UBJSON only, in memory, with exact XGBClassifier/XGBRegressor
classes chosen from mlbundle's fixed map. The manifest never supplies loading
code. Export rejects custom objectives, other wrappers, raw boosters, unsupported
objectives/boosters and multi-target models. A native copy holds the canonical
mlbundle attribute binding class labels, wrapper identity and prediction settings;
the caller's model is never mutated. Load requires and recomputes that binding,
recoverable config fields, names/types, closed booster attributes and complete
native tree/routing/category/dropout state alongside dumps/importances.
Label agreement proves consistency with the native file, not the historical
truth of a user-supplied label mapping.

The adapter rejects oversized bytes and non-UBJSON before XGBoost sees them.
A bounded Python preflight checks document layout, feature/class/target counts,
tree counts, node arrays, output groups, iteration pointers, indices, reachability
and dump-compatible depth, keeping
numeric arrays as byte views. Native errors are translated to bounded mlbundle
errors without third-party exception text. These checks reduce cheap malformed
inputs; they are not a complete validation of XGBoost's internal model schema.
XGBoost's native parser and native inference remain a dependency boundary:
mlbundle cannot guarantee their memory safety or bound their internal memory/CPU
use. Native tree dump generation also allocates inside XGBoost before mlbundle
can inspect its returned byte size. Server consumers must isolate loading and
prediction and enforce process resource limits. All predictors reject invalid inputs and wrap remaining native failures
without echoing data cells. See the exact support and
resource boundaries in [docs/format.md](docs/format.md).

## Implemented CatBoost dependency boundary

Only exact fitted CatBoostClassifier/CatBoostRegressor, closed single-target
losses and numeric symmetric trees are admitted. CBM export uses public
`save_model` in an export-only temporary directory; load uses `load_model(blob=...)`
in memory. Extraction uses CatBoost 1.2.10's audited private `_serialize_model`
CBM serializer to recompute current native state without writing reader files.
Class selection and the prediction loss come from checked native state, never
manifest strings. Native class labels, parameters, feature order, calibration,
tree state and the complete native metadata commitment are recomputed on load.
The original fitted estimator is not modified. See [the contract](docs/format-catboost.md).

Before native parsing, mlbundle checks the exact CBM core length, version,
FlatBuffer offsets/fields/strings/vectors, tree and feature counts, symmetric
depth/split offsets, leaf dimensions/values/weights, split indices and numeric
borders/NaN routing. Metadata JSON is byte/depth-bounded. Numeric buffers remain
views during preflight; a conservative 256 MiB structure expansion budget is
checked before native parsing or Python list conversion. Categorical/CTR, text/embedding/calcer parts and
non-symmetric trees are rejected before their native parsers or traversals.
CatBoost's CTR table reader lacks its core FlatBuffer verifier; supporting it
requires a separate complete preflight, not merely permitting categorical input.

Windows 1.2.10 subprocess probes returned CatBoostError for truncated/bad-root
models and UnicodeDecodeError for corrupted metadata, and accepted a model
missing its final byte. mlbundle independently rejects inconsistent core lengths
and wraps native Python exceptions in bounded errors without their content.
These probes found no crash or timeout; they cannot prove memory safety.
The native loader, serializer, feature-importance traversal and inference can
still crash or allocate inside native code. Exceptions cannot contain process
crashes, and member/count ceilings do not cap native RAM or CPU. Consumers must
isolate these operations and enforce process limits, as for the other adapters.

## Implemented ONNX dependency boundary

Original self-contained ONNX bytes are parsed and loaded in memory. Before any
ONNX Runtime session, a descriptor-driven protobuf scan bounds nesting and field
counts, validates lengths and rejects unknown/duplicate singular fields. The
count ceiling is 536,870,912 (formerly 1,000,000), derived from the 512 MiB byte
ceiling: each header/packed scalar costs at least one distinct byte. It permits
large unpacked ensembles and does not bound decoded protobuf object memory.
Repeated scalars use bounded 64 KiB NumPy/C regex scans with the same wire checks; every
header/scalar still counts. One description is retained per export/load by
immutable bytes identity, never globally or from bundle metadata. A full
message traversal rejects every external tensor reference, including initializers,
attributes, sparse tensors, nested graphs and functions; checker and runtime never
receive external-data models. Tensor dimensions/storage and graph ranks/nodes are
bounded, and stored data must agree with declared element counts. Local functions,
training/device configuration, node overloads, overridable initializers and
`ort_config` metadata are rejected. Operators are confined to standard ONNX and
ONNX-ML domains and must pass the checker and CPU runtime. No custom op library
is registered. See [the normative ONNX rules](docs/format-onnx.md) for exact limits.

Sessions use CPU only, sequential execution, one thread per pool and disabled
optimization/prepacking, profiling, file output, arena and memory patterns.
Model-provided session configuration is disabled. The host must not enable
`ORT_LOAD_CONFIG_FROM_MODEL=1`; the pinned runtime rejects that combination.
All native exceptions exposed through the API become bounded mlbundle errors.
No model file is converted, rewritten, extracted or used to select a disk path.
Graph signatures, metadata, kind/counts, model dtypes, producer/opsets/IR and native
labels are recomputed and canonically compared. A label-free graph's original
class mapping remains an author declaration, checked only against native counts.
ONNX structure is null; an added visualization member is a claim mismatch.
Producer strings allow 65,536 characters each; bound failures identify a closed
field and its limit without exposing author text. Other string/projection bounds
remain as specified in the ONNX contract.

The protobuf parser, ONNX checker and ONNX Runtime execute native code. Resource
checks reduce malformed inputs; they do not establish memory safety or cap native
allocations, constant processing, runtime-computed shapes or control-flow execution.
Consumers must isolate loading/prediction and enforce memory, CPU and row limits,
and maintain their native dependencies. Optional extras retain this dependency
boundary: importing mlbundle or inspecting any bundle imports no model library.
The native result contains the exact bytes and CPU session needed for inference;
mutating a session invalidates its verified relationship with metadata.

## Limits of these guarantees

SHA-256 hashes detect corruption, not deliberate tampering: an attacker can
replace a member and update its hash in the manifest. They do not prove who
trained the model or authenticate its provenance. Bundle signing is out of
scope. Training-environment records and free-form user metadata are assertions
by the author; checking model-derived claims does not authenticate those records.

Validation does not prove model quality, fairness, training history or the
truth of user metadata. Pickle-free formats do not sandbox model execution or
eliminate vulnerabilities in native parsers and dependencies. A consuming
runtime still owns authentication, authorization, process isolation and
execution resource limits.

Training frames are never serialized. Schema inference runs locally and includes
column names, dtype strings, inferred kinds, nullability, categorical values
(including unused categories), one-hot suffixes, dtype-based numeric steps and
pair-axis metadata. It adds no numeric/date ranges, means, row counts, arbitrary
text values or example rows. These declarations may still be sensitive and are
not proof of the underlying training data. Native models, learned structure,
class labels and user metadata can independently reveal training information.
Omitting samples does not prevent model memorization or model-side disclosure.

Only explicit `sample_rows=N` includes real feature rows: exactly the first N,
without target or index. The schema/sample members have 16 MiB/depth-32 limits.
Schema v2 is required, inferred locally from training data or a synthetic default;
source/separator remain declarations. Older bundles require re-export. Inspect/load
validate mappings by feature origin and string reports by output position without
mlschema/pandas; native label verification retains original types. Load also verifies native model claims; label-free
ONNX original labels remain declarations. Consumers must treat schema and sample
values as untrusted display/input data, never executable instructions.
See [the exact data-derived contract](docs/format-schema.md) and review what is shared.

The CLI uses the same inspect/load boundary. Human summary values escape
terminal controls and are truncated; JSON output is lossless ASCII-escaped
manifest data within format limits. Errors are one bounded stderr line with a
failure category, not bundle values. Inspect still does not attest model claims;
verify requires the matching extra and performs full loading and comparison.

Record input looks up only the expected schema keys and ignores all extra keys
without iterating the record. It accepts finite Python numbers, booleans and
bounded decimal/true/false strings after ASCII whitespace stripping. `None`
becomes NaN only for schema-declared optional features. Errors escape and bound
feature names and never echo input values. These conveniences belong only to
`row_from_record`/`predict_record`; array/DataFrame validation is unchanged.
Version diagnostics use strict version text of at most 64 characters and appear
only after failure; environment declarations remain unauthenticated claims.
Classifier record results validate native class decisions against the output's
classes before returning the report label. Invalid dimensions, values or classes
raise bounded errors without values. A missing ONNX label output yields None;
mlbundle never derives a decision from probabilities.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting for UlloaSP/mlbundle](https://github.com/UlloaSP/mlbundle/security/advisories/new).
Include the affected commit, a minimal reproducer, the expected and observed
behavior, and the impact. Share malicious bundles or sensitive details through
that private report rather than a public issue or pull request. Remove personal
or confidential training data from any reproducer you submit.

Report the affected version and commit you tested.
