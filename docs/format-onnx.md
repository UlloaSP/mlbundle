# Version 1 ONNX mapping

Normative section of [the format contract](format.md). Phase 4b implements this
mapping. ONNX is opaque for visualization: extraction returns null, and load
rejects any `structure.json` member with `ClaimMismatchError`.

## Original model and native result

Export accepts `.onnx` paths (`str` or `PathLike`, case-insensitive suffix), raw
`bytes`, `onnx.ModelProto`, and a previously validated `mlbundle._onnx.OnnxModel`.
Paths are stat-bounded and read with one excess byte for detection. Paths and
bytes are stored unchanged; the model member hash is the original bytes' SHA-256.
A ModelProto has no original file representation: serialize it once using
`SerializeToString()` without changing the object, then preserve those bytes.
There is no conversion, graph editing, external-data loading or file reserialization.
`prepare_export` validates declarations but cannot bind them into immutable bytes.

`native_model` is the frozen `mlbundle._onnx.OnnxModel` record, with `data` (exact
original bytes) and `session` (an `onnxruntime.InferenceSession`). A runtime uses
`session.get_inputs()`, `get_outputs()` and `run()` for native inference; the
bundle predictor provides validated plain-list probability/regression output.
The session and nested values are not deeply immutable. Mutating them invalidates
the loaded-result guarantee, as with other adapters.

## Inputs and outputs

Inputs are rank-two numeric tensors, dtype `float32`, `float64`, `int32` or `int64`,
with a fixed width from 1 through 100,000. Batch dimension is unknown, symbolic
or exactly one. Multiple inputs require width one each, and their graph input
names are verified feature names in graph order. A single matrix input supplies
no column names; supplied/generated names remain declarations. Verified model
dtypes repeat the matrix dtype per column or preserve the column input dtypes.

Only one target is supported. Regression has one float32/float64 output of shape
`[batch]` or `[batch, 1]`. Classifier probabilities are a float32/float64 matrix
`[batch, classes]` or a sequence of maps with int64/string keys and floating values.
An optional second output is an int64/string label tensor `[batch]` or `[batch, 1]`.
Batch declarations in outputs must also be unknown/symbolic or exactly one.
Classifier output selection uses these types/shapes, never output name heuristics.
Multi-target regression, image/string/sequence inputs, multi-column multiple inputs,
fixed batches above one and unsupported/indeterminate outputs fail before sessions.

Predictors accept row lists, NumPy arrays and DataFrames, importing pandas only
when a frame is passed. Arrays/lists are positional. Multiple-input DataFrames
must have exactly the feature width and unique model input columns; they are
selected by graph input name, permitting another column order. Single matrix
frames are positional. Integer conversion preserves Python/NumPy integer values,
requires integral finite numbers within the dtype range, and never silently
truncates fractions. Boolean/string/object values and non-finite or overflowing
numeric inputs fail. Static one-row batches require exactly one prediction row.
Output dimensions and finiteness are checked. Probabilities must lie in `[0,1]`
and sum to one (`rtol=1e-6`, `atol=1e-5`); ZipMap must contain exactly the native
labels and is read in their recorded order. No missing probabilities are invented.

## Kind and labels

Kind is a verified claim, with no `kind=` declaration. Two compatible outputs
(probabilities plus labels) determine classification. A single probability output
requires native classifier/ZipMap labels in its upstream graph, or a terminal
standard Softmax over the class dimension (`axis=1` or `-1`, optionally followed
by Identity/Cast). Otherwise only a single scalar-per-row floating output determines
regression. Other graphs are rejected rather than guessing from user metadata.
Single-output Softmax graphs without labels require a fixed class width.

Native labels come from the output's `ai.onnx.ml` ZipMap, LinearClassifier,
TreeEnsembleClassifier or SVMClassifier `classlabels_strings`, `classlabels_ints`
or `classlabels_int64s` attributes. Decode strings as UTF-8; preserve integers,
types and native column order. Identity/Cast can preserve a classifier's probability
columns; unrepresented transformations of labeled probabilities are rejected.
Final ZipMap labels own the map keys, permitting a different mapping from an
upstream classifier's encoded labels. Unrelated graph nodes cannot supply labels.
Fixed probability width determines
class count; otherwise native labels determine it. Width and native labels must
agree when both exist. Labels must satisfy the shared uniqueness/count/bounds rules.
Native labels are verified against the unchanged model; supplied labels must agree.
Absent native labels remain null in extraction. The shared pipeline resolves
supplied labels or generates `0..N-1`, checking native counts. Such labels are
declarations, and load cannot authenticate the original-label mapping.

## Claims

Estimator identity is exactly `ONNX`; output count is one and structure is null.
`hyperparameters` is the verified graph projection with exactly `inputs`, `outputs`
and `metadata_props`. It describes available graph facts, not training parameters.
Inputs/outputs are graph-ordered arrays of objects with `name`, `dtype`, `shape`.
Tensor shapes preserve integer static dimensions, symbolic dimension strings and
null unknown dimensions. Dtypes use normalized names above (plus `string` for
label tensors). Map output dtype is exactly `sequence<map<int64,float32>>`,
`sequence<map<string,float32>>` or the corresponding float64 spelling; shape is null.
`metadata_props` maps each unique model-level metadata key to its original string
value, without interpreting it as configuration. The existing 1 MiB/depth-16
hyperparameter ceiling applies. This completes the previously unimplemented ONNX
projection; sklearn/XGBoost claim semantics and the manifest schema are unchanged.

`environment.onnx` retains exactly the existing required opsets (unique, sorted
by original domain), IR version and producer name/version. These fields, graph
signatures, counts, kind, names/dtypes and native labels are all recomputed with
the writer's extraction code and compared by exact canonical bytes. Historical
environment, training dtypes, supplied matrix feature names, target name and
label-free mappings remain declarations.

## Before any native session

Both export and load perform the same checks, in this order:

1. Bound bytes to the 512 MiB native-member ceiling (archive/caller limits also
   apply). Scan protobuf wire fields through the installed ONNX descriptors
   before construction: depth at most 32, at most 1,000,000 fields/scalar values,
   valid lengths/wire types/varints/packed fields, no unknown fields or duplicate
   singular fields. This rejects hidden content and parser amplification tricks.
2. Parse in-memory with ModelProto. Visit every message, including sparse tensors,
   tensor attributes, nested graphs and functions. Reject any tensor with an
   external-data entry or non-default data location, segments or unsupported
   storage type. Supported storage types are the original ONNX scalar types
   FLOAT through BFLOAT16; newer packed/float8 formats are outside this boundary.
3. Check tensor rank (at most 32), nonnegative dimensions, dimension/element/byte
   budgets, exact raw-data length or typed scalar count and exclusive storage.
   Fixed-width decoded storage and its aggregate are capped at 512 MiB; sparse declared
   dense storage is also bounded. Bound static value dimensions and symbolic names
   (1,024 characters). Count at most 100,000 nodes across all nested graphs.
4. Permit only standard ONNX domains `""` (the wire spelling of `ai.onnx`),
   `ai.onnx` and `ai.onnx.ml` for nodes/opsets/functions; checker/runtime must also
   recognize the actual operators and versions. Reject local functions entirely,
   including standard-domain shadowing, training info, device configuration,
   node overloads and overridable root initializers. Reject model `ort_config`.
   No custom-op library is registered.
5. Run `onnx.checker.check_model(..., full_check=False)` in memory; shape inference
   and constant propagation are not requested. Validate the supported tabular
   layout and build bounded strict claims before creating any runtime session.
   Terminal LinearRegressor/TreeEnsembleRegressor native target counts must be one.

Create ONNX Runtime from those bytes with CPUExecutionProvider only, sequential
execution, one intra/inter-op thread, disabled graph optimization/prepacking,
CPU arena and memory patterns. This limits load-time work and retained buffers;
it is not a bound on execution memory/CPU. Profiling and optimized-model output
are disabled, model format is explicitly ONNX and `read_config_from_model=False`.
No files are written. The runtime host must not set `ORT_LOAD_CONFIG_FROM_MODEL=1`:
the pinned runtime rejects this combination; failures are bounded mlbundle errors.
Both export and load must successfully create a session, then compare its input/
output types, ranks and known dimensions with graph declarations. A declared
static dimension must remain exact in runtime metadata; unknown inferred width
cannot conceal disagreement. Runtime probability width must match native class
count when exposed. ONNX/checker/runtime
failures become bounded `InvalidModelError`; resource ceilings become
`LimitExceededError`. Missing dependencies name `mlbundle[onnx]`.

ONNX/protobuf/checker and ONNX Runtime are native dependency boundaries, not
sandboxes. Runtime-computed shapes and control flow can allocate or consume CPU
beyond graph declarations. Consumers still need process isolation, memory/CPU
limits, prediction row budgets and maintained native libraries.
