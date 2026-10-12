# mlbundle format version 1

This document defines the wire contract for `format_name = "mlbundle"` and
`format_version = "1"`. Format versions are independent of package versions.
"Must" denotes a compatibility or security requirement. Phases 1–5 implement
the manifest, encoding, limits, secure container, metadata-only inspection,
public export/load and the sklearn, XGBoost and ONNX adapters.
Local schema inference and explicit samples are implemented; the CLI is missing.

The normative container rules are in [Container and resource limits](format-container.md).
The normative structure, adapter and result rules are in [Adapters and structure](format-adapters.md).
The implemented ONNX mapping and validation rules are in [ONNX](format-onnx.md).
The normative schema/declaration rules are in [Schema and samples](format-schema.md).
[Prediction state and bounded work](format-state.md) is also normative.
These sections are part of this version 1 contract.

## Members

| Member | Rule and contents |
| --- | --- |
| `manifest.json` | Required; the manifest described below. |
| `model.skops` | Exactly one model member; scikit-learn, skops format. |
| `model.ubj` | Exactly one model member; XGBoost native UBJSON format. |
| `model.onnx` | Exactly one model member; original, self-contained ONNX bytes, unchanged. |
| `schema.json` | Optional; exact object with `schema_version`, `fields` and `reports` from local inference. See [Schema and samples](format-schema.md). No training DataFrame. |
| `structure.json` | Optional only for families without extractable structure; object as defined below. |
| `samples.json` | Optional; array of row objects keyed by feature name, in declared feature order when consumed. Only explicit user intent may enable it. Values use the canonical JSON value rules. Samples are real user data. |

Presence of schema, structure or samples must match both the manifest reference
and an integrity record. There is no member for arbitrary user files. A training
DataFrame must never be included. For samples, each row must contain exactly the
declared feature names, without a target column or arbitrary extra columns; the
byte/depth limits apply even when no rows are present.

For version 1, structure extraction is mandatory for supported scikit-learn
models exposing `tree_`, tree ensembles with constituent `tree_` estimators,
models exposing `coef_`/`intercept_`, and models exposing `feature_importances_`;
and for supported XGBoost tree boosters. A member is required whenever the
adapter's extraction returns an object. Other supported sklearn/XGBoost families
return null; ONNX is opaque for visualization in version 1 and returns null.
Do not manufacture a visualization for an opaque model. The loaded reader must
extract structure even when the member is absent: non-null extraction with a
missing member, or null extraction with a present member, is a claim mismatch.
Metadata-only inspection cannot determine this requirement from the model.

## Manifest

All objects representing typed records reject unknown keys. Types are strict:
booleans do not stand for integers, strings do not stand for numbers and enum
values are exact. JSON object keys are strings. Arrays are ordered. Null is
distinct from an empty array/object. Unless marked optional, fields are required.
Optional fields can be omitted or null; writers emit defaults explicitly.
`user_metadata` may be omitted and defaults to `{}`.

### Identity

| Field | Type | Meaning |
| --- | --- | --- |
| `format_name` | literal string `"mlbundle"` | Format identity. |
| `format_version` | literal string `"1"` | Wire version; unknown strings raise `UnsupportedFormatVersionError`. Wrong types are invalid manifests. |
| `bundle_id` | UUID string | Author-assigned bundle identity; writers use lowercase, hyphenated UUIDs. Readers accept UUID spellings supported by UUID parsing and normalize on output. |
| `created_at` | ISO 8601 UTC timestamp string | Creation declaration, spelled `YYYY-MM-DDTHH:MM:SS[.ffffff]Z` or with `+00:00` instead of `Z`; fractions have 1–6 digits. Writers emit `Z`. Reject epoch strings/numbers, naive timestamps, nonzero offsets and extra fractional precision. |
| `writer_version` | nonempty string, at most 256 characters | mlbundle package version that wrote the bundle. Must equal `environment.packages.mlbundle`. |

Identity fields are declarations, not proof of provenance.

### `model`

| Field | Type | Meaning / verification |
| --- | --- | --- |
| `library` | `"sklearn"`, `"xgboost"`, `"onnx"` | Closed adapter identifier; validated against serialization/member pairing. |
| `serialization` | `"skops"`, `"ubjson"`, `"onnx"` | Must pair respectively with the three libraries and model members above. |
| `estimator_class` | nonempty string, at most 256 characters | Verified class name (not a Python import path); ONNX uses the adapter's fixed `"ONNX"` identity. Never instantiate from this field. |
| `kind` | `"classifier"`, `"regressor"` | Verified prediction kind. |
| `hyperparameters` | object, at most 4,096 top-level keys | Verified JSON projection of the supported estimator's parameters. Sklearn uses `get_params(deep=True)`; other adapters use their recoverable native parameter representation. ONNX records graph signatures and metadata properties as defined in [ONNX](format-onnx.md), without inventing training settings. At most 1 MiB canonical bytes and depth 16. |
| `n_features` | integer 1–100,000 | Verified ordered input width. |
| `n_classes` | null or array of integers 1–100,000 | Verified per-output class counts; classifier array length equals `n_outputs`, regressor value is null. |
| `n_outputs` | integer 1–1,024 | Verified target output count. Probability columns are classes, not separate target outputs. |

Hyperparameter values preserve JSON scalars, nested containers and numeric tags.
Do not stringify unsupported objects or call `repr` as a fallback. If an adapter
supports nested estimators in parameters, their representation is an object
`{"$estimator":{"class":CLASS_NAME,"params":PARAMETER_OBJECT}}`, recursively,
with types admitted by the adapter's trust boundary. This is a comparison record,
never an instruction to import or construct a class. Unsupported callable,
custom-object or otherwise unrecoverable parameters must fail export.
The writer must serialize, reload and compare extracted claims and structure
before emitting a bundle. Native formats may omit training-time parameters:
export must fail when the chosen claims cannot survive that round trip; the
manifest must never restore missing parameters into a reader's model.

Sklearn parameters preserve `None`, JSON booleans, strings, integers and floats;
non-finite floats use the canonical tags. NumPy scalars/arrays use the canonical
conversion below; tuples become arrays, and dictionaries require string keys.
Nested sklearn estimators use the `$estimator` record with their class name and
recursively projected deep parameters. Deep keys such as `classifier__C` are
retained alongside the estimator record. A NumPy `RandomState` becomes
`{"$random_state":[ALGORITHM,STATE_ARRAY,POSITION,HAS_GAUSSIAN,CACHED_GAUSSIAN]}`
from its current `get_state()`, after training. No initial seed is guessed.
Callables (including built-in functions, NumPy functions, lambdas and partials),
types, cyclic containers, byte strings, complex/datetime values and other custom
objects are rejected. There is no `repr`, identity/address or stringification
fallback. A skops-trusted object can still be rejected if its parameters have no
faithful projection. The round-trip comparison checks all these records.

### `features`

| Field | Type | Meaning / verification |
| --- | --- | --- |
| `names` | array of 1–100,000 unique nonempty strings, each at most 1,024 characters | Effective input names, in prediction order; length equals `n_features`. |
| `dtypes` | equally sized array of null or nonempty strings of at most 256 characters | Training DataFrame dtype declarations in feature order; null means unknown. They do not prove model input types. |
| `model_dtypes` | optional equally sized array of nonempty strings of at most 256 characters | Verified dtypes exposed by the native model, or null if it exposes none; separate from training dtypes. ONNX uses normalized NumPy dtype names such as `float32`, `float64`, `int32`, `int64`. |
| `origin` | `"model"`, `"dataframe"`, `"declared"`, `"generated"` | Native names, local frame columns, explicitly supplied names, or generated names. |

If a model exposes feature names, origin must be `model` and names must equal
its extracted names in order, even when a frame supplies training dtypes.
Otherwise `dataframe` means columns from `training_data=`, and `declared` means
`feature_names=` without a frame. Both are author declarations: the reader checks
count/alignment, not historical truth. With neither, origin is `generated` and
names must be `feature_1` through `feature_N`. Claim extraction represents missing
model feature names as null, so falsely declaring a named model's origin as
`dataframe`, `declared` or `generated` also fails comparison. Model dtype extraction is
independent of name origin and must be compared even when its claim is null.

### `target`

| Field | Type | Meaning / verification |
| --- | --- | --- |
| `class_labels` | null or array of label arrays | Verified original labels in prediction-column order, **one array per target output**, even for a single output. Regressors use null. Each output has 1–100,000 unique labels and its length equals the corresponding class count. |
| `name` | optional nonempty string, at most 1,024 characters | Target name when known; an author declaration. |

Labels are strict strings (including an empty original label), integers, finite
binary64 numbers or booleans. Their
types are preserved; `1`, `1.0`, `true` and `"1"` have distinct canonical bytes.
NaN/infinite class labels are unsupported. String labels are at most 1,024
characters. Do not convert labels to display strings in the bundle or silently
reorder them; a consumer may format them for display separately.

`export(class_labels=...)` accepts output-major sequences, including a nested
sequence for a single output. Supplied labels must use one scalar type per output,
be unique by canonical bytes, and match native class/output counts. Regressors
reject this option. Sklearn rejects it even when identical: its native `classes_`
owns the verified labels. XGBoost wrappers expose encoded `0..N-1` classes;
omitted labels use those integers (or retain an existing mlbundle binding).
Explicit labels map those indices to original labels in probability-column order.
The adapter binds them as canonical JSON in the native booster attribute
`mlbundle`, on a copy only. Loaded XGBoost classifiers must carry that binding;
manifest labels are verified against it. This verifies agreement with the native
file, not the historical truth of the author's original-label mapping. Native
wrapper `predict` still returns encoded indices; the bundle predictor returns
probabilities aligned with the bound labels.

For an ONNX graph with no native labels, extraction may return null labels.
The shared pipeline then uses supplied labels, or generates `0..N-1` per output,
and verifies their lengths against native class counts. Such labels are
declarations, not verified original labels; the ONNX bytes must stay unchanged.
Graphs exposing labels must return them as claims for exact comparison. ONNX
must reject supplied labels that disagree with native labels; the shared resolver
enforces this comparison. Kind is verified from the supported output layouts; class counts come from
fixed probability width or native labels for an unknown width/ZipMap.
See [ONNX kind and labels](format-onnx.md#kind-and-labels).

### `environment`

| Field | Type | Meaning / verification |
| --- | --- | --- |
| `python_version` | nonempty string, max 256 | Training Python version. |
| `python_implementation` | nonempty string, max 256 | Implementation, e.g. `CPython`. |
| `os` | nonempty string, max 256 | OS name. |
| `os_version` | nonempty string, max 256 | OS release. |
| `architecture` | nonempty string, max 256 | Machine architecture, e.g. `AMD64`. |
| `packages` | object with only the keys listed below | Installed distribution versions, nonempty strings max 256. Uninstalled distributions are omitted. |
| `onnx` | optional ONNX record | Required for ONNX, null/absent otherwise; verified from the graph. |

Allowed package keys are `mlbundle`, `mlschema`, `numpy`, `pandas`, `scipy`,
`scikit-learn`, `xgboost`, `onnx`, `onnxruntime` and `skops` (maximum ten).
Capture versions using distribution metadata, without importing the libraries.
`mlbundle` is required by the writer-version consistency rule. Other environment
fields are unverifiable declarations about the training machine, including the
ONNX/ONNX Runtime package versions; they are not evidence of compatibility.

The ONNX record contains required fields: `opsets` (1–128 objects with `domain`,
a string of 0–256 characters, and `version`, an integer 1–2^31−1), `ir_version`
(integer 1–2^31−1), `producer_name` and `producer_version` (strings 0–256).
Opset domains are unique and sorted by Unicode code point. All these graph
fields are verified claims. Matching producer text proves agreement with the
file, not who produced or trained it.

### `integrity`, `references` and `user_metadata`

`integrity` is an object mapping **every other present member** to exactly
`{"sha256":HEX,"size":BYTES}`. The digest is 64 lowercase hexadecimal characters;
size is an integer 0–512 MiB, further bounded by the specific member cap.
It has 1–4 keys: exactly the selected model and referenced optional members.
There is no manifest hash, since the manifest cannot contain its own digest.
Declared sizes must equal both actual uncompressed sizes and ZIP metadata.

`references` is a required object with optional `schema`, `structure` and
`samples` fields. Each is null/absent or respectively the literal `"schema.json"`,
`"structure.json"`, `"samples.json"`. There are no URLs or arbitrary paths.

`user_metadata` is a free JSON object for name, description, author, tags,
training metrics, dataset description or other annotations. Maximum 1,024
top-level keys, 64 KiB canonical UTF-8 bytes and depth 8. Nested collection and
string sizes are bounded by that byte budget. Metadata is entirely declarative;
unknown keys here are permitted and never affect adapter selection or trust.

## Canonical JSON and claim comparison

All JSON writers use the following deterministic encoding. Readers accept other
valid whitespace/key orders but always normalize for comparison. Reject duplicate
keys at every depth, invalid UTF-8, unpaired Unicode surrogates, malformed JSON,
bare NaN/Infinity tokens, numbers overflowing binary64, and excessive size/depth.
Check byte size before UTF-8 decoding and lexical container depth before parsing.
Depth counts nested objects/arrays (root container is depth 1, scalar is depth 0).
The default decoder bound is 8 MiB/32; use each member's explicit byte cap for
larger JSON members. The typed manifest remains subject to its tighter budgets.

1. Encode UTF-8 without BOM or a trailing newline, with no insignificant spaces.
2. Sort object keys lexicographically by Unicode code point; preserve array order.
   Do not apply Unicode normalization. Non-ASCII characters are emitted directly.
   Escape quotes, backslashes and control characters exactly as Python's
   `json.dumps(ensure_ascii=False, separators=(",", ":"))` does: short escapes
   for backspace, tab, newline, form feed and carriage return, otherwise lowercase
   four-digit `\u` escapes for U+0000–U+001F. `/` is not escaped.
3. Null, booleans and integers use JSON literal spellings and base-10 integer
   representation. Integers are arbitrary precision up to 4,096 decimal digits
   (excluding a minus sign); longer integer tokens or values are rejected.
4. Finite floats use CPython's shortest round-tripping binary64 `repr` spelling,
   including `.0` for integral floats and `-0.0` for negative zero. Use lowercase
   `e`, an explicit sign and at least two exponent digits in scientific notation;
   scientific notation is used when decimal exponent is below −4 or at least 16.
   Examples: `0.1`, `1.0`, `-0.0`, `1e-07`, `1e+20`, `5e-324`.
   This is the mlbundle encoding, not RFC 8785/JCS. Readers in other languages
   must preserve integer/float distinctions and use the same numeric spelling.
5. Native NaN and infinities are encoded as exact one-key objects
   `{"$float":"nan"}`, `{"$float":"+inf"}`, `{"$float":"-inf"}`. `$float` is
   reserved: additional keys or another value are rejected. Decoding retains the
   tagged objects; it does not create non-finite JSON floats. NaN payload/sign
   distinctions are intentionally discarded; finite values round-trip exactly.
6. Writer-side NumPy bool/integer/real/string scalars map to their Python scalar
   values; arrays map via `tolist()` preserving dimensions and index order.
   float32 is promoted exactly to binary64 before encoding, without rounding to
   a decimal display precision. Complex, datetime, byte-string and other values
   lacking a supported scalar representation fail. Tuples encode as arrays.

SHA-256 is computed over the **stored uncompressed member bytes**, with size
measured in bytes. Writer-generated JSON member bytes use this encoding; native
model bytes are hashed unchanged. Readers verify original bytes before parsing,
not a re-encoding that could conceal corruption. For verified claims and structure,
compare canonical encoded values for exact byte equality, with no tolerance,
unordered-array comparison or coercion. Null/absent extracted information is
compared too. Unknown-field, count/alignment and reference checks are distinct
from recomputing claims: plausible metadata alone never verifies a model.

The manifest's claim projection is `ModelClaims`: estimator class, kind,
hyperparameters, feature/class/output counts, model-origin feature names (null
otherwise), model dtypes, original output-major class labels and ONNX graph
metadata and the optional fitted-state commitment. Classifier extraction may leave labels null only when the native model
carries none; the shared pipeline resolves the declaration as specified above.
Library/serialization/member pairing is validated by the closed
registry. Structure is compared separately using the same canonical encoding.
Schema, training dtypes, frame/declared-origin names, target name, training environment,
user metadata, identity and samples are not verified against the model; they must
still pass their own validation and integrity checks. Inspection verifies format
and integrity only, and must never attest that model claims match the model.

## Versioning and integrity

Readers must reject unknown format versions rather than guess. Any change to
required fields, accepted identifiers, member rules, claim semantics, numeric
encoding or mandatory structure shapes requires a new format version and an
explicit reader implementation. Tightening caller-selected resource limits does
not change the format. Corrections to documentation that preserve wire behavior
and implementation changes behind these contracts do not require a version bump.
Phase 5 completes the unreleased v1 schema vocabulary and adds `declared` origin.
Phase 6 completes unreleased v1 prediction-state records, missing tree routing,
native XGBoost routing/dropout and closed state/work rules without a version bump.

Hashes detect corruption, not deliberate tampering: an attacker can replace a
model and update every record. Claim recomputation prevents a bundle from showing
metadata/structure different from its prediction model; it does not establish
training history, model quality, fairness or author identity. Signing is out of
scope. See [SECURITY.md](../SECURITY.md) for the complete threat model.
