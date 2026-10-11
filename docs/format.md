# mlbundle format version 1

This document defines the wire contract for `format_name = "mlbundle"` and
`format_version = "1"`. Format versions are independent of package versions.
“Must” denotes a compatibility or security requirement. Phases 1 and 2 implement
the manifest, encoding, limits, adapter contracts, secure container reading and
writing, and public metadata-only inspection. Model adapters, public export/load
and schema inference are implemented in later phases.

## Container and resource limits

A bundle is one ZIP file with the `.mlbundle` extension. Its root contains only
the case-sensitive names below. Names are ASCII, with no directory entries,
slashes, backslashes, `..`, NULs or alternative spellings. Unknown or duplicate
names, symlinks, other non-regular entries, encrypted entries, unsupported
compression methods, multi-disk archives and malformed ZIP metadata must be
rejected. Only ZIP_STORED (method 0) and ZIP_DEFLATED (method 8) are allowed.
ZIP64 is allowed within the limits, including forced ZIP64 local records for
small files. Only a single ZIP64 extra field (ID 0x0001) is permitted in a header,
and it must contain exactly the replacement values required by that header's
sentinel fields, in ZIP order. Reject other, redundant, duplicate or malformed
extra fields. ZIP64 end records have the standard fixed 44-byte body, with no
extensible data sector; their locator and ordinary end record must agree.

Data descriptors are allowed for streaming ZIP writers. Accept the signed and
unsigned 32-bit and ZIP64 forms only when CRC and both sizes match the central
directory. With a descriptor, local CRC/sizes may be zero or match the directory;
ZIP64 sentinel sizes must be resolved through the local ZIP64 extra field.
Without a descriptor, local CRC and sizes must match the directory exactly.
The version-needed field is 10 (stored only), 20 or 45; ZIP64 headers use 45.
Allowed general-purpose flags are the UTF-8 name flag, the descriptor flag and
the two deflate-option bits (deflated entries only). Reject all other flags.

Reject archive and member comments. The first local header starts at byte zero;
local records, payloads and any descriptors cover the archive contiguously up
to the central directory. The directory ends exactly at the end record or
ZIP64 end record, followed only by its locator and the ordinary end record.
Reject prepended/appended bytes, gaps, overlapping records and unlisted local
records. Central entries may list local records in another order. These rules
exclude polyglot wrappers and metadata the mlbundle writer never produces.
File attributes may omit a Unix file type (as ordinary ZIP writers do); an
explicit type must be regular, and the DOS directory bit must be clear.

Readers must bound the source before buffering it: stat filesystem paths and
read streams in bounded chunks with at most one excess byte for detection.
Streams are consumed from their current position and are not closed by mlbundle.
Readers must read named members into memory, never extract them to disk. They
must inspect directory metadata and enforce limits before reading, allocating
member buffers or loading models; they must also limit actual bytes produced
during decompression, rather than trusting directory sizes. Reject inconsistencies
between headers, directory entries, actual sizes and ZIP CRC checks. No path,
module name, trusted-types list or external file reference can select loading
code. ONNX external tensor data is unsupported: the original file must be
self-contained. Native model formats do not sandbox their parsers or execution.
Decompression uses chunks of at most 64 KiB and a one-byte lookahead beyond the
declared output size; the lookahead is rejected before it is retained. Require
the exact output size, CRC and complete deflate stream, with no trailing packed
data. Declared sizes are checked against member, aggregate and ratio limits first.

One MiB is 1,048,576 bytes. `ArchiveLimits` defines these defaults and ceilings:

| Limit | Maximum |
| --- | ---: |
| Member count, including manifest | 5 (minimum configured value 2) |
| Entire archive, including headers | 512 MiB |
| Sum of compressed member sizes | 512 MiB |
| Any uncompressed member | 512 MiB |
| Sum of all uncompressed member sizes, including manifest | 1 GiB |
| Each member's uncompressed/compressed ratio | 200 |
| `manifest.json` | 8 MiB |
| `structure.json` | 256 MiB |
| `schema.json` | 16 MiB |
| `samples.json` | 16 MiB |
| JSON container nesting depth | 32 |

The effective member cap is the smaller of its specific cap and the general
member cap. Check the ratio as `uncompressed_size <= 200 * compressed_size`;
an empty member with both sizes zero is exempt, a nonempty member with compressed
size zero is invalid. This applies to stored members too. The mlbundle writer
always uses deflate level 6 and rejects inputs exceeding the ratio ceiling;
the reader also accepts stored entries from other tools. These limits admit
forests whose native model is tens of MiB and whose JSON tree arrays are larger.
Callers may lower archive limits, never raise the ceilings in version 1.
Consumers still need process memory, CPU and prediction limits; a 1 GiB byte
budget is not a guarantee that a parsed model uses at most 1 GiB of RAM.

### Container functions and inspection

The internal functions are `mlbundle.container.read_container(source, *,
limits=None)` and `mlbundle.writing.write_container(manifest, members, *,
destination=None, overwrite=False, limits=None)`. Reading returns a validated
`Manifest` and a dictionary of original uncompressed bytes keyed by `MemberName`,
including `manifest.json`. Source is a path, bytes or a binary stream. It never
deserializes a native model or resolves an adapter.

Writing accepts a `Manifest` or JSON-ready manifest parts, plus non-manifest
member bytes. The writer computes and replaces all integrity records; callers
must already set the model pairing and optional references. JSON-ready parts
may omit integrity. Optional JSON members pass the bounded canonical decoder;
their original bytes are preserved. The container validates JSON syntax and
bounds, while later model/schema APIs own semantic structure and schema checks.
Metadata inspection does not infer whether a model requires structure.

The writer emits manifest, model, schema, structure, samples in that fixed order,
omitting absent members. ZIP timestamps are 1980-01-01 00:00:00, the creating
platform is Unix, permissions are regular-file 0600, compression is deflate
level 6, and extra fields/comments are empty. It creates no descriptors or ZIP64
records within the version 1 ceilings. Identical inputs produce identical bytes
on Windows and Linux; the compression implementation is the standard zlib
provided by Python (byte identity across different future zlib implementations
is not a format requirement).

Writing returns bytes and optionally publishes them to a path after validating
the complete archive through the reader. Publication uses a flushed temporary
file in the destination directory: atomic replacement when overwrite is true,
or atomic hard-link creation otherwise, followed by temporary-file cleanup.
Hard-link publication requires a filesystem supporting hard links; otherwise
the filesystem error is raised and no partial destination is left. Existing
destinations raise `FileExistsError` unless overwrite is explicitly requested;
the no-overwrite decision also holds if another writer wins a race.

Public `mlbundle.inspect(source, *, limits=None)` returns `MetadataView` with
the manifest and fixed-order member listing. It validates ZIP records, limits,
manifest, member hashes and JSON syntax without importing model libraries.
It cannot attest that model-derived claims or structure match the model; only
a future full validating load can do that. Samples must still be explicitly
requested by the caller of the future export API.

Container/layout/CRC inconsistencies raise `InvalidContainerError`; resource
ceilings raise `LimitExceededError`. Invalid manifest JSON or typed fields raise
`InvalidManifestError`; unknown version strings under the mlbundle identity
raise `UnsupportedFormatVersionError` before validating version-specific fields.
Invalid optional JSON raises `InvalidJSONError`. Member hash or integrity-size
disagreement raises `IntegrityMismatchError` naming the allowed member.
Messages do not include untrusted names, JSON blobs or ZIP exception text.

## Members

| Member | Rule and contents |
| --- | --- |
| `manifest.json` | Required; the manifest described below. |
| `model.skops` | Exactly one model member; scikit-learn, skops format. |
| `model.ubj` | Exactly one model member; XGBoost native UBJSON format. |
| `model.onnx` | Exactly one model member; original, self-contained ONNX bytes, unchanged. |
| `schema.json` | Optional; object with `fields` and `reports` arrays from local schema/report generation. No training DataFrame. The schema integration phase defines the supported mlschema field/report vocabulary. |
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
| `hyperparameters` | object, at most 4,096 top-level keys | Verified JSON projection of the supported estimator's `get_params(deep=False)`; native models without `get_params` use their recoverable native parameter representation. ONNX uses `{}`. At most 1 MiB canonical bytes and depth 16. |
| `n_features` | integer 1–100,000 | Verified ordered input width. |
| `n_classes` | null or array of integers 1–100,000 | Verified per-output class counts; classifier array length equals `n_outputs`, regressor value is null. |
| `n_outputs` | integer 1–1,024 | Verified target output count. Probability columns are classes, not separate target outputs. |

Hyperparameter values preserve JSON scalars, nested containers and numeric tags.
Do not stringify unsupported objects or call `repr` as a fallback. If an adapter
supports nested estimators in parameters, their representation is an object
`{"$estimator":{"class":CLASS_NAME,"params":PARAMETER_OBJECT}}`, recursively,
with classes drawn from its own closed allowlist. This is a comparison record,
never an instruction to import or construct a class. Unsupported callable,
custom-object or otherwise unrecoverable parameters must fail export.
The writer must serialize, reload and compare extracted claims and structure
before emitting a bundle. Native formats may omit training-time parameters:
export must fail when the chosen claims cannot survive that round trip; the
manifest must never restore missing parameters into a reader's model.

### `features`

| Field | Type | Meaning / verification |
| --- | --- | --- |
| `names` | array of 1–100,000 unique nonempty strings, each at most 1,024 characters | Effective input names, in prediction order; length equals `n_features`. |
| `dtypes` | equally sized array of null or nonempty strings of at most 256 characters | Training DataFrame dtype declarations in feature order; null means unknown. They do not prove model input types. |
| `model_dtypes` | optional equally sized array of nonempty strings of at most 256 characters | Verified dtypes exposed by the native model, or null if it exposes none; separate from training dtypes. ONNX uses normalized NumPy dtype names such as `float32`, `float64`, `int32`, `int64`. |
| `origin` | `"model"`, `"dataframe"`, `"generated"` | Source of the effective names. |

If a model exposes feature names, origin must be `model` and names must equal
its extracted names in order. Otherwise `dataframe` names are author declarations
from the locally supplied frame; the reader verifies their count, not their
historical truth. With no model names or frame names, origin is `generated` and
names must be `feature_1` through `feature_N`. Claim extraction represents missing
model feature names as null, so falsely declaring a named model's origin as
`dataframe` or `generated` also fails comparison. Model dtype extraction is
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
metadata. Library/serialization/member pairing is validated by the closed
registry. Structure is compared separately using the same canonical encoding.
Schema, training dtypes, frame-origin names, target name, training environment,
user metadata, identity and samples are not verified against the model; they must
still pass their own validation and integrity checks. Inspection verifies format
and integrity only, and must never attest that model claims match the model.

## `structure.json`

Structure is never embedded in the manifest. It is a JSON object with required
`family` and `feature_importances` fields, plus exactly the family fields below.
`feature_importances` is null or an ordered numeric array of length `n_features`.
Numbers use the canonical encoding, including non-finite tags. Arrays and tree
lists preserve native order; no pruning, rounding, flattening of output/class
dimensions or display-oriented simplification is permitted.

| `family` | Additional required fields |
| --- | --- |
| `decision_tree` | `tree`: one tree object. |
| `tree_ensemble` | `estimator_shape`: original shape of the ensemble's estimator collection; `trees`: row-major flattened list of tree objects. Product of shape equals list length. A one-dimensional collection uses `[N]`. |
| `linear` | `coef`, `intercept`: native numeric arrays/scalars converted without reshaping. |
| `xgboost` | `trees`: ordered list of parsed JSON tree dumps, each from `get_dump(dump_format="json", with_stats=True)`. Retain all native fields, including node IDs, split/leaf values, missing-child IDs, gain/cover and children. Do not embed dump strings. |
| `feature_importances` | No additional fields; the importance array must be non-null. |

A scikit-learn tree object has exactly `node_count` (positive integer),
`n_features` (positive integer), `n_outputs` (positive integer), `n_classes`
(native per-output class-count array), and these native `tree_` arrays:
`children_left`, `children_right`, `feature`, `threshold`, `value`, `impurity`,
`n_node_samples`, `weighted_n_node_samples`. Except `value`, each array's length
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

## Adapter and result contracts

The internal `Adapter` protocol in `mlbundle.adapters` owns recognition,
serialization, loading, claim extraction, structure extraction and a `Predictor`.
The registry imports only the three constant mlbundle module paths `_sklearn`,
`_xgboost`, `_onnx`, and constructs their module's `Adapter` class. Library
imports must be lazy, with dependency failures naming the extra to install.
Missing adapter modules currently raise `AdapterNotImplementedError`. Public
`inspect` is implemented; there are no public export/load placeholders.

Predictors accept adapter-supported tabular objects without a core pandas/numpy
dependency. Regressors return `list[float]` for a single output or row-major
`list[list[float]]` for multiple outputs. Classifiers return row-major
`list[list[float]]` for one output or an output-major list of those matrices for
multiple outputs, aligned with class labels. Validate probability dimensions,
finite probabilities, range and row sums in the adapter. Classifier prediction
uses `predict_proba`; regressor prediction uses `predict`. Unsupported probability
or regression contracts fail rather than synthesizing outputs.

`MetadataView` carries `manifest` and a tuple of `MemberName` values including
the manifest; it has no model. `LoadedBundle` additionally carries the exact
native object, a uniform predictor, optional schema/structure/sample values, and
properties for kind, library, class name, ordered feature names and origin, and
output-major original class labels. Calling the prediction method for the wrong
kind raises a bounded `ValueError`. Result records and typed models prevent field
reassignment, but nested JSON values and the native model are not deeply frozen.
Construction itself performs no model validation; only a future validating reader
can supply the loaded-result guarantee. Consumers must not mutate a validated
model and continue treating its original metadata as verified.

## Versioning and integrity

Readers must reject unknown format versions rather than guess. Any change to
required fields, accepted identifiers, member rules, claim semantics, numeric
encoding or mandatory structure shapes requires a new format version and an
explicit reader implementation. Tightening caller-selected resource limits does
not change the format. Corrections to documentation that preserve wire behavior
and implementation changes behind these contracts do not require a version bump.

Hashes detect corruption, not deliberate tampering: an attacker can replace a
model and update every record. Claim recomputation prevents a bundle from showing
metadata/structure different from its prediction model; it does not establish
training history, model quality, fairness or author identity. Signing is out of
scope. See [SECURITY.md](../SECURITY.md) for the complete threat model.
