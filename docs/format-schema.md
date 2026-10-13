# Format 1 schema v2 and samples

Normative section of [the format contract](format.md). Local inference uses
mlschema >=0.2.1,<0.3, tested with 0.2.1. Fields, source and separator are untrusted
declarations; integrity cannot authenticate the training data or its history.

## Export inputs and defaults

Every export writes `schema.json`. `export(training_data=frame)` accepts a
nonempty pandas DataFrame. `target_names` is an output-major sequence: one unique
nonempty bounded name per verified output, each at most 1,024 characters. A bare
string is rejected with a bounded error requiring a list. Without `target_names`,
all columns are features; otherwise every target column must exist and is excluded
from inference, dtype capture and samples. Without a frame they are declarations.
Index values never leave through inference/samples; the frame is not modified.

Training columns must be unique, nonempty strings of at most 1,024 characters,
with count equal to the native feature count. Native names must match columns
in order after target exclusion. `feature_names=` with a frame must equal its
feature columns in order; it cannot rename or reorder them. Errors raise
`InvalidSchemaError` before publication. Native names retain model origin;
otherwise a frame gives dataframe origin. Training dtypes are `str(dtype)` in
feature order, separate from verified `model_dtypes`.

Without `training_data`, inference uses a synthetic one-row pandas DataFrame of
float `1.0` values, one column per feature. Native or declared names are its
columns; generated origin uses columns `range(n_features)`. mlschema supplies
all field shapes, including numeric `step: 0.1` and one-hot grouping for named
binary columns. Synthetic dtypes are not training evidence: manifest training
dtypes remain null. Generated names/labels are `feature_0` through `feature_{N-1}`.

`export(..., onehot_separator="__")` accepts a strict nonempty string of at most
16 characters. Inference calls `mlschema.infer_schema(frame, onehot_separator=...)`
without builders, overrides or custom kinds. The separator is stored and read
from the schema; consumers must not assume the default.

Model-origin fields retain name mappings. Dataframe/declared inference preserves
named columns for grouping, then rewrites every `mappedTo`, including series
subfields and one-hot options, to its zero-based feature position. Generated
origin is inferred on positional columns directly. Labels remain feature names.
For models without native names, prediction accepts 2-D NumPy arrays or nested
numeric rows in manifest feature order. Adapter-supported sparse arrays and
DataFrames retain native restrictions; supplied names do not add name-based
selection or reordering. Positional schema mappings do not change predictor inputs.

Export requires `mlbundle[schema]` even without training data. This separate extra
installs mlschema and transitively pandas on Python 3.14; it is included in `all`,
not the core or model extras. Missing pandas/mlschema raises
`MissingOptionalDependencyError` naming `schema`, before publication.
Import, inspect and schema validation on load do not import these libraries.
Load needs only the matching model extra; native libraries can opportunistically
import an installed pandas. mlschema requires Python >=3.14,<3.15; mlbundle's own
Python range and sole core dependency, Pydantic, are unchanged.

## `schema.json`

The exact top-level object is:

```json
{
  "schema_version": "2",
  "source": "training_data",
  "onehot_separator": "__",
  "fields": [
    {"kind":"number","label":"age","mappedTo":"age","required":true,"step":1.0}
  ],
  "reports": [
    {"kind":"classifier","label":"Predicted class","mappedTo":0,
     "labels":["0","1"],"showClassProbabilities":true}
  ]
}
```

All five keys are required and unknown keys are forbidden. `source` is exactly
`training_data` or `default`; `onehot_separator` is bounded as above. Version `1`
and unknown versions raise bounded `InvalidSchemaError` requiring re-export.
A missing schema member raises bounded `InvalidContainerError` saying the bundle
predates the required schema and must be re-exported, through shared member rules.
Thus 0.1.0a1 bundles without schema or with schema v1 are incompatible with 0.1.0a2.
Format version remains `1`, whose pre-release wire contract is not frozen.

Arrays preserve order. `fields` has 1–100,000 entries and `reports` 1–1,024.
Typed records use strict types, finite numbers and no arbitrary extensibility.
The vocabulary supports default inference, not every mlschema override.

Ordinary fields have exactly `kind`, `label`, `mappedTo`, `required`, plus the
kind-specific keys below. Labels are nonempty strings of at most 100 characters;
`required` is boolean. For model origin, `mappedTo` must be a manifest feature
name (nonempty, at most 1,024 characters). For every other origin, it must be a
strict integer position in `[0, n_features)`, never a bool or string. The other
mapping type is rejected. Resolve to an index before coverage/order checks;
a top-level ordinary label must equal the feature name at that index.

| Kind | Additional required keys |
| --- | --- |
| `text`, `boolean`, `date` | None. |
| `number` | `step`: finite positive number; integers infer `1.0`, floats `0.1`. |
| `category` | `options`: 1–100,000 unique strings, retaining categorical order and unused categories. |
| `series` | `field1`, `field2`: scalar fields from the same feature column; no nested series/one-hot groups. Subfield labels describe pair axes; mappings resolve to the parent's index. |
| `onehot-category` | Special shape below. |

One-hot groups have exactly `kind`, `label`, `required`, `options`, with no parent
mapping. At least two binary named columns sharing a prefix before the first
stored separator form a group. There are 2–100,000 options, each exactly
`{"label":SUFFIX,"value":SUFFIX,"mappedTo":MAPPING}`. Label/value are nonempty
strings of at most 1,024 characters and must agree. Resolve the mapping by origin;
the manifest feature at that index must equal group label + stored separator +
option value. Other fields can contain the separator; readers cannot attest
whether training values were binary. Options preserve feature order; fields
preserve the order of their first mapped feature. Non-adjacent groups are supported.

Resolved field/option indices must cover every feature exactly once, with no
duplicates, missing features, declared target or extra columns. Groups can make
the field count smaller than the feature count. Readers check this without
mlschema; they cannot verify historical inference or the declared source.

There is exactly one report per model output, in output order. `mappedTo` is the
strict zero-based output integer, including `0` for a single output. There is no
`output` key. Labels are `Predicted class` / `Predicted value` for one output;
multiple outputs append a space and the position, such as `Predicted class 0`.
A regressor has exactly `kind`, `label`, `mappedTo`. A classifier also requires
`labels` and `showClassProbabilities: true`. `labels` is an array of bounded
strings of at most 1,024 characters (including empty original strings), obtained by Python `str()` on that output's original labels.
Their order matches prediction columns. If two original labels in one output
have the same string form, export fails with bounded `InvalidSchemaError`.
The same string can occur in different outputs. Manifest labels retain their
original JSON types and are verified against the native model.

Reports are recomputed through shared construction and compared by canonical
JSON bytes. Inspect checks against manifest claims; load also verifies those
claims against the native model. An internally consistent manifest/report lie
cannot conceal native disagreement. Label-free ONNX labels remain count-checked
declarations, as in its native contract.

## What leaves the training machine

Default mlschema inference carries feature names/labels, field kinds, nullability
(`required`), categorical values (including unused categories), one-hot names and
suffixes, numeric step determined by dtype, and pair-axis names/kinds/nullability.
Training dtype strings and reports carry their declarations and native/resolved
class labels. Inference emits **no numeric/date minima, maxima, means, counts,
example values, defaults or arbitrary text row values**. No schema overrides are
offered. These declarations can still reveal sensitive category or column names.
Native models, learned structure and user metadata can independently reveal data;
omitting samples is not a guarantee against model memorization or model metadata.

## `samples.json`

Samples are **real user data**, enabled only by an explicit `sample_rows=N` with
a frame. N must be a strict positive integer (not a bool), no larger than the
frame length. Exactly the first N feature rows are selected; no random sampling,
implicit clipping, target column or index is included. The option defaults to
null; zero is rejected to keep sample intent explicit.

The member is a canonical JSON array of row objects, for example
`[{"age":42,"active":true}]`. Each row must have exactly the declared feature
names. JSON keys are sorted canonically; consumers use manifest feature order.
An externally written empty array is valid, but public export selects N > 0.
Reader validation rejects any declared target name that is a schema/sample feature;
export rejects the same overlap before publication, including synthetic defaults.

Cell mappings: Python and NumPy boolean/integer/real/string scalars become JSON
scalars; float32 promotes exactly to binary64. Categorical cells use their scalar
values. `None`, pandas NA/NaT and scalar NaN become null. Date/datetime/Timestamp
and NumPy datetime64 become ISO 8601 strings; timestamps retain available
fractional precision and UTC offset, dates use `YYYY-MM-DD`. Infinities use
canonical `$float` tags. Lists/tuples become arrays; dictionaries require string
keys and recursively use the same rules. Byte strings, complex values, timedeltas,
arbitrary objects and unsupported NumPy values fail without stringification.

Schema and samples each have the existing 16 MiB cap and JSON depth 32, further
lowerable through `ArchiveLimits`. Sample rows are encoded incrementally within
the byte budget; no row is truncated. Six members permit a composite with
manifest, skops prefix, native leaf, structure, schema and explicit samples.
Composite schema, training dtypes and samples describe original inputs, never
the transformed boundary; target reports use the final adapter's verified labels.

## Error boundaries

Invalid schema shape, vocabulary, mappings, order, report kind or report labels
raise `InvalidSchemaError`. Invalid sample selection, cell conversion or row keys
raise `InvalidSamplesError`. These are bounded `MLBundleError` subclasses without
data values or third-party exception text. Native claim disagreement remains
`ClaimMismatchError`, including a manifest/report lie that agrees internally but
disagrees with the model. Malformed JSON and resource caps retain `InvalidJSONError`
and `LimitExceededError`. Reference/integrity or actual member-presence mismatches
retain `InvalidManifestError` or `InvalidContainerError` at the container boundary.
