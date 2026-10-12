# Version 1 schema and samples

Normative section of [the format contract](format.md). Local inference uses
the mlschema 0.2 vocabulary, tested with 0.2.1. Schema and samples come from data the
reader never sees. Integrity does not authenticate that data or its history.

## Export inputs

`export(training_data=frame)` accepts a nonempty pandas DataFrame. Without
`target_name`, all columns are features. With `target_name`, that column must
exist and is excluded before inference, dtype capture and sample selection.
Without a frame, `target_name` remains a metadata declaration as before.
Index values are never included. The input frame is not modified.

Column names must be unique, nonempty strings of at most 1,024 characters.
Their count must equal the native feature count. When native names exist,
columns must match those names **in order**, after target exclusion. Supplying
`feature_names` with a frame requires exactly those feature columns in order;
it cannot rename or reorder them. Frame errors raise `InvalidSchemaError` before
publication. Names remain model-origin when the model exposes them; otherwise
the frame yields dataframe-origin names. Training dtypes are `str(dtype)` in
feature order, separate from verified `model_dtypes`.

Export calls exactly `mlschema.infer_schema(feature_frame, onehot_separator="__")`,
without builders, overrides or custom kinds. It uses the same separator as
MLSuite's `schema_service.py`. Unlike MLSuite's older positional fallback, named
frame columns are preserved even when a model has no names; inferred mappings
therefore always refer to manifest names, never integer positions. Fields retain
the direct mlschema output. MLSuite's MLForm integration still converts a `series`
field's `field1`/`field2` into `columns` with those IDs and removes nested
`mappedTo`; this is presentation mapping and needs no DataFrame. Consumers may
format typed class labels for display and select reports by output index.

`mlbundle[schema]` optionally installs mlschema >=0.2.1,<0.3 from PyPI, transitively
pandas; it is included in `all`. The dependency is marked for Python 3.14 because
that mlschema version requires `>=3.14,<3.15`. mlbundle's own `>=3.14` range and
sole core dependency (Pydantic) are unchanged. Training export with missing
pandas/mlschema raises `MissingOptionalDependencyError` naming `schema`.
mlbundle import and inspection do not import either library. Loading does not
require them, although native sklearn imports an installed pandas opportunistically.

## `schema.json`

The exact top-level object is:

```json
{
  "schema_version": "1",
  "fields": [
    {"kind":"number","label":"age","mappedTo":"age","required":true,"step":1.0}
  ],
  "reports": [
    {"kind":"classifier","label":"Predicted class","mappedTo":"classifier",
     "output":0,"labels":["cold","warm"],"showClassProbabilities":true}
  ]
}
```

All keys shown are required. Unknown keys and versions are rejected. Arrays are
ordered; `fields` has 1–100,000 entries and `reports` 1–1,024. Typed records use
strict types, finite numbers and no arbitrary extensibility. This envelope
supports only the default inferred vocabulary, not every mlschema override.

Every ordinary field has exactly `kind`, `label`, `mappedTo`, `required`, plus
the kind-specific keys below. Labels are nonempty strings of at most 100
characters; `mappedTo` is a manifest feature name; `required` is a boolean.
At the top level, an ordinary field's label equals its feature name.

| Kind | Additional required keys |
| --- | --- |
| `text`, `boolean`, `date` | None. |
| `number` | `step`: finite positive number; default inference emits `1.0` for integers or `0.1` for floats. |
| `category` | `options`: 1–100,000 unique strings, retaining categorical order and unused categories. |
| `series` | `field1`, `field2`: ordinary scalar fields from the same feature column; no nested series or one-hot groups. Subfield labels describe pair axes; each `mappedTo` equals its parent mapping. |
| `onehot-category` | Special shape below. |

A one-hot group has exactly `kind`, `label`, `required`, `options`; no parent
`mappedTo`. Its label is the shared prefix before the first `__`. At least two
binary columns with that prefix form a group. There are 2–100,000 options, each
exactly `{"label":SUFFIX,"value":SUFFIX,"mappedTo":COLUMN}`. These three strings
are nonempty and at most 1,024 characters. `COLUMN = PREFIX + "__" + SUFFIX`;
the option label and value agree. Other columns stay ordinary fields, even if
their names contain `__`; the reader cannot verify whether their values were
binary. Group options preserve feature order, and fields are emitted in order
of their first mapped feature. Non-adjacent group columns are supported.

Field/option mappings must cover every manifest feature exactly once, with no
duplicates, missing features, declared target or additional columns. The field count may be
smaller than the feature count because of one-hot groups. The reader verifies
these mappings and ordering without mlschema, not historical inference outcomes.

There is exactly one report per model output, in output order. `output` is a
zero-based integer index; every report has the model's kind and `mappedTo` equal
to that kind. A classifier report has exactly the classifier keys in the example:
its label is `Predicted class`, `showClassProbabilities` is exactly true, and
`labels` equals that output's manifest class-label array by canonical byte equality.
Types and prediction-column order are preserved. A regressor report has exactly
`kind`, `label`, `mappedTo`, `output`, with label `Predicted value` and no labels
or probability setting. This extends MLSuite's single-output report shape with
an output index and typed labels instead of display-string coercion.

Inspection validates reports against the manifest; it cannot verify that manifest
against the model. Load also verifies native claims, so a changed manifest/report
pair cannot conceal disagreement with native kind, counts or labels. Label-free
ONNX mappings remain count-checked declarations, as in the ONNX contract.

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
Reader validation also rejects a declared target name that is a sample feature.

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
