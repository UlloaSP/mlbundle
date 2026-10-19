---
title: Errors
description: The MLBundleError hierarchy, when each error is raised, and how messages stay bounded.
---

Validation of bundles, models, schemas and prediction inputs fails with a bounded `MLBundleError`.
Messages never echo record values or native third-party exception text. They can name a member, a
claim, an escaped and truncated feature key, an extra or a writer/version hint. Filesystem failures
and API misuse remain builtin exceptions, listed at the end.

```python
from mlbundle import MLBundleError, load

try:
    bundle = load(b"not a bundle")
except MLBundleError as error:
    print(type(error).__name__)  # InvalidContainerError
```

## Hierarchy

```text
MLBundleError
├── InvalidContainerError
├── LimitExceededError
├── UnsupportedFormatVersionError
├── InvalidManifestError
├── InvalidJSONError
├── IntegrityMismatchError
├── ClaimMismatchError
├── UnsupportedModelError
├── InvalidModelError
│   ├── EnvironmentMismatchError
│   └── PredictionInputError   (also a ValueError)
├── InvalidSchemaError
├── InvalidSamplesError
├── MissingOptionalDependencyError
└── AdapterNotImplementedError
```

## When they are raised

| Error | Raised when | Attributes |
| --- | --- | --- |
| `InvalidContainerError` | The ZIP violates the container rules: unknown, duplicate or missing members, unsafe names, bad records or CRCs, a missing schema in an old bundle. | |
| `LimitExceededError` | An input exceeds an [archive or member limit](/reference/limits/). | |
| `UnsupportedFormatVersionError` | The manifest declares a format version other than `1`. | |
| `InvalidManifestError` | The manifest fails strict validation; may hint at a different writer version. | |
| `InvalidJSONError` | A JSON member is malformed or uses an unsupported value. | |
| `IntegrityMismatchError` | A member's SHA-256 or size disagrees with the manifest. | `member` |
| `ClaimMismatchError` | A recomputed model claim or the structure differs from the bundle. | `claim` |
| `UnsupportedModelError` | Export of a model outside the adapter boundary. | `untrusted_types` |
| `InvalidModelError` | Native model content is malformed or violates the adapter contract. | |
| `EnvironmentMismatchError` | A skops load failed and the bundle declares a different scikit-learn version. | |
| `PredictionInputError` | Invalid prediction input or record. | |
| `InvalidSchemaError` | Training inputs or `schema.json` violate the schema contract. | |
| `InvalidSamplesError` | Invalid `sample_rows`, cell conversion or sample keys. | |
| `MissingOptionalDependencyError` | A needed extra is not installed. | `extra` |
| `AdapterNotImplementedError` | A library has no implemented adapter. | `library` |

Identifiers in `member`, `claim`, `extra` and `library` are restricted to short ASCII names;
anything else is reported as `unrecognized`.

Not every failure is an `MLBundleError`: an existing destination without `overwrite=True` raises
`FileExistsError`, and calling `predict` on a classifier (or `predict_proba` on a regressor) raises
`ValueError`.

The [command line](/guides/cli/#errors-and-exit-codes) maps `MLBundleError` to exit code 1 and
I/O failures to exit code 2.
