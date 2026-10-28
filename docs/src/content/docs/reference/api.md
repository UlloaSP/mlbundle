---
title: Python API
description: Signatures and contracts of the public mlbundle symbols exported from the package root.
---

The package root exports the supported API. Everything else, including adapters and container
internals, is private and can change without notice.

```python
from mlbundle import (
    FORMAT_NAME,  # "mlbundle"
    FORMAT_VERSION,  # "1"
    ArchiveLimits,
    EnvironmentLock,
    LoadedBundle,
    Manifest,
    MetadataView,
    OutputPrediction,
    check_environment,
    export,
    inspect,
    load,
)
```

Model records are strict, frozen Pydantic models: they reject unknown fields and type coercion.

## Functions

### `export`

```python
def export(
    model: object,
    destination: str | PathLike[str],
    *,
    training_data: object | None = None,
    feature_names: Sequence[str] | None = None,
    class_labels: Sequence[Sequence[Label]] | None = None,
    target_names: Sequence[str] | None = None,
    sample_rows: int | None = None,
    onehot_separator: str = "__",
    metadata: Mapping[str, JsonValue] | None = None,
    environment_lock: bytes | str | PathLike[str] | None = None,
    overwrite: bool = False,
    limits: ArchiveLimits | None = None,
) -> MetadataView
```

Publishes a bundle only after its claims survive a native serialization round trip and the reader
validates it. Requires the `schema` extra, and the `lock` extra unless `environment_lock` supplies
the runtime lock. `Label` is a JSON scalar class label. Every argument is
described in [Export a bundle](/guides/export/#arguments).

### `inspect`

```python
def inspect(source: BundleSource, *, limits: ArchiveLimits | None = None) -> MetadataView
```

Validates the container, manifest, JSON bounds, hashes, runtime lock, schema and samples without
importing a model library. It cannot attest model-derived claims.

### `load`

```python
def load(source: BundleSource, *, limits: ArchiveLimits | None = None) -> LoadedBundle
```

Everything `inspect` does, then deserializes through the closed adapter and verifies recomputed
claims and structure. `BundleSource` is `str | PathLike[str] | bytes | BinaryIO`.

### `check_environment`

```python
def check_environment(bundle: MetadataView | LoadedBundle) -> None
```

Raises `EnvironmentMismatchError` unless the running interpreter is CPython with the GIL at the
lock's Python version, on a locked target (Linux with glibc 2.28 or newer, or Windows), with exactly the target's
locked distributions plus mlbundle at `writer_version` installed. It reads distribution metadata
only. See [Inspect and load](/guides/inspect-and-load/#serve-in-the-locked-environment).

## Results

### `MetadataView`

Frozen dataclass returned by `export` and `inspect`.

| Field | Type |
| --- | --- |
| `manifest` | `Manifest` |
| `members` | `tuple[MemberName, ...]`, in canonical order |
| `environment_lock` | `EnvironmentLock` |
| `schema` | `dict[str, JsonValue] \| None` |
| `samples` | `list[dict[str, JsonValue]] \| None` |

### `LoadedBundle`

Frozen dataclass returned by `load`, constructed only after verification.

| Member | Type / behaviour |
| --- | --- |
| `manifest`, `members`, `environment_lock`, `schema`, `samples` | As in `MetadataView`. |
| `structure` | `dict[str, JsonValue] \| None` |
| `native_model` | The native model object. |
| `predictor` | The adapter's uniform predictor. |
| `kind` | `ModelKind` |
| `library` | `Library` |
| `class_name` | `str`, the estimator class |
| `feature_names` | `tuple[str, ...]` |
| `feature_origin` | `FeatureOrigin` |
| `class_labels` | `list[list[Label]] \| None`, output-major |
| `predict_proba(data)` | `list`, classifiers only; otherwise `ValueError` |
| `predict(data)` | `list[float]` or rows of floats, regressors only; otherwise `ValueError` |
| `row_from_record(record)` | `list[float]` in feature order |
| `predict_record(record)` | `tuple[OutputPrediction, ...]`, one per output |

### `EnvironmentLock`

Frozen record of the validated `pylock.toml`.

| Member | Type / behaviour |
| --- | --- |
| `python_version` | `str`, the exact CPython version |
| `packages` | `tuple[LockedPackage, ...]`, sorted by name; each has `name`, `version`, `platforms` and `wheels` |
| `platforms`, `glibc` | `("linux-x86_64", "linux-aarch64", "win-amd64")` and the Linux minimum `(2, 28)` |
| `environment_id` | `str`, SHA-256 of the Python version, packages, targets, wheel names and digests |
| `version(name)` | Locked version of a normalized distribution name, or `None` |
| `to_bytes()` | The canonical TOML file |
| `EnvironmentLock.from_bytes(data)` | Parse and validate canonical bytes |

`LockedWheel` has `name`, `url`, `size` and `sha256`. The rules are in
[the lock contract](/format/environment/).

### `OutputPrediction`

Frozen dataclass with `output`, `kind`, `labels`, `probabilities`, `value` and `predicted_label`.
See [Records and predictions](/guides/records/#outputprediction).

## Manifest records

| Class | Fields |
| --- | --- |
| `Manifest` | `format_name`, `format_version`, `bundle_id`, `created_at`, `writer_version`, `model`, `features`, `target`, `environment`, `integrity`, `references`, `user_metadata` |
| `ModelInfo` | `estimator_class`, `kind`, `hyperparameters`, `n_features`, `n_classes`, `n_outputs`, `library`, `serialization`, `pipeline` and state fields |
| `FeatureInfo` | `names`, `dtypes` (training), `model_dtypes` (verified), `origin` |
| `TargetInfo` | `class_labels`, `names` |
| `TrainingEnvironment` | `python_version`, `python_implementation`, `free_threaded`, `os`, `os_version`, `architecture`, `onnx` |
| `OnnxInfo`, `Opset` | ONNX opsets, IR version and producer |
| `IntegrityRecord` | `sha256`, `size` |
| `MemberReferences` | `schema`, `lock`, `structure`, `samples` |
| `ModelClaims` | The model-derived claims the reader recomputes |

Field types and verification rules are normative in [Format 1](/format/#manifest).

## Enums

All are `StrEnum`s whose values appear in the manifest.

| Enum | Values |
| --- | --- |
| `Library` | `sklearn`, `xgboost`, `catboost`, `onnx` |
| `ModelKind` | `classifier`, `regressor` |
| `SerializationFormat` | `skops`, `ubjson`, `cbm`, `onnx`, `pipeline` |
| `FeatureOrigin` | `model`, `dataframe`, `declared`, `generated` |
| `MemberName` | `manifest.json`, `model.skops`, `model.ubj`, `model.cbm`, `model.onnx`, `schema.json`, `structure.json`, `samples.json`, `pylock.toml` |

## Limits and errors

[`ArchiveLimits`](/reference/limits/) lowers the safety ceilings. Every exception mlbundle raises
for a bundle or model derives from `MLBundleError`; see [Errors](/reference/errors/).

`mlbundle.__version__` is the installed package version.
