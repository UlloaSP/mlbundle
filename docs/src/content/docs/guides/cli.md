---
title: Command line
description: The mlbundle inspect and verify commands, their output, JSON mode and exit codes.
---

Installing mlbundle adds an `mlbundle` console script with two commands. Both use the same readers
as the Python API.

```bash
mlbundle inspect trained.mlbundle
mlbundle inspect trained.mlbundle --json
mlbundle verify trained.mlbundle
```

## `inspect`

Validates metadata and integrity without importing any model library, so it works with the base
install. The first line says what was and was not checked:

```text
Metadata and integrity validated; model claims NOT verified.
Format: mlbundle 1
Bundle id: aa6f6b39-574a-499b-a0ff-de4cdd334050
Created: 2026-10-19T11:17:51.949658+00:00
Writer: "0.2.0"
Model: sklearn "DecisionTreeClassifier" (classifier)
Counts: features=2, outputs=1, classes=[2]
Features (declared): ["temperature", "humidity"]
Class labels (output-major): [["cold", "warm"]]
Target names (output-major): null
Members:
  manifest.json: validated (no self-integrity record)
  model.skops: 13065 bytes sha256=24d7cbab1d775ab279dc8ebc3c762b551b397ebf9b4917ac3c4474ff611b8813
  schema.json: 361 bytes sha256=ea62a6a7502ae8ea30b45b61bb18d8e31bd2ec55f026d9fbf0c05ef42d01ff25
  structure.json: 526 bytes sha256=66204da26781a4e2a800a1f3ed246ab1a43e8c97ced73d06a2640f897eeda03c
Training: {"python": "3.14.7", "implementation": "CPython", "os": "Windows", "os_version": "11", "architecture": "AMD64"}
Packages: {"catboost": "1.2.10", "mlbundle": "0.2.0", "mlschema": "0.3.0", ...}
ONNX environment: null
Present: schema=yes, structure=yes, samples=no
User metadata: {}
```

Composite pipelines add their step names, final estimator and transformed-feature boundary.
Values in the human summary escape terminal control characters and are truncated at 512 characters,
so a hostile bundle cannot rewrite your terminal.

`--json` prints the full validated manifest instead: lossless, ASCII-escaped JSON, suitable for
`jq` or archiving.

## `verify`

Loads the bundle through its adapter and recomputes every model-derived claim and the structure. It
needs the model's extra:

```text
Valid bundle: sklearn "DecisionTreeClassifier"; verified format, integrity, schema/samples and native model claims/structure.
```

## Errors and exit codes

Errors are a single bounded line on stderr naming the failure category. They never print bundle
content or a traceback.

| Exit code | Meaning | Example stderr |
| --- | --- | --- |
| `0` | Success. | |
| `1` | The bundle was rejected, or a dependency is missing. | `mlbundle: InvalidContainerError: bundle rejected.` |
| `1` | | `mlbundle: missing optional dependency; install mlbundle[xgboost].` |
| `2` | Usage or I/O failure. | `mlbundle: FileNotFoundError: I/O failed.` |
| `2` | | `mlbundle: invalid arguments; use --help.` |

Use `verify` in CI or before publishing a bundle elsewhere; use `inspect` where model libraries are
not installed or should not run.
