# mlbundle

mlbundle is the training-side companion to MLSuite: a Python library for putting
a trained model, a rich manifest, a locally inferred schema and model structure
in one compressed `.mlbundle` file. It owns both writing and validated reading
so that the format is defined once.

The pre-release v0.1 library implements strict format 1 manifests, secure container
I/O, inspect/export/load, sklearn/XGBoost/CatBoost/ONNX adapters, local schema, training
dtypes, explicit samples and prediction-state verification. The CLI provides
metadata-only `inspect` and full `verify`. Format 1 is not yet frozen.
Implementation claims require code and test proof. `docs/format.md` owns the wire contract.

## What makes mlbundle special?

### 1. Reproducibility across the training/runtime boundary

MLSuite turns trained models into reviewable applications; it does not train
them. Its current model and training-DataFrame joblib uploads permit pickle code
execution, omit training versions, and send real data just to infer a schema.
mlbundle must replace that workflow with one self-describing file. Preserve the
identity and meaning of the model, schema, features, classes and outputs.

### 2. Metadata is a claim, even when it looks useful

The author of a bundle may be an attacker. Everything derived from a model,
including hyperparameters, feature/class/output counts, coefficients, feature
importances and tree structure, is a claim. Writer and reader must use the same
extraction code. The reader must recompute these claims from the loaded model
and reject mismatches. A reviewer must never see a tree that differs from the
model used for prediction.

### 3. Training data stays on the training machine

mlbundle depends on mlschema to call `mlschema.infer_schema` locally; mlschema
does not depend on or know mlbundle. The bundle carries the resulting fields
and report definitions. Do not include the training DataFrame. Include sample
rows only when explicitly requested: samples are real user data.

### 4. Native formats with a narrow support boundary

Version 0.1 supports four libraries: scikit-learn via `skops.io.dumps/loads`,
XGBoost via native in-memory UBJSON, CatBoost via CBM, and the user's ONNX
file unchanged. Do not convert models. Adapters import their library lazily;
their dependencies belong in the optional extras `mlbundle[sklearn]`,
`mlbundle[xgboost]`, `mlbundle[catboost]` and `mlbundle[onnx]`.

The design takes ideas from MLflow's versioned manifest and native model
storage, not its code. MLflow is Apache-2.0; mlbundle is MIT.

## A note from the maintainer

Pablo Ulloa Santín prefers ambitious ML workflows and simple systems. Find the
owner of a rule, implement it once, and make the behavior obvious. Measure
twice, cut once, and resist scope creep. The smallest complete change must
still preserve validation, security and reproducibility.

## A small glossary

- **bundle** — the zip container with the `.mlbundle` extension.
- **manifest** — the versioned JSON record of identity, model metadata,
  environment, member hashes and sizes, and user metadata.
- **member** — a named file inside the zip; only a closed set is permitted.
- **adapter** — mlbundle-owned code for one supported model library, including
  serialization, loading and shared extraction of model-derived claims.
- **claim** — information asserted by the bundle's author; model-derived claims
  require comparison with the loaded model using canonical JSON byte equality.
- **predictor** — the adapter's uniform probability/regression interface; the
  native model remains available separately for runtime explanations.
- **schema** — locally inferred mlschema fields plus report definitions.
- **structure** — model internals for visualisation, tied to the actual model.
- **trusted types** — the mlbundle-owned allowlist used for skops loading.
- **inspect** — metadata-only reading that never deserializes the model.
- **runtime** — the consumer that validates a bundle before using its model.

## The three ways to hurt yourself

1. **Trusting the manifest.** Checksums and plausible metadata do not make an
   attacker honest. Recompute model claims and reject disagreement.
2. **Opening a convenience loading path.** Pickle fallbacks, dynamic imports and
   author-supplied trusted types undo the security boundary. Reject unsupported
   models on the training machine instead.
3. **Defining the format twice.** Separate writer/reader extraction or schema
   rules drift. Share the owning implementation and change every affected path.

## Hit every affected layer

- **Format.** Manifest validation, allowed members, version handling and
  `docs/format.md` must agree. Reject unknown versions instead of guessing.
- **Writer.** Compute the schema locally, record the environment and fail early
  for unsupported model types. Samples require explicit user intent.
- **Reader.** Validate the archive and manifest, verify hashes, load only through
  closed adapters, and recompute model-derived claims.
- **Structure.** Tree arrays, ensemble tree lists, XGBoost tree dumps, linear
  coefficients and feature importances must describe the loaded model.
- **Packaging.** Keep model libraries optional, imports lazy and the base
  package usable without an adapter's dependencies.
- **Tests.** Cover successful round trips and every rejection path together in
  the feature's test file.

## Guardrails

### Reader security rules

- Never use pickle, joblib or cloudpickle, even as an option or fallback.
- Never import or instantiate from strings in the manifest. The class to
  instantiate must come from a closed list inside mlbundle.
- The skops trusted-types list (Tree, AdamOptimizer, SGDOptimizer) lives in mlbundle; never read it from a bundle. Optimizers admit ObjectNode/CachedNode only.
- Export must fail on the user's machine when the model contains custom types,
  listing unsupported types from `skops.io.get_untrusted_types`. Do not defer
  discovery to upload time or offer to trust those types.
- Read the zip in memory by allowed member names. Never extract it to disk.
  Reject unknown or duplicate members, paths with separators or `..`, symlinks
  and encrypted entries. Enforce limits on member count, compressed size,
  uncompressed size and compression ratio before allocating or loading content.
- Verify each member's SHA-256 against the manifest. Hashes detect corruption,
  not deliberate tampering; signing is out of scope.
- Validate the manifest strictly and reject an unknown format version.
- Provide a metadata-only `inspect` read that does not deserialize the model.
- Recompute all model-derived claims with the writer's extraction code and
  reject a bundle when they differ from the loaded model.

### Repository rules

- No source, test or documentation file may exceed 300 lines. Split by responsibility before the limit.
- Add no runtime dependency without a concrete reason. The sole core dependency
  is Pydantic v2 for strict manifest validation; model libraries belong in extras.
- Keep a feature's tests in one file covering its success case and every error
  case. Test observable behavior and supported contracts.
- Avoid Python 3.14-only syntax; widening the Python range should be a metadata
  change rather than a source rewrite.
- Preserve public contracts and change format versions deliberately. Leave no
  dead stubs, speculative flags or undocumented fallback behavior.
- Use Conventional Commits (`chore:`, `docs:`, `ci:`, `feat:`, `fix:`, `test:`).
- Never add `Co-Authored-By`, AI attribution or "Generated with" lines to commits
  or PR text.
- `C:\software\mlsuite` and `C:\software\mlschema` are read-only references.
  Never modify, commit or run write commands inside either checkout.
- Do not publish to PyPI, create releases or version tags, or add a publish job.
  "Tags" in this project means GitHub repository topics.
- sklearn-to-ONNX conversion, per-model environments, bundle signing and additional
  libraries (including LightGBM and PyTorch) are outside the project scope.
- Never claim a verification that was not run; report its real output or blocker.

## Development

Work from the repository root using `uv` on Python 3.14. The optional `schema`
extra admits mlschema >=0.2.1,<0.3 on 3.14 only. Keep `uv_build` as the build backend.

```shell
uv python install 3.14
uv sync --locked --all-extras
uv run pre-commit install
```

After changing dependency metadata, use `uv sync` to regenerate `uv.lock` and
commit the lockfile. CI uses `uv sync --locked --group dev --all-extras` on Python 3.14.
The default dev group installs `mlbundle[all]`; base users can omit the dev group.
Ruff handles lint and formatting; Pyright uses the family's basic mode.

## Verifying

Run these commands from the repository root:

```shell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

The Ruff commands also work without the final `.` because the default is the
current directory: `uv run ruff check` and `uv run ruff format --check`.
`uv run pytest` runs `tests/`, measures branch coverage and enforces the family's
80% minimum, writing terminal and XML reports. Tests use the installed package;
its version comes from package metadata. CI checks lint and types and runs tests
and a clean base-wheel smoke check on Ubuntu and Windows. The local build must produce a wheel and source archive.

Start with the smallest proof of a behavior change, then broaden when needed.
If a check cannot run, report the exact command and blocker. Never claim a test,
environment or behavior was verified when it was not.

## Documentation

- `README.md` owns purpose, honest implementation status and development setup.
- `SECURITY.md` owns the threat model, guarantees, limits and private reporting.
- `docs/format.md` owns the version 1 wire specification and verification rules.
- Rewrite stale guidance instead of appending competing accounts.

Keep agent policy in this root file. Do not add nested `AGENTS.md` files.

## Plans and work artifacts

Do not create plan, notes, TODO, lessons, scratch or tech-debt files in the repo.
The conversation, issues, code, tests and final diff are the work record.

## How it works

The container writer accepts serialized member bytes and a manifest or its
JSON-ready parts, computes integrity records and produces a deterministic ZIP.
It enforces the same limits and validation as the container reader and publishes
paths atomically on hard-link filesystems, requiring explicit overwrite. The
exclusive-creation fallback removes partial destinations on failure but exposes
the file during copying. Public sklearn/XGBoost/CatBoost export extracts claims/structure,
serializes with skops/native UBJSON/CBM and verifies a native round trip
before publication. XGBoost supports exact XGBClassifier/XGBRegressor tree
wrappers with closed probability/regression objectives and one target. Original
class labels supplied through output-major `class_labels=` are bound to a native
copy through an mlbundle-owned booster attribute and verified on load. Omitted
labels use encoded indices; sklearn/CatBoost reject this option because they own classes_. Local `training_data=` inference uses mlschema with `__`, excludes a named target,
and records training dtypes. Only explicit `sample_rows=N` includes real rows.
Schema/report and sample validation is shared by export, inspect and load.
ONNX preserves path/bytes unchanged (ModelProto is serialized once), validates
bytes once per export/load with bounded scalar scans, and rejects external tensors,
custom domains, functions and training/runtime configuration before a CPU session. It verifies graph signatures, metadata and environment
claims. `native_model` is an OnnxModel with original bytes and a CPU session.
Kind is inferred from supported outputs; native labels are verified, absent
labels use count-checked declarations. Structure is null. See the normative
ONNX section linked from `docs/format.md` for exact layouts and resource bounds.
CatBoost supports exact numeric single-target symmetric classifiers/regressors, with native state commitments and in-memory loading.
The closed member list is: required `manifest.json`, exactly
one of `model.skops`, `model.ubj`, `model.cbm` or `model.onnx`, optional `schema.json`, and
`structure.json` when the supported model has extractable structure. Optional
`samples.json` requires explicit user intent. No other members are allowed.

The manifest must record format name/version, bundle id, UTC creation time and
writer version; model library, estimator class, classifier/regressor kind,
hyperparameters and feature/class/output counts; feature names, order, dtypes
and origin; original class labels and target name; training Python, OS,
architecture and package versions; member SHA-256 hashes and sizes; and
free-form user metadata. Sample rows are allowed only on explicit request.
All of this is untrusted input, and model-derived fields must be verified.
Class counts and original labels are output-major arrays, including for a single
output. Model-exposed dtypes are verified separately from declared training dtypes.
The writer must test serialization/reloading and reject claims or structure that
cannot survive the native round trip. Default safety ceilings may only be lowered.

The container reader validates the archive, manifest, JSON bounds and integrity
without loading a model. `inspect` exposes validated metadata and a member list
without importing model libraries; it cannot attest that model-derived claims
match the model. Public `load` uses a closed adapter and recomputes claims and
structure before returning the native model and validated predictor.

An adapter module provides `Adapter` with library/member identity, recognition, export preparation, serialization/deserialization, `ModelClaims`, structure extraction
and a `Predictor`. Extraction uses only the native model, shared by export/load;
ONNX graph fields belong in `ModelClaims.onnx`. Missing names are null claims;
the shared pipeline owns supplied/generated names and environment declarations.
Native errors must become bounded `MLBundleError` subclasses. `prepare_export`
receives the strict `ExportDeclarations` label record and leaves the user's
object unchanged. Classifier claims may use null labels only for a native model
without labels; `declarations.py` then resolves count-validated supplied/generated
labels. ONNX can use that declaration path without changing its original bytes;
its kind/count inference remains ONNX-owned. Keep native behavior in adapters; follow `adapters.py`.

## Where code lives

- `src/mlbundle/__init__.py` — public inspect/export/load, contract models, enums, errors and package version.
- `src/mlbundle/cli.py` — argparse inspect/verify console entry with safe bounded display.
- `src/mlbundle/_base.py` and `errors.py` — strict frozen model configuration, string bounds and bounded errors.
- `src/mlbundle/contract.py` — closed identifiers and native member/format pairing.
- `src/mlbundle/limits.py` — shared archive ceilings and specific member caps.
- `src/mlbundle/canonical.py` and `_canonical_work.py` — deterministic JSON and bounded decoding/expansion.
- `src/mlbundle/model_io.py` — library-independent public export/load, manifest construction and round-trip verification.
- `src/mlbundle/training.py` — lazy local DataFrame alignment and schema inference.
- `src/mlbundle/schema.py` and `_schema_fields.py` — closed schema/report declarations and reader cross-checks without mlschema.
- `src/mlbundle/samples.py` — explicit feature-row selection and bounded cell conversion.
- `src/mlbundle/_sklearn/` — skops trust/loading (including MLP adam/sgd/lbfgs state), parameter and claim extraction, native structure, nested payload bounds and validated prediction.
- `src/mlbundle/_xgboost/` — closed wrappers/objectives, native UBJSON preflight/loading, label binding, recoverable config, complete tree structure and validated prediction.
- `src/mlbundle/_catboost/` — closed numeric symmetric wrappers/losses, CBM preflight/in-memory loading, complete native state/structure and validated prediction.
- `src/mlbundle/_onnx/` — bounded protobuf/graph validation, unchanged bytes, native signatures/labels, CPU sessions and tabular prediction.
- `src/mlbundle/declarations.py` — export-time class labels and resolution for native models without labels.
- `src/mlbundle/claims.py` — shared extracted claims and exact claim/structure comparison.
- `src/mlbundle/manifest.py` — manifest sections, cross-field validation and JSON boundaries.
- `src/mlbundle/integrity.py` — member SHA-256/size records and verification.
- `src/mlbundle/environment.py` — training declarations and distribution metadata capture.
- `src/mlbundle/adapters.py` and `_prediction_input.py` — adapter/predictor protocols, output types and shared input checks.
- `src/mlbundle/registry.py` — closed lazy resolution of `_sklearn`, `_xgboost`, `_catboost` and `_onnx` adapters.
- `src/mlbundle/results.py` — metadata-only and loaded reader result types.
- `src/mlbundle/_container_rules.py` — shared source type, closed member sets/order and ceilings.
- `src/mlbundle/_zip_directory.py` — bounded ZIP/ZIP64 directory and extra-field validation.
- `src/mlbundle/_zip_members.py` — local headers/descriptors and bounded decompression with CRC.
- `src/mlbundle/container.py` — source bounds, validated in-memory reading and integrity checks.
- `src/mlbundle/writing.py` — integrity ownership, deterministic ZIP and atomic publication.
- `src/mlbundle/inspection.py` — public metadata-only inspection without adapter resolution.
- `docs/format.md` — version 1 wire specification; linked container, adapter/structure, ONNX and schema/sample sections are normative.
- `tests/` — one file per feature, with success and error cases together.
- `scripts/package_smoke.py` — fixture, isolated install and artifact-content checks.
- `pyproject.toml` and `uv.lock` — package metadata, tool configuration and lock.
- `.github/workflows/ci.yml` — lint, types and cross-platform tests.
- `.pre-commit-config.yaml` — Ruff, Pyright and basic file hygiene.
- `README.md`, `SECURITY.md` and this file — project and contributor context.

## Taste

- One format definition beats synchronized copies.
- A closed adapter boundary beats dynamic loading.
- Early export errors beat surprises after upload.
- Honest metadata beats impressive but unverifiable visualisations.
- Simplicity never excuses weak validation or data safety.
