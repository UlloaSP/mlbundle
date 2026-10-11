# mlbundle

mlbundle is the training-side companion to MLSuite: a Python library for putting
a trained model, a rich manifest, a locally inferred schema and model structure
in one compressed `.mlbundle` file. It owns both writing and validated reading
so that the format is defined once.

Phases 1 and 2 implement the version 1 format contract, strict manifest models,
canonical JSON, integrity helpers, environment capture, adapter/result contracts,
secure container reading/writing and public metadata-only `inspect`.
No public export/load API, model adapter, schema inference or CLI exists yet.
Rules for later phases must not be described as implemented until code and tests
prove them. `docs/format.md` owns the wire contract.

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

Version 0.1 supports three libraries: scikit-learn via `skops.io.dump` and
`skops.io.load`, XGBoost via native `save_model("model.ubj")`, and the user's ONNX
file unchanged. Do not convert models. Adapters import their library lazily;
their dependencies belong in the optional extras `mlbundle[sklearn]`,
`mlbundle[xgboost]` and `mlbundle[onnx]`, declared in Phase 1.

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
- The skops trusted-types list lives in mlbundle. Never read it from a bundle.
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

- No source file may exceed 300 lines. Split by responsibility before the limit.
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
- sklearn-to-ONNX conversion, per-model environments, bundle signing and other
  libraries (including LightGBM and PyTorch) are outside the project scope.
- Never claim a verification that was not run; report its real output or blocker.

## Development

Work from the repository root using `uv`. Python 3.14 is the minimum because
mlschema 0.2.1 supports that version. Keep `uv_build` as the build backend.

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
on Ubuntu and Windows. The local build must produce a wheel and source archive.

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
paths atomically, requiring explicit overwrite. Future model export will infer
schema locally, extract claims/structure and serialize through native adapters.
The closed member list is: required `manifest.json`, exactly
one of `model.skops`, `model.ubj` or `model.onnx`, optional `schema.json`, and
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
match the model. Future full loading for MLSuite must use a closed adapter and
recompute claims before model use.

## Where code lives

- `src/mlbundle/__init__.py` — public contract models, enums, errors and package version.
- `src/mlbundle/_base.py` — strict frozen model configuration and string bounds.
- `src/mlbundle/contract.py` — closed identifiers and native member/format pairing.
- `src/mlbundle/limits.py` — shared archive ceilings and specific member caps.
- `src/mlbundle/canonical.py` — deterministic JSON encoding and bounded strict decoding.
- `src/mlbundle/claims.py` — shared extracted claims and exact claim/structure comparison.
- `src/mlbundle/manifest.py` — manifest sections, cross-field validation and JSON boundaries.
- `src/mlbundle/integrity.py` — member SHA-256/size records and verification.
- `src/mlbundle/environment.py` — training declarations and distribution metadata capture.
- `src/mlbundle/adapters.py` — adapter/predictor protocols and prediction output types.
- `src/mlbundle/registry.py` — closed lazy imports of future `_sklearn`, `_xgboost`, `_onnx` modules; each provides an `Adapter` class.
- `src/mlbundle/results.py` — metadata-only and loaded reader result types.
- `src/mlbundle/_container_rules.py` — shared source type, closed member sets/order and ceilings.
- `src/mlbundle/_zip_directory.py` — bounded ZIP/ZIP64 directory and extra-field validation.
- `src/mlbundle/_zip_members.py` — local headers/descriptors and bounded decompression with CRC.
- `src/mlbundle/container.py` — source bounds, validated in-memory reading and integrity checks.
- `src/mlbundle/writing.py` — integrity ownership, deterministic ZIP and atomic publication.
- `src/mlbundle/inspection.py` — public metadata-only inspection without adapter resolution.
- `docs/format.md` — version 1 wire specification; model adapters remain unimplemented.
- `tests/` — one file per feature, with success and error cases together.
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
