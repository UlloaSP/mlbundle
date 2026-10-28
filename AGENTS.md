# MLbundle

mlbundle is a Python library for packaging a trained model, a manifest, a locally
inferred schema and model structure in one compressed `.mlbundle` file.

It owns export, metadata-only inspection and validated loading. It is the
training-side companion to MLSuite, whose runtime accepts `.mlbundle` uploads.
MLSuite does not train models. The package version is 0.3.0; format 1 is not frozen.
Implementation claims require code and test proof.

## What must remain true

### 1. The format is one contract

`docs/format.md` owns the wire contract and links its normative sections.
Writer, reader, adapters, schemas and visualizations must agree with it.
Reject unknown format versions; change public contracts and versions deliberately.

### 2. Metadata is a claim

The author of a bundle may be an attacker. Model parameters, feature/class/output
counts, coefficients, importances and tree structure are claims, even when useful.
Writer and reader use the same extraction code. Load recomputes model-derived
claims and structure and compares canonical JSON bytes, rejecting disagreement.
A reviewer must see the model used for prediction.

### 3. Training data stays local

mlbundle calls `mlschema.infer_schema` locally. mlschema does not depend on or know
mlbundle. Carry its fields and report definitions, never the training DataFrame.
Include real rows only through explicit `sample_rows=N`; samples are user data.
Names, categories, learned models and structure can still reveal training information.

### 4. Native models cross a closed boundary

Support scikit-learn through skops, XGBoost through in-memory UBJSON, CatBoost
through CBM and existing ONNX bytes unchanged. Do not convert models.
Adapters import their libraries lazily; model dependencies belong in optional
`sklearn`, `xgboost`, `catboost` and `onnx` extras. Every bundle's `pylock.toml`
locks those runtimes for Linux and Windows through uv (`lock` extra); it is a claim that
mlbundle validates and never installs. Runtimes own installation policy.

### 5. Inspection and verification mean different things

`inspect` validates container, manifest, integrity and schema/sample declarations
without importing model libraries or deserializing models. It cannot attest
model-derived claims. `load` additionally verifies native claims and structure.
The CLI exposes these as `inspect` and `verify`. Keep this distinction visible
in APIs, docs and the playground.

## Maintainer intent

Pablo Ulloa Santín favors ambitious ML workflows built from simple systems.
Find the owner of a rule, implement it once and make its behavior obvious.
The smallest complete change preserves validation, security and reproducibility
through every affected path.

Remove complexity whose constraint no longer exists. Keep compatibility only
when required by the public contract or the task. Avoid speculative flags,
dead stubs and undocumented fallback behavior.

## Vocabulary

- **bundle**: the ZIP container with the `.mlbundle` extension
- **manifest**: versioned identity, model metadata, environment, integrity records and user metadata
- **member**: a named file from the container's closed allowed set
- **adapter**: owned recognition, native I/O, shared extraction and prediction for one library
- **claim**: information asserted by the author; model-derived claims require recomputation
- **schema**: locally inferred mlschema fields and report definitions
- **structure**: model internals for visualization, bound to the loaded model
- **predictor**: uniform probability/regression interface, separate from the native model
- **trusted types**: mlbundle-owned allowlists for skops loading
- **runtime**: the consumer that validates a bundle before using its model

## Invariants

- Source, test and documentation files stay at or below 300 lines. Split by responsibility.
- Pydantic v2 is the sole core runtime dependency. New dependencies need a concrete reason; model libraries remain extras.
- Keep `uv_build` as the Python build backend.
- Avoid Python 3.14-only syntax so widening support can remain a metadata change.
- Preserve model, feature, class, output and schema identity across export and load.
- Safety ceilings may only be lowered by callers.
- Tests for a feature keep successful round trips and rejection cases together in one file.
- Verification claims describe commands actually run and their real results.
- `C:\software\mlsuite`, `C:\software\mlschema`, `C:\software\mlform` and `C:\software\eshkra` are read-only references. Never modify, commit or run write commands in them.
- sklearn-to-ONNX conversion, provisioning per-model environments (bundles only describe them), bundle signing and additional model libraries (including LightGBM and PyTorch) are outside scope.

## Reader security

These rules apply to every loading and inspection path, including examples and UI:

- Never use pickle, joblib or cloudpickle, even as an option or fallback.
- Instantiate only from mlbundle's closed class lists. Never import or instantiate from manifest strings.
- Own skops trusted lists in mlbundle; never accept them from a bundle or add XGBoost/CatBoost types. Trees, optimizers, scalar dtypes and closed functions admit only their owning loaders.
- Reject custom types at export on the training machine. List unsupported types from `skops.io.get_untrusted_types`; never offer to trust them or defer rejection to upload.
- Read ZIP members in memory by allowed names. Never extract the archive to disk.
- Reject unknown/duplicate members, separators or `..` in names, symlinks and encrypted entries.
- Enforce member count, compressed size, uncompressed size and compression ratio before allocation or loading.
- Validate the manifest strictly and verify every non-manifest member's SHA-256 and size. Hashes detect corruption, not authenticity or deliberate tampering.
- Recompute model-derived claims and structure using the writer's extraction code before returning a loaded model.
- Keep errors bounded. Record-input errors name escaped keys without echoing values.

Read `SECURITY.md` before changing trust boundaries. Read `docs/format-container.md`
before changing archive parsing or limits, and `docs/format-state.md` before
changing skops trust, fitted state or supported types.

## How MLbundle fits together

1. A closed adapter recognizes the model and prepares it without mutating the user's object.
2. Shared extraction derives model claims and structure from native state.
3. Native serialization and reload prove those claims before export publishes the bundle.
4. Local mlschema inference supplies fields/reports; shared code resolves features, targets, environment and the runtime lock.
5. The writer computes integrity records, validates through the reader and writes a deterministic ZIP.
6. Inspect validates bounded container/JSON/lock data and declarations without adapter resolution.
7. Load uses the closed adapter and compares recomputed claims and structure.
8. The predictor and record APIs expose validated inputs and output-major report results.

The allowed members are manifest, required schema, one native member (or a declared
skops + UBJSON/CBM pair), extractable structure, explicit samples and the required
lock: at most seven.
Publication uses atomic hard links or replacement and requires explicit overwrite.
The exclusive-create fallback removes partial destinations on failure but exposes
the file during copying. Preserve that behavior and document its limits honestly.

## Ownership and contract references

Use the entry points and linked contracts to find the owning implementation:

- **Public I/O:** `model_io.py`, `inspection.py`, `results.py` and `cli.py`; `__init__.py` owns public exports.
- **Container:** `container.py`, `writing.py`, `_container_rules.py`, `_zip_directory.py`, `_zip_members.py`, `limits.py` and `integrity.py`.
- **Declarations:** `manifest.py`, `canonical.py`, `declarations.py`, `environment.py`, `training.py`, `schema.py` and `samples.py`.
- **Runtime lock:** `lock.py`, `_lock_toml.py`, `_wheel_tags.py`, `_lock_generation.py` and `_environment_check.py`; read `docs/format-environment.md` before changing them.
- **Adapters:** `adapters.py` and `registry.py` define closed lazy resolution; `_sklearn/`, `_xgboost/`, `_catboost/`, `_onnx/` and `_composite/` own native behavior.
- **Claims and predictions:** `claims.py`, `records.py`, `_prediction_input.py`, `_prediction_decisions.py` and `_display.py`.
- **Websites:** `landing/`, `docs/` and `playground/`; `brand/` owns the shared identity sourced from eshkra.

Paths above are relative to `src/mlbundle/` unless they name a root directory.
Use package configuration and source for detailed inventories and command options.

When changing schema, mappings or samples, read `docs/format-schema.md`:

- Every export needs the schema extra, including default inference from one float-1.0 row.
- Schema v2 records source and separator (default `__`). Real frames exclude every declared target and record training dtypes; default dtypes are null.
- Native model names map by name; other origins map by position, with generated names starting at `feature_0`.
- Unique target names map by output position. Reports use string labels; manifest labels retain their types.
- MLschema owns series columns and explicit conversion to positional mappings. Export/inspect/load share declaration validation without reader-side mlschema.

When changing native I/O or extraction, read `docs/format-adapters.md` and the
adapter-specific contracts it links:

- `prepare_export` accepts strict `ExportDeclarations` and leaves the original model unchanged. Missing native names are null; shared code owns declared/generated names and environment.
- Null native classifier labels permit count-checked declarations. Sklearn/CatBoost own `classes_` and reject `class_labels=`; XGBoost binds supplied output-major labels to a native copy and otherwise uses encoded indices.
- XGBoost supports exact single-target tree wrappers with closed objectives. CatBoost supports exact numeric single-target symmetric models with complete native state/structure and in-memory CBM loading.
- Read `docs/format-pipeline.md` for composites. Exact outer sklearn Pipelines ending in XGBoost/CatBoost use checked skops/native components; other native placements remain rejected. Composite trees address transformed inputs; all-sklearn Pipeline structure stays null.
- Read `docs/format-onnx.md` before changing ONNX. Preserve path/bytes unchanged, serialize ModelProto once and validate once per export/load with bounded scalar scans before CPU sessions. Reject external tensors, custom domains/functions and runtime/training configuration. Verify signatures, metadata/environment claims and native labels; structure stays null.
- Native errors become bounded `MLBundleError`s. Version hints diagnose failed manifests/skops loads without adding compatibility gates.

When changing record prediction, preserve shared schema-key lookup, bounded
numeric-string/bool conversion and schema-permitted None-to-NaN handling.
Never iterate extra keys. Frozen output-major results use the native class
decision's report label, never one inferred from probabilities.

## Check every affected path

Apply the entries relevant to the change:

- **Format:** versions, manifest validation, allowed members and normative docs agree.
- **Writer:** unsupported models fail early; local inference, environment and explicit samples remain correct.
- **Reader:** archive bounds, hashes, closed loading and recomputed claims remain enforced.
- **Structure:** visualized trees, coefficients and importances describe the loaded model.
- **Schema:** fields, reports, names, dtypes, mappings and samples agree across export/inspect/load.
- **Packaging:** base installs work without adapter libraries; imports remain lazy.
- **Websites:** examples use actual APIs, inspection claims are accurate, controls work by keyboard and layouts remain usable on mobile.
- **Tests:** success and affected rejection paths exercise observable contracts.

A change is incomplete when it works only through the path used by its first test.

## Development and testing

Work from the repository root with `uv` on Python 3.14:

```shell
uv python install 3.14
uv sync --locked --all-extras
uv run pre-commit install
```

The schema extra admits mlschema `>=0.3.0,<0.4` on Python 3.14 only. Default dev
installs include `mlbundle[all]`; base users can omit them. Dependency metadata
changes require `uv sync` to regenerate `uv.lock` and inclusion of the updated
lockfile in the change. CI and publishing use locked dev/all extras on 3.14.

Start with the smallest proof of observable behavior. Cover success and affected
errors together; use regression tests for bugs and integration tests for shared
boundaries. Use browser checks when rendering, focus or keyboard behavior matters.

Before handing off Python source or package changes, run:

```shell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

Ruff owns lint/formatting; Pyright uses basic mode. Tests use the installed package,
obtain its version from metadata and enforce 80% branch-inclusive coverage with
terminal/XML reports. Build both wheel and sdist. CI tests the built wheel on
Ubuntu/Windows, including base installs.

Use Vite+ (`vp`) for all three frontend projects: `vp install`, `vp run check`
and `vp run build` at the root, or the site's `package.json` scripts via `vp run`.
Astro commands go through `vp exec`; the playground uses Vite+ directly.
A docs-only edit needs relevant examples/links/build checks, not every
Python gate. If any check cannot run, report its exact command and blocker.

## Documentation

Keep README focused on identity, installation, primary usage, supported surfaces
and links to detailed guides. `SECURITY.md` owns the threat model, guarantees,
limits and private reporting. `docs/format.md` owns the wire specification;
web docs must reuse the normative files instead of maintaining parallel copies.

Document user-visible contracts and durable cross-module decisions. Rewrite
stale guidance rather than appending a competing account. Keep all agent policy
in this root file; do not add nested AGENTS.md files.

## Git, publishing and work artifacts

Preserve unrelated working-tree changes. Do not commit, push, create branches
or open pull requests unless explicitly requested. Use Conventional Commits
(`chore:`, `docs:`, `ci:`, `feat:`, `fix:`, `test:`).
Never add `Co-Authored-By`, AI attribution or "Generated with" lines to commits or PRs.

PyPI publication is automatic when `project.version` changes in a push to `main`.
`publish.yml` must run the full CI workflow first and publish only that new version
through the `pypi_env` environment, whose branch policy is limited to `main` and
whose `MLBUNDLE_API_TOKEN` secret is project-scoped. Never read or expose secret
values, change versions just to trigger publication, or invoke uploads manually.
No GitHub releases or version tags; "tags" means repository topics.

Do not create plan, notes, TODO, lessons, scratch or tech-debt files in the repo.
Keep generated coverage and browser evidence out of changes. The conversation,
issues, source, tests and final diff are the work record.

## Done

Work is complete when every applicable path works, the owning layer contains
the rule, contracts remain intentional, obsolete behavior is gone, useful
regression coverage exists, relevant checks have run and documentation agrees.
Simplicity never excuses weaker validation or data safety.
