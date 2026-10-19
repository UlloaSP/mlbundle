# MLbundle playground

Write Python on the left, export a real `.mlbundle`, and inspect it on the right: manifest, schema, structure, members and samples. Part of [eshkra](https://eshkra.dev).

Intended host: <https://playground.mlbundle.eshkra.dev/>. Documentation: <https://docs.mlbundle.eshkra.dev/>.

## How it runs

Python runs in your browser on [Pyodide](https://pyodide.org/), inside a dedicated Web Worker. There is no server that executes code, and neither your code nor your files are uploaded anywhere.

- The build bundles this repository's `src/mlbundle` (every `.py` file plus the `state_fields.json` package data) into the worker, with distribution metadata whose version is read from `pyproject.toml`. The page always runs the mlbundle source it was built from, at version 0.2.0 today.
- Each run executes your code as written, in a fresh namespace and an empty working directory. Every `*.mlbundle` the code writes into that directory (at most 8 per run) is passed to `mlbundle.inspect`, then to `mlbundle.load`, which deserializes it through the closed sklearn adapter and recomputes model claims and structure. Only then is it marked **Verified**. Loading is done automatically only for bundles your own code just wrote.
- **Download .mlbundle** saves the exact bytes mlbundle wrote.
- **Inspect a .mlbundle file** (the upload icon) reads a local file in the browser and runs `mlbundle.inspect` on it. It never calls `load` and never deserializes the model, so uploaded files show **Inspected only** and their structure is withheld: structure is only trustworthy once `load` has recomputed it. Files that mlbundle rejects show **Rejected** with mlbundle's error.
- Inspection and verification use `mlbundle.ArchiveLimits` lowered for a browser tab: 64 MiB archive, compressed and per-member bytes, 128 MiB total uncompressed bytes; the other ceilings keep mlbundle's defaults. Files above 64 MiB are refused before they are read into memory.
- A run that exceeds 60 seconds, or that you stop, terminates the worker; a fresh Python starts. **Restart Python** does the same on request. `print()` output and stderr appear under **Console**, capped at 100,000 characters per run, together with the last expression and any traceback.
- Tables show at most 200 rows and tree outlines at most 511 nodes; each says what it omits. Raw `schema.json`/`structure.json` views are capped at 200,000 characters.
- Bundle contents are rendered as text only, never as HTML.

The playground is a local, in-browser tool for trying mlbundle with your own trusted code. It is not a sandbox for untrusted bundles and gives no guarantee beyond what `inspect` and `load` themselves provide; see [SECURITY.md](../SECURITY.md).

## Interface

The page follows the MLschema and MLForm playgrounds: the same header, floating options menu (design system, style, appearance, viewport size), split editor and icon controls with tooltips and accessible names. Narrow windows start with the code above the inspector; **Code position** offers every layout, and the separator resizes with the pointer or the arrow keys.

The inspector has one tab per view. Arrow keys, Home and End move between tabs.

- **Overview**: verification status, model facts, features with training dtypes, class labels, training environment and author metadata.
- **Manifest**: the canonical manifest JSON.
- **Schema**: the fields and reports mlschema inferred locally, plus raw `schema.json`.
- **Structure**: decision trees as expandable outlines read from sklearn's native arrays, a selectable tree for ensembles, coefficient and importance bars, plus raw `structure.json`.
- **Members**: archive members with the sizes and SHA-256 hashes the manifest records.
- **Samples**: rows included only through an explicit `sample_rows=N`.
- **Console**: errors, the last expression and printed output.

Ctrl/⌘ + Enter runs the code. Tab indents; press Escape, then Tab, to leave the editor. The example, your code and the layout are saved in this browser; **Reset example** restores the example.

## Runtime

Only the scikit-learn adapter runs here. XGBoost (Pyodide ships 2.1.4, below mlbundle's 3.4.1 floor), CatBoost and ONNX Runtime have no builds for this runtime, so those adapters are not offered.

| Component | Version | Source |
| --- | --- | --- |
| Pyodide | 314.0.7 (Python 3.14.2) | jsDelivr CDN |
| scikit-learn, scipy, numpy, pandas | 1.8.0, 1.18.0, 2.4.6, 3.0.2 | Pyodide distribution |
| pydantic, pydantic-core | 2.14.1, 2.50.1 | PyPI |
| skops, mlschema, prettytable, wcwidth, packaging | 0.16.0, 0.3.0, 3.18.0, 0.9.2, 26.3 | PyPI |
| typing-extensions, typing-inspection, annotated-types | 4.16.0, 0.4.4, 0.8.0 | PyPI |

PyPI pins match `uv.lock` (a unit test checks them). scikit-learn 1.8.0 satisfies mlbundle's `>=1.6`, but mlbundle's sklearn catalog is audited against 1.9.1, which has no build for this runtime. pandas 3.0.2 is likewise the newest Pyodide build. Bundles written here record the versions that actually ran, with `Emscripten`/`wasm32` as the platform; loading them with a different scikit-learn version may produce mlbundle's version diagnostics.

Pyodide's `os` module has no `link`. The harness defines one that raises `ENOSYS`, so mlbundle takes its documented publication path for file systems without hard links.

The first visit downloads Python and the wheels from `cdn.jsdelivr.net`, `pypi.org` and `files.pythonhosted.org`; later visits use the browser cache. If those hosts are unreachable, the page reports that Python could not start and offers **Restart Python**.

To move to a new Pyodide release, update `pyodide` in `package.json` and `PYODIDE_VERSION` in `src/runtime/versions.ts` together (a unit test checks they match), then review `PYPI_PINS`.

## Examples

`src/examples/*.py` are ordinary Python files: a decision tree classifier with a training-frame schema, a random forest regressor, a logistic regression with one-hot columns and explicit samples, an all-sklearn Pipeline (verified, with no structure in format version 1) and rejected models. `test:runtime` runs every example against the real package.

## Commands

From this directory, after `vp install --frozen-lockfile` at the repository root:

```bash
vp run dev          # http://127.0.0.1:10021
vp run build
vp run preview
vp run typecheck
vp run test:unit    # runtime client, upload requests, pins, version and saved state
vp run test:runtime # real Pyodide in Node: examples, verification, uploads, rejections
vp run test:e2e     # production build on port 10022, desktop and mobile Chromium
vp run test:e2e:built # same browser tests against an already built dist/
```

`test:runtime` and `test:e2e` download Pyodide and wheels, so they need network access. Before the first `test:e2e`, run `vp exec playwright install chromium`. The Python examples and harness are linted with the repository's Ruff configuration.

Actions builds once, runs `test:e2e:built`, then uploads that tested build for
Cloudflare deployment. See [website deployment](../docs/deployment.md).
