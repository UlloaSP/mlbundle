// Pinned so every visitor gets the same interpreter and wheels.
export const PYODIDE_VERSION = "314.0.7";
export const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Wheels shipped with this Pyodide release: numpy 2.4.6, pandas 3.0.2, scipy 1.18.0 and
// scikit-learn 1.8.0. mlbundle's sklearn catalog is audited against 1.9.1, which has no
// build for this ABI; the manifest records the versions that actually ran.
export const PYODIDE_PACKAGES = [
  "micropip",
  "numpy",
  "pandas",
  "scipy",
  "scikit-learn",
] as const;

// Pure-Python wheels from PyPI at the versions in the repository's uv.lock.
// Pyodide's pydantic (2.12.5) is below mlbundle's floor; pydantic-core publishes a
// pyemscripten_2026_0 build. The others have no compiled code.
export const PYPI_PINS = [
  "pydantic==2.14.1",
  "pydantic-core==2.50.1",
  "typing-extensions==4.16.0",
  "typing-inspection==0.4.4",
  "annotated-types==0.8.0",
  "skops==0.16.0",
  "prettytable==3.18.0",
  "wcwidth==0.9.2",
  "packaging==26.3",
  "mlschema==0.3.0",
] as const;

// Where the repository's src/mlbundle tree is written inside the Pyodide file system.
export const SOURCE_ROOT = "/mlbundle/src";

export const BOOT_TIMEOUT_MS = 240_000;
export const RUN_TIMEOUT_MS = 60_000;
export const OUTPUT_LIMIT_CHARS = 100_000;

// Uploads above this size are refused before they are read into the browser.
export const UPLOAD_LIMIT_BYTES = 64 * 1024 * 1024;
