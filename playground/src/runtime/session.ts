import type { PyodideAPI } from "pyodide";
import type { BundleReport, RunReport, RuntimeVersions } from "./protocol.ts";
import { PYODIDE_PACKAGES, PYPI_PINS, SOURCE_ROOT } from "./versions.ts";

export interface PythonSession {
  versions: RuntimeVersions;
  run(code: string): Promise<RunReport>;
  inspect(name: string, data: Uint8Array): Promise<RunReport>;
}

/** Package-relative path, e.g. `mlbundle/inspection.py`, to file contents. */
export type SourceTree = Readonly<Record<string, string>>;

/** The [project] version, so importlib.metadata reports what the manifest records. */
export const projectVersion = (pyproject: string) => {
  const match = /^\[project\][^[]*?^version\s*=\s*"([^"]+)"/m.exec(pyproject);
  if (!match) throw new Error("pyproject.toml declares no [project] version.");
  return match[1]!;
};

const writeSources = (pyodide: PyodideAPI, sources: SourceTree, version: string) => {
  if (!Object.hasOwn(sources, "mlbundle/__init__.py")) {
    throw new Error("The mlbundle sources are missing from this build.");
  }
  for (const [path, text] of Object.entries(sources)) {
    const target = `${SOURCE_ROOT}/${path}`;
    pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
    pyodide.FS.writeFile(target, text);
  }
  // Minimal distribution metadata: mlbundle reads its own version from it.
  const info = `${SOURCE_ROOT}/mlbundle-${version}.dist-info`;
  pyodide.FS.mkdirTree(info);
  pyodide.FS.writeFile(
    `${info}/METADATA`,
    `Metadata-Version: 2.4\nName: mlbundle\nVersion: ${version}\n`,
  );
};

const UPLOAD_PATH = "/uploads/upload.mlbundle";

/**
 * Prepare a loaded Pyodide instance: install the pinned wheels, write the real
 * mlbundle sources and load the execution harness. Shared by the browser worker
 * and the Node runtime tests so both exercise the same path.
 */
export const startSession = async (
  pyodide: PyodideAPI,
  sources: SourceTree,
  pyproject: string,
  harness: string,
  onProgress: (message: string) => void = () => {},
): Promise<PythonSession> => {
  onProgress("Loading scikit-learn and pandas…");
  await pyodide.loadPackage([...PYODIDE_PACKAGES], { messageCallback: () => {} });
  onProgress("Installing pydantic, skops and mlschema…");
  const micropip = pyodide.pyimport("micropip");
  try {
    await micropip.install.callKwargs([...PYPI_PINS], { deps: false });
  } finally {
    micropip.destroy();
  }
  onProgress("Loading mlbundle…");
  writeSources(pyodide, sources, projectVersion(pyproject));
  const namespace = pyodide.toPy({});
  pyodide.runPython(harness, { globals: namespace, filename: "harness.py" });
  const versionsOf = namespace.get("runtime_versions");
  const runCode = namespace.get("run");
  const inspectUpload = namespace.get("inspect_upload");
  const versions = JSON.parse(versionsOf(SOURCE_ROOT)) as RuntimeVersions;
  versionsOf.destroy();

  // Attach the written bytes, then remove the run directory's files.
  const collect = (raw: string, source: BundleReport["source"]): RunReport => {
    const report = JSON.parse(raw) as RunReport & { bundles: (BundleReport & { path?: string })[] };
    for (const bundle of report.bundles) {
      bundle.source = source;
      if (bundle.path && bundle.status !== "rejected") {
        bundle.bytes = pyodide.FS.readFile(bundle.path) as Uint8Array;
      }
      delete bundle.path;
    }
    return report;
  };

  return {
    versions,
    async run(code) {
      return collect(await runCode(code), "generated");
    },
    async inspect(name, data) {
      pyodide.FS.mkdirTree("/uploads");
      pyodide.FS.writeFile(UPLOAD_PATH, data);
      const bundle = JSON.parse(inspectUpload(UPLOAD_PATH)) as BundleReport;
      // The visitor's own file name, never a path: the harness saw a fixed one.
      bundle.fileName = name;
      bundle.source = "uploaded";
      return { outcome: { kind: "none" }, bundles: [bundle], skipped: 0 };
    },
  };
};
