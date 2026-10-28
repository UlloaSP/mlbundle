import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";
import { startSession, type PythonSession, type SourceTree } from "../../src/runtime/session.ts";

const here = fileURLToPath(new URL(".", import.meta.url));
export const repoRoot = join(here, "../../..");
export const examplesDir = join(here, "../../src/examples");

const packageFiles = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === "__pycache__" ? [] : packageFiles(path);
    return /\.(py|json)$/.test(entry.name) ? [path] : [];
  });

/** Same tree the Vite build bundles: every .py and .json file under src/mlbundle. */
export const readSources = (): SourceTree => {
  const srcRoot = join(repoRoot, "src");
  return Object.fromEntries(
    packageFiles(join(srcRoot, "mlbundle")).map((path) => [
      relative(srcRoot, path).replaceAll("\\", "/"),
      readFileSync(path, "utf8"),
    ]),
  );
};

export const readExample = (file: string) => readFileSync(join(examplesDir, file), "utf8");

/** Boots real Pyodide with the pinned wheels; downloads from jsDelivr and PyPI. */
export const bootNodeSession = async (): Promise<PythonSession> => {
  const pyodide = await loadPyodide({ stdout: () => {}, stderr: () => {} });
  return startSession(
    pyodide,
    readSources(),
    readFileSync(join(repoRoot, "pyproject.toml"), "utf8"),
    readFileSync(join(here, "../../src/runtime/harness.py"), "utf8"),
    readFileSync(join(here, "../../src/runtime/pylock.toml"), "utf8"),
  );
};
