import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { projectVersion } from "../../src/runtime/session.ts";
import { PYODIDE_VERSION, PYPI_PINS } from "../../src/runtime/versions.ts";
import { readWorkspaceState, saveWorkspaceState } from "../../src/workspace/state.ts";

const FALLBACK = { code: "x", exampleId: "classifier", position: "left", ratio: 50 } as const;
const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("the CDN loader version matches the installed pyodide package", () => {
  const installed = JSON.parse(read("../../node_modules/pyodide/package.json")) as { version: string };
  assert.equal(PYODIDE_VERSION, installed.version);
});

test("pure-Python pins match the repository lock", () => {
  const lock = read("../../../uv.lock");
  for (const pin of PYPI_PINS) {
    const [name, version] = pin.split("==") as [string, string];
    assert.match(lock, new RegExp(`name = "${name}"\nversion = "${version.replaceAll(".", "\\.")}"`), pin);
  }
});

test("the bundled package reports the version pyproject.toml declares", () => {
  const pyproject = read("../../../pyproject.toml");
  const declared = /^version = "([^"]+)"/m.exec(pyproject)![1];
  assert.equal(projectVersion(pyproject), declared);
  assert.equal(projectVersion('[tool.x]\nversion = "9"\n[project]\nname = "a"\nversion = "1.2"\n'), "1.2");
  assert.throws(() => projectVersion('[tool.x]\nversion = "9"\n'), /no \[project\] version/);
});

test("workspace state restores valid fields and falls back on invalid ones", () => {
  const storage = new Map<string, string>();
  const backend = {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => void storage.set(key, value),
  };
  assert.deepEqual(readWorkspaceState(backend, FALLBACK), FALLBACK);
  assert.equal(
    saveWorkspaceState(backend, { code: "y", exampleId: "samples", position: "top", ratio: 95 }),
    true,
  );
  assert.deepEqual(readWorkspaceState(backend, FALLBACK), {
    code: "y",
    exampleId: "samples",
    position: "top",
    ratio: 80,
  });
  assert.equal([...storage.keys()][0], "mlbundle-playground-workspace");
  storage.set([...storage.keys()][0]!, '{"position":"diagonal","ratio":"wide"}');
  assert.deepEqual(readWorkspaceState(backend, FALLBACK), FALLBACK);
  storage.set([...storage.keys()][0]!, "{not json");
  assert.deepEqual(readWorkspaceState(backend, FALLBACK), FALLBACK);
});

test("saving reports unavailable storage", () => {
  const full = {
    setItem() {
      throw new Error("QuotaExceededError");
    },
  };
  assert.equal(saveWorkspaceState(full, FALLBACK), false);
  assert.equal(saveWorkspaceState(undefined, FALLBACK), false);
});
