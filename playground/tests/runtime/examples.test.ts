import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { before, test } from "node:test";
import type { BundleReport, Json, RunReport } from "../../src/runtime/protocol.ts";
import { projectVersion, type PythonSession } from "../../src/runtime/session.ts";
import { bootNodeSession, examplesDir, readExample, repoRoot } from "./load.ts";

type Obj = { [key: string]: Json };
let session: PythonSession;
before(async () => {
  session = await bootNodeSession();
});

const runExample = async (file: string): Promise<RunReport> => {
  const report = await session.run(readExample(file));
  assert.notEqual(report.outcome.kind, "error", JSON.stringify(report.outcome));
  return report;
};
const only = (report: RunReport): BundleReport => {
  assert.equal(report.bundles.length, 1);
  const bundle = report.bundles[0]!;
  assert.equal(bundle.status, "verified", bundle.error ?? "");
  assert.equal(bundle.source, "generated");
  return bundle;
};
const value = (report: RunReport): unknown => {
  assert.equal(report.outcome.kind, "json", JSON.stringify(report.outcome));
  return JSON.parse((report.outcome as { text: string }).text);
};
const model = (bundle: BundleReport) => bundle.manifest!.model as Obj;
const names = (bundle: BundleReport) => bundle.members!.map((member) => member.name);

test("the runtime runs the repository's mlbundle on the pinned wheels", () => {
  assert.equal(session.versions.python.split(".").slice(0, 2).join("."), "3.14");
  assert.equal(session.versions.mlbundle, projectVersion(readFileSync(join(repoRoot, "pyproject.toml"), "utf8")));
  assert.equal(session.versions.mlschema, "0.3.0");
  assert.equal(session.versions.skops, "0.16.0");
  assert.match(session.versions["scikit-learn"], /^1\.8\./);
});

test("the example catalog is complete", () => {
  const files = readdirSync(examplesDir).filter((file) => file.endsWith(".py"));
  assert.deepEqual(files.sort(), [
    "classifier.py",
    "errors.py",
    "pipeline.py",
    "regressor.py",
    "samples.py",
  ]);
});

test("the classifier example writes a verified bundle with a tree", async () => {
  const report = await runExample("classifier.py");
  const bundle = only(report);
  assert.deepEqual(value(report), [["cold", "warm"]]);
  assert.equal(model(bundle).estimator_class, "DecisionTreeClassifier");
  assert.equal(model(bundle).kind, "classifier");
  assert.deepEqual(names(bundle), [
    "manifest.json",
    "model.skops",
    "schema.json",
    "structure.json",
    "pylock.toml",
  ]);
  assert.equal(bundle.environment?.pythonVersion, session.versions.python);
  assert.ok(bundle.environment?.packages.some(([name, version]) => name === "scikit-learn" && version === session.versions["scikit-learn"]));
  for (const member of bundle.members!.slice(1)) assert.match(member.sha256 ?? "", /^[0-9a-f]{64}$/);
  assert.equal(bundle.structure!.family, "decision_tree");
  assert.equal(bundle.samples, null);
  // The download is the archive mlbundle wrote: a ZIP of the reported size.
  assert.equal(bundle.bytes!.length, bundle.sizeBytes);
  assert.deepEqual([...bundle.bytes!.subarray(0, 4)], [0x50, 0x4b, 0x03, 0x04]);
  const environment = bundle.manifest!.environment as Obj;
  assert.equal(environment.os, "Emscripten");
  assert.equal(environment.packages, undefined);
});

test("the regressor example records one tree per estimator", async () => {
  const bundle = only(await runExample("regressor.py"));
  assert.equal(model(bundle).kind, "regressor");
  assert.equal(bundle.structure!.family, "tree_ensemble");
  assert.equal((bundle.structure!.trees as Json[]).length, 3);
  assert.deepEqual((bundle.manifest!.features as Obj).names, ["area_m2", "rooms", "floor"]);
});

test("samples appear only when requested and one-hot columns form one field", async () => {
  const report = await runExample("samples.py");
  const bundle = only(report);
  assert.deepEqual(value(report), ["number", "onehot-category", "boolean"]);
  assert.equal(bundle.samples!.length, 3);
  assert.ok(names(bundle).includes("samples.json"));
  assert.equal(bundle.structure!.family, "linear");
});

test("an all-sklearn pipeline is verified with null structure", async () => {
  const bundle = only(await runExample("pipeline.py"));
  assert.equal(bundle.structure, null);
  assert.ok(!names(bundle).includes("structure.json"));
  assert.equal((bundle.schema as Obj).source, "default");
});

test("rejections happen at export and write nothing", async () => {
  const report = await runExample("errors.py");
  assert.equal(report.bundles.length, 0);
  const results = value(report) as Obj[];
  assert.deepEqual(
    results.map((entry) => [entry.case, entry.error]),
    [
      ["custom subclass", "UnsupportedModelError"],
      ["HistGradientBoosting", "UnsupportedModelError"],
      ["samples without a frame", "InvalidSamplesError"],
    ],
  );
});

test("an uploaded bundle is inspected without loading its model", async () => {
  const generated = only(await runExample("classifier.py"));
  const report = await session.inspect("my model.mlbundle", generated.bytes!.slice());
  const bundle = report.bundles[0]!;
  assert.equal(bundle.status, "inspected");
  assert.equal(bundle.source, "uploaded");
  assert.equal(bundle.fileName, "my model.mlbundle");
  assert.equal(bundle.structure, null);
  assert.deepEqual(bundle.manifest, generated.manifest);
  assert.deepEqual(bundle.schema, generated.schema);
  assert.equal(bundle.bytes, undefined);
});

test("damaged and non-bundle uploads are rejected by mlbundle.inspect", async () => {
  const generated = only(await runExample("classifier.py"));
  const tampered = generated.bytes!.slice();
  const middle = Math.floor(tampered.length / 2);
  tampered[middle] = tampered[middle]! ^ 0xff;
  const damaged = (await session.inspect("damaged.mlbundle", tampered)).bundles[0]!;
  assert.equal(damaged.status, "rejected");
  assert.match(damaged.error!, /Error: /);
  const notes = new TextEncoder().encode("hello");
  const text = (await session.inspect("notes.mlbundle", notes)).bundles[0]!;
  assert.equal(text.status, "rejected");
  assert.match(text.error!, /^InvalidContainerError/);
});

test("user edits run as written in a fresh namespace and directory", async () => {
  const failed = await session.run("from mlbundle import export\nexport(object(), 'x.mlbundle')");
  assert.equal(failed.outcome.kind, "error");
  const traceback = (failed.outcome as { text: string }).text;
  assert.match(traceback, /File "<playground>", line 2[\s\S]*UnsupportedModelError/);
  assert.equal(failed.bundles.length, 0);

  const stray = await session.run("open('stray.mlbundle', 'wb').write(b'not a zip')\nleftover = 1");
  assert.equal(stray.bundles[0]!.status, "rejected");
  assert.equal(stray.bundles[0]!.bytes, undefined);
  // A fresh directory holds only the session's runtime lock, never earlier files.
  const next = await session.run("import os\nsorted(os.listdir('.'))");
  assert.deepEqual(JSON.parse((next.outcome as { text: string }).text), ["pylock.toml"]);
  assert.deepEqual(next.bundles, []);
  const name = await session.run("leftover");
  assert.match((name.outcome as { text: string }).text, /NameError/);
});
