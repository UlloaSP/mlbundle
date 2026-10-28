import type { FromWorker, ToWorker } from "./protocol.ts";
import harness from "./harness.py?raw";
import { startSession, type PythonSession } from "./session.ts";
import { MLBUNDLE_SOURCES, PLAYGROUND_LOCK, PYPROJECT } from "./sources.ts";
import { OUTPUT_LIMIT_CHARS, PYODIDE_INDEX_URL } from "./versions.ts";

// The DOM and WebWorker libs conflict in one program; type only what is used.
const scope = self as unknown as {
  postMessage(message: FromWorker, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent<ToWorker>) => void): void;
};
const post = (message: FromWorker, transfer: Transferable[] = []) =>
  scope.postMessage(message, transfer);
let runId = 0;
let written = 0;

const forward = (stream: "stdout" | "stderr") => (line: string) => {
  if (runId === 0 || written > OUTPUT_LIMIT_CHARS) return;
  written += line.length + 1;
  const text =
    written > OUTPUT_LIMIT_CHARS ? "\n[Output truncated: over 100,000 characters.]\n" : `${line}\n`;
  post({ type: "stream", id: runId, stream, text });
};

const boot = async (): Promise<PythonSession> => {
  post({ type: "progress", message: "Downloading Python…" });
  // Loaded from the pinned CDN release so the loader always matches its WebAssembly.
  const { loadPyodide } = (await import(
    /* @vite-ignore */ `${PYODIDE_INDEX_URL}pyodide.mjs`
  )) as typeof import("pyodide");
  const pyodide = await loadPyodide({
    indexURL: PYODIDE_INDEX_URL,
    stdout: forward("stdout"),
    stderr: forward("stderr"),
  });
  return startSession(pyodide, MLBUNDLE_SOURCES, PYPROJECT, harness, PLAYGROUND_LOCK, (message) =>
    post({ type: "progress", message }),
  );
};

const session = boot();
session.then(
  ({ versions }) => post({ type: "ready", versions }),
  (error: unknown) =>
    post({ type: "boot-error", message: error instanceof Error ? error.message : String(error) }),
);

scope.addEventListener("message", async ({ data }) => {
  const python = await session;
  runId = data.id;
  written = 0;
  const started = performance.now();
  try {
    const report =
      data.type === "run"
        ? await python.run(data.code)
        : await python.inspect(data.name, new Uint8Array(data.data));
    const transfer = report.bundles.flatMap((bundle) => (bundle.bytes ? [bundle.bytes.buffer] : []));
    post({ type: "done", id: data.id, report, durationMs: performance.now() - started }, transfer);
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    const report = { outcome: { kind: "error", text } as const, bundles: [], skipped: 0 };
    post({ type: "done", id: data.id, report, durationMs: 0 });
  } finally {
    runId = 0;
  }
});
