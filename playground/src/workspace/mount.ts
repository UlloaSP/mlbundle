import { icon, labelIcon } from "../icons.ts";
import { formatBytes } from "../inspector/dom.ts";
import { createPythonRuntime, type RunResult, type RuntimeState } from "../runtime/client.ts";
import { RUN_TIMEOUT_MS, UPLOAD_LIMIT_BYTES } from "../runtime/versions.ts";
import { createCodeEditor } from "./editor.ts";
import { CODE_EXAMPLES, getCodeExample } from "./examples.ts";
import { createOutputPanel } from "./output.ts";
import { bindCodeSplit } from "./split.ts";
import { isCodePosition, readWorkspaceState, saveWorkspaceState } from "./state.ts";
import { createStatus } from "./status.ts";

const MARKUP = `
  <div class="code-toolbar" role="group" aria-label="Playground controls">
    <label class="code-select"><span class="visually-hidden">Example</span><select data-role="example"></select></label>
    <label class="code-select code-position"><span class="visually-hidden">Code position</span>
      <select data-role="position">
        <option value="left">Code on the left</option><option value="right">Code on the right</option>
        <option value="top">Code above</option><option value="bottom">Code below</option>
      </select>
    </label>
    <button type="button" class="icon-button" data-role="reset" aria-label="Reset example" data-tooltip="Reset example">${icon("reset")}</button>
    <button type="button" class="icon-button" data-role="restart" aria-label="Restart Python" data-tooltip="Restart Python">${icon("restart")}</button>
    <button type="button" class="icon-button" data-role="upload" aria-label="Inspect a .mlbundle file" data-tooltip="Inspect a .mlbundle file (metadata only)">${icon("upload")}</button>
    <input type="file" data-role="file" accept=".mlbundle" hidden />
    <span class="code-status" data-role="status" role="status" aria-live="polite" tabindex="0"></span>
    <button type="button" class="icon-button code-run" data-role="run" aria-keyshortcuts="Control+Enter Meta+Enter"></button>
    <p class="code-notice" data-role="notice" role="status" hidden></p>
  </div>
  <div class="code-split">
    <section class="code-source code-pane" id="code-source" aria-label="Python code">
      <div class="code-editor"></div>
    </section>
    <div class="code-separator" role="separator" tabindex="0" aria-label="Code panel size"
      aria-controls="code-source" aria-valuemin="20" aria-valuemax="80"
      title="Drag to resize. Arrow keys: move; Enter: equal sizes."></div>
    <section class="code-output code-pane" aria-label="Bundle inspector"></section>
  </div>
`;

const failedRead = (text: string): RunResult => ({
  kind: "finished",
  durationMs: 0,
  report: { outcome: { kind: "error", text }, bundles: [], skipped: 0 },
});

export const mountWorkspace = (container: HTMLElement, storage: Storage | undefined) => {
  const first = CODE_EXAMPLES[0]!;
  let state = readWorkspaceState(storage, {
    code: first.code,
    exampleId: first.id,
    // Narrow screens start stacked; the position control still offers every layout.
    position: window.innerWidth < 720 ? "top" : "left",
    ratio: 50,
  });
  let runtimeState: RuntimeState = { status: "booting", message: "Starting Python…" };
  let autoRun = true;

  const shell = document.createElement("section");
  shell.className = "code-workspace";
  shell.setAttribute("aria-label", "Python code and bundle inspector");
  shell.innerHTML = MARKUP;
  container.replaceChildren(shell);
  const part = <T extends HTMLElement>(role: string) =>
    shell.querySelector<T>(`[data-role="${role}"]`)!;
  const exampleSelect = part<HTMLSelectElement>("example");
  const positionSelect = part<HTMLSelectElement>("position");
  const runButton = part<HTMLButtonElement>("run");
  const uploadButton = part<HTMLButtonElement>("upload");
  const fileInput = part<HTMLInputElement>("file");
  const status = createStatus(part("status"));
  const notice = part("notice");
  exampleSelect.replaceChildren(...CODE_EXAMPLES.map(({ id, label }) => new Option(label, id)));
  exampleSelect.value = getCodeExample(state.exampleId).id;
  positionSelect.value = state.position;
  const output = createOutputPanel(shell.querySelector<HTMLElement>(".code-output")!);

  const save = () => {
    if (saveWorkspaceState(storage, state)) return;
    notice.hidden = false;
    notice.textContent = "Saving in this browser is unavailable. Copy your code before leaving.";
  };
  const renderRunButton = () => {
    const running = runtimeState.status === "running";
    if (running) labelIcon(runButton, "stop", "Stop");
    else labelIcon(runButton, "run", "Run", "Run (Ctrl/⌘ + Enter)");
    runButton.setAttribute("aria-description", running
      ? "Stop this run. Python restarts."
      : `Run the code (Ctrl/⌘ + Enter). Runs stop after ${RUN_TIMEOUT_MS / 1000} seconds.`);
    runButton.disabled = runtimeState.status === "failed";
    runButton.setAttribute("aria-busy", String(running));
    // A new example would otherwise show the result of the code still running.
    exampleSelect.disabled = running;
    part<HTMLButtonElement>("reset").disabled = running;
    uploadButton.disabled = runtimeState.status !== "ready";
  };

  const run = async () => {
    if (runtimeState.status === "running") return;
    if (runtimeState.status !== "ready") {
      autoRun = runtimeState.status === "booting";
      if (autoRun) status.show("Will run when Python is ready", "loading");
      return;
    }
    output.begin();
    const result = await python.run(state.code);
    output.finish(result, "run");
    if (result.kind === "finished") {
      const failed = result.report.outcome.kind === "error";
      status.show(failed ? "Error" : "Done", failed ? "error" : "check");
    } else if (result.kind === "timeout") status.show("Timed out", "error");
    else if (result.kind === "cancelled") status.show("Stopped", "stop");
  };

  const inspectFile = async () => {
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (!file || runtimeState.status !== "ready") return;
    output.begin();
    if (file.size > UPLOAD_LIMIT_BYTES) {
      // Refused before reading, so a huge file never enters the tab's memory.
      output.finish(failedRead(`${file.name} is ${formatBytes(file.size)}. The playground inspects files up to ${formatBytes(UPLOAD_LIMIT_BYTES)}; use the mlbundle CLI for larger bundles.`), "inspect");
      status.show("File too large", "error");
      return;
    }
    let data: ArrayBuffer;
    try {
      data = await file.arrayBuffer();
    } catch {
      output.finish(failedRead(`${file.name} could not be read.`), "inspect");
      status.show("Error", "error");
      return;
    }
    const result = await python.inspect(file.name, data);
    output.finish(result, "inspect");
    if (result.kind === "finished") {
      const rejected = result.report.bundles[0]?.status !== "inspected";
      status.show(rejected ? "Rejected" : "Inspected", rejected ? "error" : "check");
    } else if (result.kind === "timeout") status.show("Timed out", "error");
  };

  const python = createPythonRuntime(
    () => new Worker(new URL("../runtime/worker.ts", import.meta.url), { type: "module" }),
    {
      onState(next) {
        runtimeState = next;
        renderRunButton();
        if (next.status === "booting") {
          status.show(next.message, "loading");
          output.showBooting(next.message);
        } else if (next.status === "failed") {
          status.show("Python unavailable", "error");
          output.showFailure(next.message);
        } else if (next.status === "running") status.show("Running…", "loading");
        else {
          output.showReady();
          if (!["Done", "Error", "Inspected", "Rejected", "File too large"].includes(status.label)) {
            status.show("Ready");
          }
          if (autoRun) {
            autoRun = false;
            void run();
          }
        }
      },
      onStream: (stream, text) => output.stream(stream, text),
    },
  );

  const editor = createCodeEditor(
    shell.querySelector<HTMLElement>(".code-editor")!,
    state.code,
    (code) => {
      state = { ...state, code };
      save();
    },
    () => void run(),
  );
  const split = bindCodeSplit(
    shell.querySelector<HTMLElement>(".code-split")!,
    shell.querySelector<HTMLElement>(".code-separator")!,
    state.position,
    state.ratio,
    (position, ratio) => {
      state = { ...state, position, ratio };
      save();
    },
  );

  const loadExample = (id: string) => {
    const example = getCodeExample(id);
    state = { ...state, exampleId: example.id, code: example.code };
    editor.setCode(example.code);
    save();
    void run();
  };
  runButton.addEventListener("click", () => {
    if (runtimeState.status === "running") python.cancel();
    else void run();
  });
  exampleSelect.addEventListener("change", () => loadExample(exampleSelect.value));
  part("reset").addEventListener("click", () => {
    loadExample(state.exampleId);
    editor.focus();
  });
  part("restart").addEventListener("click", () => {
    autoRun = false;
    python.restart();
  });
  uploadButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => void inspectFile());
  positionSelect.addEventListener("change", () => {
    if (isCodePosition(positionSelect.value)) split.setPosition(positionSelect.value);
  });
  renderRunButton();
};
