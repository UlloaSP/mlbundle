import { createInspector } from "../inspector/panel.ts";
import { h } from "../inspector/dom.ts";
import type { RunResult } from "../runtime/client.ts";
import type { OutputStream, RunReport } from "../runtime/protocol.ts";

const RESTART_HINT = "Check your connection, then choose Restart Python.";
const NO_BUNDLE =
  "This run wrote no .mlbundle file. Call export(model, \"name.mlbundle\", …) to write one into the run's working directory.";

/**
 * The right pane: the bundle inspector, with the run's console as its last tab.
 * The console holds errors, the last expression and print() output.
 */
export const createOutputPanel = (pane: HTMLElement) => {
  const error = h("pre", { class: "output-error", role: "alert", hidden: "" });
  const message = h("p", { class: "output-note", role: "status" });
  const result = h("pre", { class: "output-repr", tabindex: "0", "aria-label": "Last expression", hidden: "" });
  const streams = h("pre", { class: "output-streams", tabindex: "0", "aria-label": "Console output", hidden: "" });
  const consoleView = h("div", { class: "output-console" }, error, message, result, streams);
  const notice = h("p", { class: "output-banner", role: "status", hidden: "" });
  pane.append(notice);
  const inspector = createInspector(pane, consoleView);
  let bootFailed = false;

  const setText = (element: HTMLElement, text: string) => {
    element.hidden = !text;
    element.textContent = text;
  };
  const showFailure = (text: string) => {
    setText(error, `${text}\n${RESTART_HINT}`);
    setText(notice, "Python is unavailable. See Console.");
    bootFailed = true;
    inspector.show("console");
  };
  const showReport = (report: RunReport, label: string) => {
    const { outcome } = report;
    setText(error, outcome.kind === "error" ? outcome.text : "");
    setText(result, outcome.kind === "json" || outcome.kind === "repr" ? outcome.text : "");
    const skipped = report.skipped ? ` ${report.skipped} more bundle(s) were not inspected (limit 8 per run).` : "";
    setText(message, `${label}${skipped}`);
    inspector.setBundles(report.bundles, outcome.kind === "error" ? "The code raised an error before writing a bundle. See Console." : NO_BUNDLE);
    if (outcome.kind === "error" && report.bundles.length === 0) inspector.show("console");
    else if (inspector.active === "console") inspector.show("overview");
  };

  return {
    begin() {
      streams.replaceChildren();
      streams.hidden = true;
      setText(notice, "");
      pane.setAttribute("aria-busy", "true");
    },
    stream(stream: OutputStream, text: string) {
      streams.append(h("span", { class: `console-${stream}` }, text));
      streams.hidden = false;
    },
    finish(run: RunResult, action: "run" | "inspect") {
      pane.setAttribute("aria-busy", "false");
      if (run.kind === "timeout") {
        setText(error, `Stopped after ${run.limitMs / 1000} seconds. Python restarted, so variables and imports were reset.`);
        setText(result, "");
        inspector.setBundles([], "The run timed out before finishing.");
        inspector.show("console");
      } else if (run.kind === "cancelled") {
        setText(error, "");
        setText(message, "Stopped. Python restarted, so variables and imports were reset.");
        inspector.setBundles([], "The run was stopped.");
      } else if (run.kind === "failed") {
        showFailure(run.message);
      } else if (run.kind === "unavailable") {
        setText(notice, "Python is not ready yet.");
      } else {
        const seconds = (run.durationMs / 1000).toFixed(1);
        const label = action === "inspect" ? `Inspected in ${seconds} s with mlbundle.inspect.` : `Finished in ${seconds} s.`;
        if (run.report.outcome.kind === "none" && action === "run") setText(result, "");
        showReport(run.report, label);
      }
    },
    /** A startup failure stays visible until Python boots again. */
    showFailure,
    showBooting(text: string) {
      if (bootFailed) {
        bootFailed = false;
        setText(error, "");
      }
      setText(notice, `${text} The first visit downloads Python, scikit-learn and pandas; later visits use your browser cache.`);
    },
    showReady() {
      if (!bootFailed) setText(notice, "");
    },
  };
};
