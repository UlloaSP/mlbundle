import { labelIcon } from "../icons.ts";
import type { BundleReport } from "../runtime/protocol.ts";
import { createJsonViewer } from "../workspace/viewer.ts";
import { formatBytes, h, note } from "./dom.ts";
import { overview, statusBanner } from "./overview.ts";
import { membersView, samplesView, schemaView } from "./schema.ts";
import { structureView } from "./structure.ts";

export const TABS = [
  ["overview", "Overview"],
  ["manifest", "Manifest"],
  ["schema", "Schema"],
  ["structure", "Structure"],
  ["members", "Members"],
  ["samples", "Samples"],
  ["console", "Console"],
] as const;
export type TabId = (typeof TABS)[number][0];

const RAW_LIMIT = 200_000;
const STATUS_LABEL: Record<BundleReport["status"], string> = {
  verified: "Verified",
  inspected: "Inspected only",
  unverified: "Load failed",
  rejected: "Rejected",
};

const rawJson = (title: string, value: unknown) => {
  const text = JSON.stringify(value, null, 2);
  const clipped = text.length > RAW_LIMIT ? `${text.slice(0, RAW_LIMIT)}\n… [${text.length - RAW_LIMIT} more characters]` : text;
  return h("details", { class: "inspector-raw" }, h("summary", {}, title), h("pre", { tabindex: "0" }, clipped));
};

/**
 * Bundle inspector: a bundle picker, a download action and one tab per view.
 * Tabs follow the ARIA tabs pattern: arrow keys, Home and End move between them.
 */
export const createInspector = (host: HTMLElement, consoleView: HTMLElement) => {
  const bar = h("div", { class: "inspector-bar" });
  const picker = h("select", { "aria-label": "Bundle" });
  const title = h("span", { class: "inspector-title" });
  const chip = h("span", { class: "inspector-chip" });
  const download = h("button", { type: "button", class: "icon-button" });
  labelIcon(download, "download", "Download .mlbundle");
  bar.append(h("label", { class: "code-select inspector-picker" }, picker), title, chip, download);
  const tablist = h("div", { class: "inspector-tabs", role: "tablist", "aria-label": "Bundle views" });
  const panel = h("div", { class: "inspector-panel", role: "tabpanel", tabindex: "0", id: "inspector-panel" });
  const manifestHost = h("div", { class: "output-json" });
  const manifestViewer = createJsonViewer(manifestHost, "Manifest JSON");
  host.append(bar, tablist, panel);

  let bundles: BundleReport[] = [];
  let selected = 0;
  let active: TabId = "overview";
  let empty = "Run the code to export a bundle, or open a .mlbundle file to inspect it.";
  const tabs = TABS.map(([id, label]) => {
    const tab = h("button", { type: "button", role: "tab", id: `tab-${id}`, "aria-controls": "inspector-panel" }, label);
    tab.addEventListener("click", () => show(id));
    return tab;
  });
  tablist.append(...tabs);
  tablist.addEventListener("keydown", (event) => {
    const index = TABS.findIndex(([id]) => id === active);
    const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: TABS.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    const target = (next + TABS.length) % TABS.length;
    show(TABS[target]![0]);
    tabs[target]!.focus();
  });

  const content = (bundle: BundleReport | undefined): (Node | null)[] => {
    if (active === "console") return [consoleView];
    if (!bundle) return [note(empty)];
    if (active === "manifest") {
      if (!bundle.manifest) return [statusBanner(bundle)];
      manifestViewer.show(JSON.stringify(bundle.manifest, null, 2));
      return [manifestHost];
    }
    if (!bundle.manifest) return [statusBanner(bundle)];
    switch (active) {
      case "overview":
        return [statusBanner(bundle), ...overview(bundle)];
      case "schema":
        return [...schemaView(bundle), bundle.schema ? rawJson("schema.json", bundle.schema) : null];
      case "structure":
        return [...structureView(bundle), bundle.structure ? rawJson("structure.json", bundle.structure) : null];
      case "members":
        return membersView(bundle);
      default:
        return samplesView(bundle);
    }
  };

  const render = () => {
    const bundle = bundles[selected];
    for (const [index, tab] of tabs.entries()) {
      const current = TABS[index]![0] === active;
      tab.setAttribute("aria-selected", String(current));
      tab.tabIndex = current ? 0 : -1;
    }
    panel.setAttribute("aria-labelledby", `tab-${active}`);
    panel.classList.toggle("is-manifest", active === "manifest" && Boolean(bundle?.manifest));
    panel.replaceChildren(...content(bundle).filter((item): item is Node => item !== null));
    panel.scrollTop = 0;
    bar.hidden = !bundle;
    if (!bundle) return;
    picker.parentElement!.hidden = bundles.length < 2;
    title.textContent = `${bundle.fileName} · ${formatBytes(bundle.sizeBytes)}`;
    chip.textContent = STATUS_LABEL[bundle.status];
    chip.dataset.status = bundle.status;
    download.hidden = !bundle.bytes;
  };
  const show = (id: TabId) => {
    active = id;
    render();
  };

  download.addEventListener("click", () => {
    const bundle = bundles[selected];
    if (!bundle?.bytes) return;
    const url = URL.createObjectURL(new Blob([bundle.bytes as BlobPart], { type: "application/zip" }));
    const link = h("a", { href: url, download: bundle.fileName });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  picker.addEventListener("change", () => {
    selected = Number(picker.value);
    render();
  });

  render();
  return {
    get active() {
      return active;
    },
    show,
    /** Replace the shown bundles; `message` explains an empty list. */
    setBundles(next: BundleReport[], message: string) {
      bundles = next;
      selected = 0;
      empty = message;
      picker.replaceChildren(...next.map((bundle, index) => h("option", { value: String(index) }, bundle.fileName)));
      render();
    },
  };
};

export type Inspector = ReturnType<typeof createInspector>;
