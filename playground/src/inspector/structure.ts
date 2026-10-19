import type { BundleReport } from "../runtime/protocol.ts";
import { asArray, asObject, display, h, note, section, type JsonObject } from "./dom.ts";

const NODE_LIMIT = 511;
const numbers = (value: unknown) => asArray(value).map((item) => (typeof item === "number" ? item : Number.NaN));

/** Relative bars for importances or coefficients, as a table so values stay readable. */
const bars = (caption: string, names: readonly string[], values: readonly number[]) => {
  const peak = Math.max(...values.map((value) => Math.abs(value)).filter(Number.isFinite), 0) || 1;
  return h(
    "div",
    { class: "inspector-table", tabindex: "0", role: "region", "aria-label": caption },
    h(
      "table",
      { class: "inspector-bars" },
      h("caption", {}, caption),
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Feature"), h("th", { scope: "col" }, "Value"), h("th", { scope: "col" }, h("span", { class: "visually-hidden" }, "Relative size")))),
      h(
        "tbody",
        {},
        ...values.map((value, index) => {
          const width = Number.isFinite(value) ? (Math.abs(value) / peak) * 100 : 0;
          return h(
            "tr",
            {},
            h("td", {}, names[index] ?? `#${index}`),
            h("td", {}, display(value)),
            h("td", { "aria-hidden": "true" }, h("span", { class: "inspector-bar-fill", "data-sign": value < 0 ? "negative" : "positive", style: `width: ${width.toFixed(1)}%` })),
          );
        }),
      ),
    ),
  );
};

/** One sklearn tree as an expandable outline, read straight from the native arrays. */
export const treeOutline = (tree: JsonObject, names: readonly string[]): HTMLElement[] => {
  const left = numbers(tree.children_left);
  const right = numbers(tree.children_right);
  const feature = numbers(tree.feature);
  const threshold = asArray(tree.threshold);
  const value = asArray(tree.value);
  const samples = asArray(tree.n_node_samples);
  let rendered = 0;
  const node = (index: number, depth: number, branch: string): HTMLElement => {
    rendered += 1;
    const leaf = left[index] === -1 || left[index] === undefined;
    const counts = `samples ${display(samples[index])} · value ${display(value[index])}`;
    if (leaf || rendered >= NODE_LIMIT) {
      return h("li", { class: "tree-leaf" }, h("span", { class: "tree-branch" }, branch), ` ${leaf ? "leaf" : "…"} · ${counts}`);
    }
    const name = names[feature[index]!] ?? `feature ${display(feature[index])}`;
    const summary = h("summary", {}, h("span", { class: "tree-branch" }, branch), " ", h("strong", {}, name), ` ≤ ${display(threshold[index])}`, h("span", { class: "tree-meta" }, ` · ${counts}`));
    const children = h("ul", {}, node(left[index]!, depth + 1, "true"), node(right[index]!, depth + 1, "false"));
    const open = depth < 3 ? { open: "" } : {};
    return h("li", {}, h("details", open, summary, children));
  };
  const root = h("ul", { class: "tree-outline", "aria-label": "Decision tree" }, node(0, 0, "root"));
  const total = Number(tree.node_count);
  return [
    note(`${display(tree.node_count)} nodes, depth ${display(tree.max_depth)}. Splits send rows with feature ≤ threshold to "true". value is sklearn's native node value, shown as stored.`),
    root,
    rendered < total ? note(`Showing the first ${rendered} of ${total} nodes; the raw JSON below has every node.`) : null,
  ].filter((item) => item !== null);
};

const ensemble = (structure: JsonObject, names: readonly string[]) => {
  const trees = asArray(structure.trees).map(asObject);
  const host = h("div", { class: "tree-host" });
  const select = h("select", { "aria-label": "Tree" }, ...trees.map((_, index) => h("option", { value: String(index) }, `Tree ${index + 1} of ${trees.length}`)));
  const show = () => host.replaceChildren(...treeOutline(trees[Number(select.value)] ?? {}, names));
  select.addEventListener("change", show);
  show();
  return [note(`Estimator shape ${display(structure.estimator_shape)}.`), h("label", { class: "code-select" }, select), host];
};

const linear = (structure: JsonObject, names: readonly string[]) => {
  const coef = asArray(structure.coef);
  const rows = Array.isArray(coef[0]) ? coef.map(numbers) : [numbers(coef)];
  return [
    note(`Intercept: ${display(structure.intercept)}. Coefficients apply to the model's input features, in order.`),
    ...rows.map((row, index) => bars(rows.length > 1 ? `Coefficients, row ${index}` : "Coefficients", names, row)),
  ];
};

export const structureView = (bundle: BundleReport) => {
  const structure = bundle.structure;
  if (!structure) {
    if (bundle.status === "inspected") {
      return [note("Uploaded files are inspected without deserializing the model, so the playground does not show their structure: structure is only trustworthy once mlbundle.load has recomputed it from the model.")];
    }
    if (bundle.status !== "verified") return [note("No verified structure: mlbundle did not load this bundle.")];
    return [note("This model has no structure in format version 1 (for example an all-scikit-learn Pipeline or an MLP). Claims still cover the whole model.")];
  }
  const names = asArray(asObject(bundle.manifest?.features).names).map(String);
  const family = String(structure.family);
  const importances = asArray(structure.feature_importances);
  const body =
    family === "decision_tree" ? treeOutline(asObject(structure.tree), names)
    : family === "tree_ensemble" ? ensemble(structure, names)
    : family === "linear" ? linear(structure, names)
    : [];
  return [
    note(`Family: ${family}. Recomputed by mlbundle.load from the loaded model and compared with structure.json.`),
    ...body,
    importances.length ? section("Feature importances", bars("Feature importances", names, numbers(importances))) : null,
  ];
};
