type Child = Node | string | number | null | undefined | false;

/**
 * Builds elements from text only. Bundle contents are untrusted (uploads included),
 * so nothing from a bundle is ever parsed as HTML.
 */
export const h = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attributes: Record<string, string> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] => {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    element.append(typeof child === "object" ? child : String(child));
  }
  return element;
};

/** Short display for any JSON value, including mlbundle's {"$float": "nan"} tags. */
export const display = (value: unknown): string => {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(+value.toPrecision(6));
  if (typeof value === "string") return value;
  if (typeof value === "object" && !Array.isArray(value) && "$float" in value) {
    return String((value as { $float: unknown }).$float);
  }
  const text = JSON.stringify(value);
  return text.length > 160 ? `${text.slice(0, 157)}…` : text;
};

export const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
};

/** An accessible data table. Rows are capped; the caption reports what is hidden. */
export const table = (
  caption: string,
  headers: readonly string[],
  rows: readonly (readonly unknown[])[],
  limit = 200,
) => {
  const shown = rows.slice(0, limit);
  const note = rows.length > limit ? ` (first ${limit} of ${rows.length})` : "";
  return h(
    "div",
    { class: "inspector-table", tabindex: "0", role: "region", "aria-label": caption },
    h(
      "table",
      {},
      h("caption", {}, `${caption}${note}`),
      h("thead", {}, h("tr", {}, ...headers.map((name) => h("th", { scope: "col" }, name)))),
      h(
        "tbody",
        {},
        ...shown.map((row) => h("tr", {}, ...row.map((cell) => h("td", {}, display(cell))))),
      ),
    ),
  );
};

/** A term/description list for key facts. */
export const facts = (label: string, entries: readonly (readonly [string, unknown])[]) =>
  h(
    "dl",
    { class: "inspector-facts", "aria-label": label },
    ...entries.flatMap(([term, value]) => [h("dt", {}, term), h("dd", {}, display(value))]),
  );

export const section = (title: string, ...children: Child[]) =>
  h("section", { class: "inspector-section" }, h("h3", {}, title), ...children);

export const note = (text: string) => h("p", { class: "inspector-note" }, text);

export type JsonObject = { [key: string]: unknown };
export const asObject = (value: unknown): JsonObject =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as JsonObject) : {};
export const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);
