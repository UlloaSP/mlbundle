import type { BundleReport } from "../runtime/protocol.ts";
import { asArray, asObject, display, h, note, section, table, type JsonObject } from "./dom.ts";

const detail = (field: JsonObject) => {
  const options = asArray(field.options).map(asObject);
  if (field.kind === "onehot-category") return options.map((option) => `${display(option.label)} → ${display(option.mappedTo)}`).join(", ");
  const parts: string[] = [];
  for (const key of ["step", "min", "max", "options", "categories", "dtype"]) {
    if (key in field) parts.push(`${key}: ${display(field[key])}`);
  }
  const columns = asArray(field.columns).map(asObject);
  if (columns.length) parts.push(`columns: ${columns.map((column) => `${display(column.id)} (${display(column.kind)})`).join(", ")}`);
  return parts.join(" · ");
};

export const schemaView = (bundle: BundleReport) => {
  if (!bundle.schema) return [note("This bundle has no readable schema.")];
  const schema = bundle.schema;
  const fields = asArray(schema.fields).map(asObject);
  const reports = asArray(schema.reports).map(asObject);
  return [
    note(`Schema version ${display(schema.schema_version)}, inferred locally by mlschema from ${schema.source === "default" ? "one default row of 1.0 values" : "the training frame"}. One-hot separator: "${display(schema.onehot_separator)}".`),
    section(
      "Fields",
      table("Input fields", ["Label", "Kind", "Mapped to", "Required", "Details"], fields.map((field) => [field.label, field.kind, field.mappedTo ?? "options", field.required, detail(field)])),
    ),
    section(
      "Reports",
      table("Outputs", ["Label", "Kind", "Mapped to", "Labels"], reports.map((report) => [report.label, report.kind, report.mappedTo, asArray(report.labels).map(display).join(", ") || null])),
    ),
  ];
};

export const membersView = (bundle: BundleReport) => [
  note("Members in archive order. The manifest records the SHA-256 and size of every other member; mlbundle checked them. Hashes detect corruption, not deliberate tampering."),
  table("Archive members", ["Member", "Size (bytes)", "SHA-256"], (bundle.members ?? []).map((member) => [member.name, member.size ?? "manifest", member.sha256 ?? "—"])),
];

export const samplesView = (bundle: BundleReport) => {
  if (!bundle.samples) {
    return [note("No samples. mlbundle includes sample rows only when export is called with sample_rows=N, because they are real training data.")];
  }
  const names = asArray(asObject(bundle.manifest?.features).names).map(String);
  return [
    h("p", { class: "inspector-note" }, `${bundle.samples.length} explicitly requested feature rows, without target or index.`),
    table("Sample rows", ["#", ...names], bundle.samples.map((row, index) => [index, ...names.map((name) => row[name])])),
  ];
};
