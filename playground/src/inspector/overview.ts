import type { BundleReport } from "../runtime/protocol.ts";
import { asArray, asObject, display, facts, formatBytes, h, note, section, table } from "./dom.ts";

const STATUS: Record<BundleReport["status"], { title: string; detail: string }> = {
  verified: {
    title: "Verified with mlbundle.load",
    detail:
      "Inspected, then loaded through the closed sklearn adapter: model claims and structure were recomputed and matched.",
  },
  inspected: {
    title: "Inspected only",
    detail:
      "mlbundle.inspect validated the archive, limits, manifest, hashes, schema and samples. The model was not deserialized, so model claims and structure are not verified.",
  },
  unverified: {
    title: "Rejected by mlbundle.load",
    detail: "The metadata passed inspection, but loading and recomputing the model claims failed.",
  },
  rejected: {
    title: "Rejected by mlbundle.inspect",
    detail: "mlbundle refused this file before reading its metadata.",
  },
};

export const statusBanner = (bundle: BundleReport) => {
  const { title, detail } = STATUS[bundle.status];
  return h(
    "div",
    { class: "inspector-status", "data-status": bundle.status, role: "note" },
    h("strong", {}, title),
    h("span", {}, detail),
    bundle.error ? h("code", { class: "inspector-error" }, bundle.error) : null,
  );
};

export const overview = (bundle: BundleReport) => {
  const manifest = asObject(bundle.manifest);
  if (!bundle.manifest) {
    return [facts("File", [["File", bundle.fileName], ["Size", formatBytes(bundle.sizeBytes)]])];
  }
  const model = asObject(manifest.model);
  const features = asObject(manifest.features);
  const target = asObject(manifest.target);
  const environment = asObject(manifest.environment);
  const names = asArray(features.names);
  const dtypes = asArray(features.dtypes);
  const labels = asArray(target.class_labels);
  const targetNames = asArray(target.names);
  const metadata = asObject(manifest.user_metadata);
  return [
    facts("Model", [
      ["Estimator", model.estimator_class],
      ["Kind", model.kind],
      ["Library", model.library],
      ["Serialization", model.serialization],
      ["Features", model.n_features],
      ["Outputs", model.n_outputs],
      ["File", `${bundle.fileName} · ${formatBytes(bundle.sizeBytes)}`],
      ["Format", `${display(manifest.format_name)} ${display(manifest.format_version)}`],
      ["Writer", manifest.writer_version],
      ["Created", manifest.created_at],
    ]),
    section(
      "Features",
      note(`Origin: ${display(features.origin)}. Training dtypes are declarations; null means none were recorded.`),
      table("Features in model order", ["#", "Name", "Training dtype"], names.map((name, index) => [index, name, dtypes[index] ?? null])),
    ),
    section(
      "Targets",
      labels.length
        ? table("Class labels per output", ["Output", "Name", "Labels"], labels.map((output, index) => [index, targetNames[index] ?? null, asArray(output).map(display).join(", ")]))
        : note(targetNames.length ? `Regression targets: ${targetNames.map(display).join(", ")}` : "Regression output; no target names were declared."),
    ),
    section(
      "Training environment",
      note(`Python ${display(environment.python_version)} on ${display(environment.os)} ${display(environment.os_version)} (${display(environment.architecture)}). Declared by the writer, not authenticated.`),
      table("Package versions", ["Package", "Version"], Object.entries(asObject(environment.packages))),
    ),
    Object.keys(metadata).length
      ? section("User metadata", table("Author metadata", ["Key", "Value"], Object.entries(metadata)))
      : null,
  ];
};
