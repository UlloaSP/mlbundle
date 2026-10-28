const github = "https://github.com/UlloaSP/mlbundle";
const blob = `${github}/blob/main`;

// Builds link the intended hosts; `vp run dev` links the sibling docs and playground dev servers.
const dev = import.meta.env.DEV;

export const links = {
  home: "https://mlbundle.eshkra.dev/",
  eshkra: "https://eshkra.dev/",
  docs: dev ? "http://127.0.0.1:4341/" : "https://docs.mlbundle.eshkra.dev/",
  playground: dev ? "http://127.0.0.1:10021/" : "https://playground.mlbundle.eshkra.dev/",
  github,
  issues: `${github}/issues`,
  license: `${blob}/LICENSE`,
  security: `${blob}/SECURITY.md`,
  format: `${blob}/docs/format.md`,
  formatSchema: `${blob}/docs/format-schema.md`,
  pypi: "https://pypi.org/project/mlbundle/",
};

// Docs pages the landing points into; the base already switches between dev and build.
export const docs = {
  install: `${links.docs}getting-started/installation/`,
  quickStart: `${links.docs}getting-started/quick-start/`,
  inspectAndLoad: `${links.docs}guides/inspect-and-load/`,
  trust: `${links.docs}guides/trust/`,
  schema: `${links.docs}guides/schema-and-samples/`,
  adapters: `${links.docs}adapters/overview/`,
  format: `${links.docs}format/`,
};

export const description =
  "Package a trained model, a verifiable manifest and a locally inferred schema in one .mlbundle file. Native formats for scikit-learn, XGBoost, CatBoost and ONNX, never pickle.";

// The release install from the docs; checkout installs live on the docs installation page.
export const install = 'pip install "mlbundle[sklearn,schema,lock]"';

// The eshkra path from a trained model to a reviewed prediction, as on eshkra.dev.
export const family = [
  { name: "mlbundle", stage: "Package", href: "https://mlbundle.eshkra.dev/" },
  { name: "mlschema", stage: "Describe", href: "https://mlschema.eshkra.dev/" },
  { name: "mlform", stage: "Render", href: "https://mlform.eshkra.dev/" },
  { name: "mlsuite", stage: "Run & review", href: "https://github.com/UlloaSP/mlsuite" },
] as const;
