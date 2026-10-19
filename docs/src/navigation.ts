import { specPages } from "./spec";

// Sidebar order follows the reader's path: start, package, consume, choose an adapter, look up.
export const sidebar = [
  {
    label: "Start",
    items: [
      { label: "Overview", slug: "" },
      { label: "Installation", slug: "getting-started/installation" },
      { label: "Quick start", slug: "getting-started/quick-start" },
    ],
  },
  {
    label: "Guides",
    items: [
      { label: "Export a bundle", slug: "guides/export" },
      { label: "Inspect and load", slug: "guides/inspect-and-load" },
      { label: "Command line", slug: "guides/cli" },
      { label: "Schema and samples", slug: "guides/schema-and-samples" },
      { label: "Records and predictions", slug: "guides/records" },
      { label: "What is verified", slug: "guides/trust" },
    ],
  },
  {
    label: "Adapters",
    items: [
      { label: "Supported models", slug: "adapters/overview" },
      { label: "scikit-learn", slug: "adapters/sklearn" },
      { label: "XGBoost", slug: "adapters/xgboost" },
      { label: "CatBoost", slug: "adapters/catboost" },
      { label: "ONNX", slug: "adapters/onnx" },
      { label: "Composite pipelines", slug: "adapters/pipelines" },
    ],
  },
  {
    label: "Reference",
    items: [
      { label: "Python API", slug: "reference/api" },
      { label: "Errors", slug: "reference/errors" },
      { label: "Limits", slug: "reference/limits" },
      { label: "Development", slug: "reference/development" },
    ],
  },
  {
    label: "Format 1 contract",
    items: specPages.map(({ label, route }) => ({ label, link: route })),
  },
];
