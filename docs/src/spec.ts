// The normative contract lives in the repository's Markdown files; the site renders them unchanged.
const repository = "https://github.com/UlloaSP/mlbundle";

/** Repository file (relative to the root) and its route on this site. */
export const specPages = [
  { file: "docs/format.md", route: "/format/", label: "Format 1" },
  { file: "docs/format-container.md", route: "/format/container/", label: "Container and limits" },
  { file: "docs/format-adapters.md", route: "/format/adapters/", label: "Adapters and structure" },
  { file: "docs/format-state.md", route: "/format/state/", label: "Prediction state" },
  { file: "docs/format-schema.md", route: "/format/schema/", label: "Schema and samples" },
  { file: "docs/format-pipeline.md", route: "/format/pipeline/", label: "Composite pipelines" },
  { file: "docs/format-catboost.md", route: "/format/catboost/", label: "CatBoost mapping" },
  { file: "docs/format-onnx.md", route: "/format/onnx/", label: "ONNX mapping" },
  { file: "SECURITY.md", route: "/security/", label: "Security policy" },
] as const;

const byFile = new Map<string, string>(specPages.map(({ file, route }) => [file, route]));

const posix = (path: string) => path.replaceAll("\\", "/");

/** Resolves a relative link written for GitHub to its route here, or to the file on GitHub. */
export function resolveSpecLink(from: string, url: string): string {
  if (/^([a-z]+:|#|\/)/i.test(url)) return url;
  const [path = "", hash] = url.split("#");
  const parts = posix(from).split("/").slice(0, -1);
  for (const part of path.split("/")) {
    if (part === "..") parts.pop();
    else if (part && part !== ".") parts.push(part);
  }
  const target = parts.join("/");
  const route = byFile.get(target);
  const anchor = hash ? `#${hash}` : "";
  return route ? `${route}${anchor}` : `${repository}/blob/main/${target}${anchor}`;
}

/** The spec's first-level heading, used as the page title. */
export function specTitle(body: string, fallback: string): string {
  return body.match(/^# (.+)$/m)?.[1]?.trim() ?? fallback;
}

/**
 * Prepares a spec for rendering without changing the file: drops the first-level heading, which
 * Starlight renders as the title, and points repository-relative links at this site or GitHub.
 */
export function specMarkdown(file: string, body: string): string {
  return body
    .replace(/^# .+\r?\n/m, "")
    .replace(/\]\(([^)\s]+)\)/g, (_, url: string) => `](${resolveSpecLink(file, url)})`);
}
