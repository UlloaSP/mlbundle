// Run with `vp run test`: the spec pages must keep resolving every relative link they contain.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { resolveSpecLink, specMarkdown, specPages, specTitle } from "./spec.ts";

const root = new URL("../../", import.meta.url);
const github = "https://github.com/UlloaSP/mlbundle/blob/main/";

test("relative spec links become site routes with anchors", () => {
  assert.equal(resolveSpecLink("docs/format.md", "format-onnx.md"), "/format/onnx/");
  assert.equal(
    resolveSpecLink("docs/format.md", "format-adapters.md#implemented-xgboost-mapping"),
    "/format/adapters/#implemented-xgboost-mapping",
  );
  assert.equal(resolveSpecLink("docs/format-state.md", "format.md"), "/format/");
  assert.equal(resolveSpecLink("docs/format-schema.md", "../SECURITY.md"), "/security/");
  assert.equal(resolveSpecLink("SECURITY.md", "docs/format-schema.md"), "/format/schema/");
});

test("other repository files link to GitHub, absolute links are untouched", () => {
  assert.equal(resolveSpecLink("SECURITY.md", "tests/test_cli.py"), `${github}tests/test_cli.py`);
  assert.equal(resolveSpecLink("docs/format.md", "../AGENTS.md#guardrails"), `${github}AGENTS.md#guardrails`);
  assert.equal(resolveSpecLink("docs/format.md", "#members"), "#members");
  assert.equal(resolveSpecLink("docs/format.md", "https://example.com/a.md"), "https://example.com/a.md");
});

test("rendering drops only the title heading", () => {
  const body = "# Title\r\n\nSee [x](format.md).\n\n## Section\n";
  assert.equal(specTitle(body, "fallback"), "Title");
  assert.equal(specMarkdown("docs/format-onnx.md", body), "\nSee [x](/format/).\n\n## Section\n");
});

test("every relative link in the real spec files resolves", () => {
  for (const { file } of specPages) {
    const body = readFileSync(new URL(file, root), "utf8");
    assert.notEqual(specTitle(body, ""), "", `${file} has a title`);
    for (const [, url] of body.matchAll(/\]\(([^)\s]+)\)/g)) {
      const resolved = resolveSpecLink(file, url!);
      if (resolved.startsWith(github)) {
        const path = resolved.slice(github.length).split("#")[0]!;
        assert.ok(existsSync(new URL(path, root)), `${file}: ${url} points to a missing file`);
      } else if (!/^(https?:|#)/.test(resolved)) {
        const route = resolved.split("#")[0];
        assert.ok(specPages.some((page) => page.route === route), `${file}: ${url}`);
      }
    }
  }
});
