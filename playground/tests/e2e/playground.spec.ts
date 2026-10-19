import { readFile } from "node:fs/promises";
import { expect, test, type FrameLocator, type Page } from "@playwright/test";

const frame = (page: Page) => page.frameLocator('iframe[title="Python editor"]');
const editor = (ws: FrameLocator) => ws.getByRole("textbox", { name: "Python code" });
const tab = (ws: FrameLocator, name: string) => ws.getByRole("tab", { name, exact: true });
const panel = (ws: FrameLocator) => ws.getByRole("tabpanel");
const chip = (ws: FrameLocator) => ws.locator(".inspector-chip");

const replaceCode = async (page: Page, code: string) => {
  await editor(frame(page)).click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.press("Delete");
  // insertText keeps indentation exactly; typing would trigger auto-indent.
  await page.keyboard.insertText(code);
};

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
});

test("exports, verifies and visualizes the classifier example", async ({ page }) => {
  const ws = frame(page);
  await expect(chip(ws)).toHaveText("Verified");
  await expect(page).toHaveTitle("MLbundle Playground");
  await expect(page.getByRole("link", { name: "MLbundle home" })).toBeVisible();
  await expect(ws.locator(".inspector-title")).toContainText("classifier.mlbundle");
  await expect(panel(ws)).toContainText("DecisionTreeClassifier");
  await expect(panel(ws)).toContainText("Emscripten");
  // Narrow screens start with the code stacked above the inspector.
  const narrow = page.viewportSize()!.width < 720;
  await expect(ws.locator(".code-split")).toHaveAttribute("data-position", narrow ? "top" : "left");
  await expect(ws.getByRole("combobox", { name: "Bundle" })).toBeHidden();

  await tab(ws, "Structure").click();
  await expect(panel(ws)).toContainText("Family: decision_tree");
  await expect(ws.getByRole("list", { name: "Decision tree" })).toContainText("≤");
  // Arrow keys move between tabs, as in the ARIA tabs pattern.
  await tab(ws, "Structure").press("ArrowRight");
  await expect(tab(ws, "Members")).toBeFocused();
  await expect(panel(ws)).toContainText("model.skops");
  await tab(ws, "Members").press("ArrowRight");
  await expect(panel(ws)).toContainText("sample_rows=N");
  await tab(ws, "Samples").press("ArrowLeft");
  await tab(ws, "Members").press("Home");
  await expect(tab(ws, "Overview")).toHaveAttribute("aria-selected", "true");
  await tab(ws, "Schema").click();
  await expect(panel(ws)).toContainText("Predicted class");
  await tab(ws, "Manifest").click();
  await expect(ws.getByRole("textbox", { name: "Manifest JSON" })).toContainText('"format_name": "mlbundle"');
});

test("downloads the written bundle and inspects uploads without loading them", async ({ page }) => {
  const ws = frame(page);
  await expect(chip(ws)).toHaveText("Verified");
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    ws.getByRole("button", { name: "Download .mlbundle" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("classifier.mlbundle");
  const bytes = await readFile((await download.path())!);
  expect([...bytes.subarray(0, 2)]).toEqual([0x50, 0x4b]);

  const input = ws.locator('input[type="file"]');
  await input.setInputFiles({ name: "uploaded.mlbundle", mimeType: "application/zip", buffer: bytes });
  await expect(chip(ws)).toHaveText("Inspected only");
  await expect(ws.locator(".inspector-title")).toContainText("uploaded.mlbundle");
  await expect(ws.getByRole("button", { name: "Download .mlbundle" })).toBeHidden();
  await tab(ws, "Structure").click();
  await expect(panel(ws)).toContainText("without deserializing the model");

  await input.setInputFiles({ name: "notes.mlbundle", mimeType: "application/zip", buffer: Buffer.from("hello") });
  await expect(chip(ws)).toHaveText("Rejected");
  await tab(ws, "Overview").click();
  await expect(panel(ws)).toContainText("InvalidContainerError");
});

test("runs edited code, reports errors and keeps examples resettable", async ({ page }) => {
  const ws = frame(page);
  await expect(chip(ws)).toHaveText("Verified");
  await replaceCode(page, "from mlbundle import export\nprint('about to fail')\nexport(object(), 'x.mlbundle')");
  await page.keyboard.press("ControlOrMeta+Enter");
  await expect(tab(ws, "Console")).toHaveAttribute("aria-selected", "true");
  await expect(panel(ws)).toContainText("UnsupportedModelError");
  await expect(panel(ws)).toContainText("about to fail");

  await ws.getByRole("combobox", { name: "Example" }).selectOption("regressor");
  await expect(ws.locator(".inspector-title")).toContainText("regressor.mlbundle");
  await tab(ws, "Structure").click();
  await expect(ws.getByRole("combobox", { name: "Tree" })).toHaveValue("0");
  await ws.getByRole("combobox", { name: "Tree" }).selectOption("2");
  await expect(panel(ws)).toContainText("Feature importances");

  await replaceCode(page, "1 + 1");
  await ws.getByRole("button", { name: "Reset example" }).click();
  await expect(editor(ws)).toContainText("RandomForestRegressor");
  await expect(chip(ws)).toHaveText("Verified");
});
