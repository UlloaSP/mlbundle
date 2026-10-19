import { defineConfig, devices } from "@playwright/test";

// Tests the production build on its own port, so a running dev server on 10021 is untouched.
export default defineConfig({
  testDir: "./tests/e2e",
  // First boot downloads Python, scikit-learn and pandas from jsDelivr and PyPI.
  timeout: 240_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:10022", trace: "retain-on-failure" },
  webServer: {
    command: "vp preview --host 127.0.0.1 --port 10022 --strictPort",
    url: "http://127.0.0.1:10022",
    reuseExistingServer: false,
    timeout: 120_000,
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] } },
  ],
});
