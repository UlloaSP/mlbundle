import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { fileURLToPath } from "node:url";
import { sidebar } from "./src/navigation";

const repo = (path: string) =>
  fileURLToPath(new URL(`../${path}`, import.meta.url)).replaceAll("\\", "/");

export default defineConfig({
  site: "https://docs.mlbundle.eshkra.dev",
  trailingSlash: "always",
  devToolbar: { enabled: false },
  vite: {
    // The docs read the shared brand assets at the repository root.
    server: {
      fs: {
        allow: [fileURLToPath(new URL(".", import.meta.url)), repo("brand")],
      },
    },
  },
  integrations: [
    starlight({
      title: "mlbundle",
      description:
        "Pickle-free, verifiable model bundles with a manifest, a local schema and model structure.",
      components: {
        Head: "./src/components/BrandHead.astro",
        SiteTitle: "./src/components/BrandTitle.astro",
        SocialIcons: "./src/components/SocialIcons.astro",
        Footer: "./src/components/Footer.astro",
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/UlloaSP/mlbundle",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/UlloaSP/mlbundle/edit/main/docs/",
      },
      expressiveCode: {
        // The landing's code themes, switched with the documentation theme.
        themes: ["github-dark-default", "github-light-default"],
        styleOverrides: {
          borderRadius: "12px",
          borderColor: "var(--mb-border)",
          codeFontFamily: "var(--brand-font-mono)",
          uiFontFamily: "var(--brand-font-sans)",
          codeFontSize: "0.8125rem",
          codeLineHeight: "1.7",
          frames: {
            shadowColor: "transparent",
            editorTabBarBackground: "var(--mb-subtle)",
            editorActiveTabBackground: "var(--mb-code-bg)",
            editorActiveTabIndicatorTopColor: "var(--mb-ember)",
            editorActiveTabIndicatorBottomColor: "transparent",
            editorBackground: "var(--mb-code-bg)",
            terminalTitlebarBackground: "var(--mb-subtle)",
            terminalBackground: "var(--mb-code-bg)",
          },
        },
      },
      customCss: [
        "./src/styles/typography.css",
        "./src/styles/theme.css",
        "./src/styles/content.css",
        "./src/styles/home.css",
      ],
      sidebar,
    }),
  ],
});
