import { readFile } from "node:fs/promises";
import { defineCollection } from "astro:content";
import { z } from "astro/zod";
import type { Loader } from "astro/loaders";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";
import { specMarkdown, specPages, specTitle } from "./spec";

const root = new URL("../../", import.meta.url);

// The normative format and security files, read in place from the repository root on every build.
const specLoader: Loader = {
  name: "mlbundle-spec",
  load: async ({ store, renderMarkdown, watcher }) => {
    store.clear();
    for (const { file, label } of specPages) {
      const url = new URL(file, root);
      const body = await readFile(url, "utf8");
      store.set({
        id: file,
        data: { title: specTitle(body, label) },
        body,
        rendered: await renderMarkdown(specMarkdown(file, body), { fileURL: url }),
      });
    }
    watcher?.add(specPages.map(({ file }) => new URL(file, root).pathname));
  },
};

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
  spec: defineCollection({ loader: specLoader, schema: z.object({ title: z.string() }) }),
};
