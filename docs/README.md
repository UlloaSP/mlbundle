# mlbundle documentation site

The Astro Starlight site for `docs.mlbundle.eshkra.dev`, package `mlbundle-docs` in the repository's
pnpm workspace. Run every command from the repository root with Vite+ (`vp`) and Node.js 22.18+,
24.11+ or 26+:

```shell
vp install --frozen-lockfile
vp run mlbundle-docs#dev      # http://127.0.0.1:4341/
vp run mlbundle-docs#build    # astro check, then a static build in docs/dist
```

In development the header links point to the local landing (`4340`) and playground (`10021`);
built output links to the production hosts.

From `docs/`, use `vp run dev`, `vp run build` or `vp run preview`. These scripts
invoke Astro through `vp exec`; bare `vp dev` and `vp build` run Vite's own
commands and do not build an Astro site.

## Single-sourced contract

The Markdown files beside this README (`format*.md`) and the root `SECURITY.md` are the normative
contract. The site renders them in place through the `spec` collection in `src/content.config.ts`,
mapped to routes in `src/spec.ts`; relative links become site routes or GitHub links. Edit those
files, never a copy. `publishing.md` is maintainer guidance and is not rendered.

Guides live in `src/content/docs/`. Their Python fences are checked by `uv run ruff format --check`.
The link resolver has a self-test:

```shell
vp run mlbundle-docs#test
```

The brand mark is read from the repository's `brand/` directory. Actions builds,
checks and deploys this site with the other frontends; see [website deployment](deployment.md).
