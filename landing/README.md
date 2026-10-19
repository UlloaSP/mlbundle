# MLbundle landing

The product site intended for <https://mlbundle.eshkra.dev>, built with Astro. It shares the
eshkra palette, Mona Sans and JetBrains Mono fonts, and the original MLbundle mark from the
repository's `brand/mlbundle.png`. The page follows the visitor's light or dark system preference.

## Local development

Install once from the repository root, which owns the pnpm workspace and lockfile:

```sh
vp install --frozen-lockfile
```

Then, from this directory, with Node.js 22.18+, 24.11+, or 26+:

```sh
vp run dev
```

Open <http://127.0.0.1:4340>. In development, docs and playground links point to the sibling
dev servers on ports 4341 and 10021; builds link the intended `*.mlbundle.eshkra.dev` hosts.
`vp run build` checks Astro/TypeScript and produces the static site in `dist/`; `vp run preview`
serves that build. Use `vp run check:format` to check formatting or `vp run format` to apply it.

Actions deploys the tested build to Cloudflare after all checks pass on `main`.
See [website deployment](../docs/deployment.md) for triggers and configuration.
Use these package scripts rather than a bare `vp build`, which runs Vite directly and cannot build
Astro.

## The example is real output

`examples/train.py` trains the model whose export and load the hero shows, from the `export(`
call on. `scripts/example.py` runs it in a temporary directory with the repository's `uv`
environment and stores the run in `src/data/example.json`: the printed prediction, the bundle's
members, sizes and hashes, its manifest and schema, and the CLI output of `inspect` and `verify`.
It then edits one claim in `manifest.json` (`max_depth` from `null` to `3`) and records how
`inspect` and `verify` treat that copy. Every version, size, member and terminal line on the page
comes from this file. The build does not run Python; regenerate the stored output after changing
the example or the library:

```sh
vp run example
```

The bundle id, timestamp, environment and `model.skops` hash change on every run.
