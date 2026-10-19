---
title: Development
description: Set up a checkout of mlbundle, run its checks, and build this documentation site locally.
---

## The library

mlbundle uses [uv](https://docs.astral.sh/uv/) on Python 3.14, [Ruff](https://docs.astral.sh/ruff/)
and [Pyright](https://microsoft.github.io/pyright/). The default development install includes
every extra.

```bash
uv python install 3.14
uv sync --locked --all-extras
uv run pre-commit install
```

Run the checks from the repository root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

`pytest` measures branch coverage with an 80% minimum. CI runs lint and type checks and tests the
built wheel on Ubuntu and Windows, including clean installs without extras. After changing
dependency metadata, run `uv sync` and commit the updated `uv.lock`.

Contributor rules, including the security guardrails and the 300-line file limit, are in
[AGENTS.md](https://github.com/UlloaSP/mlbundle/blob/main/AGENTS.md). Report vulnerabilities
privately as described in the [security policy](/security/#reporting-a-vulnerability).

## This documentation

The site is an Astro Starlight project in `docs/`, part of the repository's pnpm workspace and run
with Vite+ (`vp`). Use Node.js 22.18+, 24.11+ or 26+:

```bash
vp install --frozen-lockfile
vp run mlbundle-docs#dev    # http://127.0.0.1:4341/
vp run mlbundle-docs#build
```

The [Format 1 contract](/format/) and [security policy](/security/) pages render the repository's
`docs/format*.md` and `SECURITY.md` files directly, so the specification has one source. Edit
those files, not a copy.
