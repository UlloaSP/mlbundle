# Website deployment

The three websites deploy as Cloudflare Workers Static Assets. They need no
server-side Python: the playground runs Python in the visitor's browser.

| Site | Worker | Domain |
| --- | --- | --- |
| Landing | `mlbundle-landing` | `mlbundle.eshkra.dev` |
| Documentation | `mlbundle-docs` | `docs.mlbundle.eshkra.dev` |
| Playground | `mlbundle-playground` | `playground.mlbundle.eshkra.dev` |

Each site's `wrangler.jsonc` owns its worker name, custom domain, asset directory
and routing. Astro sites use trailing slashes and their generated 404 pages.
Cloudflare manages the custom domains and certificates during deployment.

## Triggers and checks

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main` or
`develop`, and supports **Run workflow** in GitHub Actions.

- Python gates build wheel/sdist, validate metadata, lint, check types and test
  the built wheel and base install on Ubuntu and Windows.
- The reusable `sites.yml` runs frontend lint, landing formatting, Astro/TypeScript
  checks, docs link tests, playground unit tests and real Pyodide example tests.
  It builds all three sites, validates Wrangler configuration without uploading,
  and runs Chromium tests on desktop and mobile against the built playground.
- Only successful frontend runs upload `landing-build`, `docs-build` and
  `playground-build`. Browser failures retain their traces for seven days.
- After **all** Python and frontend jobs pass, pushes to `main` and manual CI
  runs on `main` call `deploy-sites.yml`. PRs, forks and `develop` never deploy.

Deployment downloads each tested artifact from the same run; it does not rebuild.
It uses the locked Wrangler dependency through `vp exec`, deploys each site to
its own GitHub environment, and checks the resulting homepage title over HTTPS.
Deployments for the same site are serialized and are not cancelled midway.

`publish.yml` runs reuse these checks, but skip website deployment. PyPI
publication starts automatically when a push to `main` changes
`project.version`, runs only from `main` through `pypi_env`, and is independent
of website deployments. No tags or GitHub releases are created; see
[publishing](publishing.md).

## GitHub and Cloudflare setup

The repository's Actions secrets must contain `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID`. The token must authorize Workers deployments and custom
domain management for `eshkra.dev`. The zone must belong to that account and be
active in Cloudflare. Keep credentials in Actions;
never put them in a Wrangler file or commit them.

The `landing`, `docs` and `playground` GitHub environments allow deployment only
from the `main` branch. They have no manual review gate; passing CI permits the
automatic website deployment. `pypi_env` is also limited to `main` and has no
manual review gate.

Secrets being present does not prove their permissions or domain ownership.
The first successful Actions deployment verifies that external configuration.

## Local validation and redeployment

From the repository root:

```sh
vp install --frozen-lockfile
vp run check
vp run mlbundle-playground#test:runtime
vp run build
vp run check:deploy
vp run mlbundle-playground#test:e2e:built
```

Install Chromium first with `vp exec playwright install chromium` from
`playground/`. `check:deploy` uses `wrangler deploy --dry-run`; it never uploads.
Node 24.14.1 and the Vite+ version pinned in `package.json` are used in Actions.

To redeploy current `main`, select **CI Pipeline → Run workflow → main**. This
reruns every gate and does not publish to PyPI. For a rollback, revert the website
change through the usual review process and merge to `main`; the new CI run
deploys the reverted build. The repository files must be committed and pushed
before GitHub can execute a changed workflow.

The setup follows the official [Vite+ action](https://github.com/voidzero-dev/setup-vp),
[reusable workflow](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)
and [Wrangler configuration](https://developers.cloudflare.com/workers/wrangler/configuration/)
documentation.
