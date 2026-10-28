# Publishing to PyPI

Version `0.3.0` is prepared by this checkout. Merging a version change into
`main` automatically starts the PyPI workflow; no one needs to start a release
run manually. Format version 1 is not frozen, so incompatible manifest/schema
shapes require re-export; 0.3.0 requires every bundle to carry `pylock.toml`, as
documented in [the environment lock contract](format-environment.md). The security design has not had an
independent review; do not accept bundles from untrusted parties.

## Release flow

1. Set `project.version` in `pyproject.toml` to a new version in a reviewed
   change, then merge it into `main` through the normal integration process.
2. `publish.yml` checks that the push is to `main`, that the version changed
   from the previous commit, and that this exact version is not already on
   PyPI. Unchanged versions skip publication; an existing version, network
   failure or unexpected PyPI response fails the workflow.
3. The workflow runs the full reusable CI: Python packaging, lint, type checks,
   tests on Ubuntu and Windows, and all frontend checks, tests and builds.
4. If every check passes, Actions uploads the exact CI-built wheel and source
   archive to [mlbundle on PyPI](https://pypi.org/project/mlbundle/).
   Install with `python -m pip install "mlbundle[sklearn,schema,lock]==0.3.0"` on
   Python 3.14, replacing the example version as needed.

The GitHub `pypi_env` environment contains the project-scoped
`MLBUNDLE_API_TOKEN` Actions secret and allows deployments from `main` only.
The workflow uses it as the PyPI token; the token value must never appear in
the repository or logs. There is no manual approval gate, tag, GitHub release
or alternate publish workflow.

## Checks and recovery

`publish.yml` uses a read-only preflight before calling `ci.yml`. It skips when
the version has not changed and rejects versions already on PyPI. The reusable
CI checks the same version again before building, so a release that appears
during the run blocks publication. One concurrency group prevents overlapping
publish runs.

The CI artifact is immutable within the run. `publish` waits for all reusable
CI jobs, downloads the `distributions` artifact and uploads those files without
rebuilding or checking out source. The workflow has `contents: read`; only the
publish job receives the environment-scoped token. Website deployment is
skipped here and follows its own automatic path after successful CI on `main`,
described in [website deployment](deployment.md).

If upload fails after PyPI accepted one artifact, inspect the version on PyPI
before retrying. The preflight refuses any existing version, and the publish
action does not skip duplicate files.
