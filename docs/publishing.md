# Manual pre-alpha publishing

Version `0.1.0a2` is the next unpublished pre-alpha; `0.1.0a1` reserved the PyPI name.
Format version 1 is not frozen; schema v2 requires older bundles to be re-exported. The security design has not had an
independent review; do not accept bundles from untrusted parties.

The owner, or an agent on the owner's explicit request for that specific version,
runs this workflow. Agents never run or approve it on their own initiative.
There is no API token, repository secret, tag, GitHub release or other publish path.

## Publishing steps

The owner, or an agent on the owner's explicit request, follows these steps:

1. Review and merge the change into `develop`, then bring it into `main` through
   the normal reviewed integration process. The workflow refuses any other ref.
2. Confirm the registered PyPI pending publisher at
   [account publishing](https://pypi.org/manage/account/publishing/) has these
   exact values: project `mlbundle`, owner `UlloaSP`, repository `mlbundle`,
   workflow `publish.yml`, environment `pypi_env`. Confirm the GitHub `pypi_env`
   environment still requires `UlloaSP` approval and permits only `main`.
3. Open [Publish to PyPI](https://github.com/UlloaSP/mlbundle/actions/workflows/publish.yml),
   choose **Run workflow**, select **main**, and run it once.
4. Wait for the build/CI jobs to succeed. Review the run's commit and downloaded
   `distributions` artifact, containing the `0.1.0a2` wheel and source archive.
   Approve the `pypi_env` deployment when satisfied. This approval permits upload.
5. Check [mlbundle on PyPI](https://pypi.org/project/mlbundle/0.1.0a2/) and install
   with `python -m pip install "mlbundle[sklearn,schema]==0.1.0a2"` on Python 3.14.

A pending publisher does not reserve a name until a successful upload. The
[PyPI pending-publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
explains this first-upload step. If publishing fails or only one artifact uploads,
inspect the PyPI state before taking further action; the preflight rejects an
existing version and the workflow does not skip existing files.

## What the workflow checks

`publish.yml` has two jobs. Its `build` job calls `ci.yml` from the same commit,
with publishing preflight enabled. That preflight checks `refs/heads/main`, reads
the version from `pyproject.toml`, and requires HTTP 404 from the exact PyPI
version endpoint. An existing version, other HTTP errors or a network failure
stop the run. One concurrency group prevents overlapping publish runs.

Reusable CI builds one wheel and source archive with Python 3.14, uv 0.12.19 and
`uv_build`. Strict Twine checks validate both artifacts' metadata and README
rendering. Both OSes check artifact contents, matching wheel/sdist package bytes,
and an isolated base-wheel install. Ubuntu and Windows install that exact wheel
and run the full test suite; lint and type checks must also succeed. Tests run
from the runner's temporary directory so CatBoost writes stay outside the checkout.

The artifact is immutable within the run. `publish` waits for all reusable CI
jobs and `pypi_env` approval, downloads `distributions` and uploads those files without
rebuilding or checking out source. Only this job has `id-token: write`; the
workflow otherwise permits only `contents: read`. Trusted Publishing exchanges
OIDC identity for short-lived PyPI credentials through the documented
[PyPA publish action](https://github.com/pypa/gh-action-pypi-publish).

Publication and OIDC authorization cannot be tested locally. The first authorized
upload is the verification of the registered publisher and approval gate.
