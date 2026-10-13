"""Read-only preflight for the owner's manual PyPI workflow."""

import os
import sys
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def check(ref: str, project: dict) -> None:
    if ref != "refs/heads/main":
        raise SystemExit("Publishing requires refs/heads/main; select main.")
    version = project["version"]
    url = f"https://pypi.org/pypi/mlbundle/{version}/json"
    try:
        with urlopen(url, timeout=30):
            pass
    except HTTPError as error:
        error.close()
        if error.code != 404:
            raise SystemExit(f"PyPI version check failed: HTTP {error.code}.") from None
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"PyPI version check failed: {error}.") from None
    else:
        raise SystemExit(f"mlbundle {version} already exists on PyPI; refusing upload.")
    sys.stdout.write(f"mlbundle {version} is not on PyPI (HTTP 404); main ref OK.\n")


if __name__ == "__main__":
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    check(os.environ["GITHUB_REF"], project)
