"""Check a main-branch version change before the automatic PyPI workflow."""

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def check(ref: str, project: dict, previous_version: str | None = None) -> bool:
    if ref != "refs/heads/main":
        raise SystemExit("Publishing requires refs/heads/main; select main.")
    version = project["version"]
    if version == previous_version:
        sys.stdout.write(f"Version remains {version}; skipping PyPI publication.\n")
        return False
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
    return True


def version_at(ref: str) -> str | None:
    if not ref or set(ref) == {"0"}:
        return None
    result = subprocess.run(
        ["git", "show", f"{ref}:pyproject.toml"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SystemExit("Could not read the previous pyproject.toml version.")
    return tomllib.loads(result.stdout)["project"]["version"]


def write_output(path: str, publish: bool) -> None:
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(f"publish={str(publish).lower()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-ref")
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    previous_version = version_at(args.previous_ref) if args.previous_ref else None
    publish = check(os.environ["GITHUB_REF"], project, previous_version)
    if args.previous_ref and os.environ.get("GITHUB_OUTPUT"):
        write_output(os.environ["GITHUB_OUTPUT"], publish)
