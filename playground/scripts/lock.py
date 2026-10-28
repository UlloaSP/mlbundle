"""Write the playground's pylock.toml for the exact versions its Pyodide session runs.

Pyodide cannot run uv, so playground exports pass this lock as environment_lock.
It locks pydantic, scikit-learn and skops at the session's versions for Linux
and Windows, like a training-machine export. Rerun after changing the Pyodide
release or PYPI_PINS; export rejects a lock that disagrees with the installed
versions.
Usage, from playground/: uv run --project .. python scripts/lock.py
"""

import json
import re
import sys
from pathlib import Path

from mlbundle._lock_generation import resolve_lock
from mlbundle._wheel_tags import normalize

PLAYGROUND = Path(__file__).resolve().parents[1]
VERSIONS = PLAYGROUND / "src" / "runtime" / "versions.ts"
PYODIDE = PLAYGROUND / "node_modules" / "pyodide" / "pyodide-lock.json"
OUTPUT = PLAYGROUND / "src" / "runtime" / "pylock.toml"
ROOTS = ("pydantic", "scikit-learn", "skops")


def session_versions() -> tuple[str, dict[str, str]]:
    """Python and the distributions startSession loads: Pyodide packages, then pins."""
    source = VERSIONS.read_text(encoding="utf-8")
    loaded = re.search(r"PYODIDE_PACKAGES = \[(.*?)\]", source, re.DOTALL)
    pins = re.search(r"PYPI_PINS = \[(.*?)\]", source, re.DOTALL)
    assert loaded is not None and pins is not None
    lock = json.loads(PYODIDE.read_text(encoding="utf-8"))
    packages = {normalize(name): value for name, value in lock["packages"].items()}
    versions: dict[str, str] = {}
    pending = [normalize(name) for name in re.findall(r'"([^"]+)"', loaded[1])]
    while pending:
        name = pending.pop()
        if name in versions:
            continue
        versions[name] = packages[name]["version"]
        pending.extend(normalize(item) for item in packages[name]["depends"])
    for name, version in re.findall(r'"([^"=]+)==([^"]+)"', pins[1]):
        versions[normalize(name)] = version
    return lock["info"]["python"], versions


def main() -> None:
    python, versions = session_versions()
    pins = {name: versions[name] for name in ROOTS}
    lock = resolve_lock(python, pins, versions)
    OUTPUT.write_bytes(lock.to_bytes())
    sys.stdout.write(f"{OUTPUT.name}: Python {python}, {len(lock.packages)} packages\n")


if __name__ == "__main__":
    main()
