"""Training-side locks: installed versions pin one uv resolution per target."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from ._display import display
from ._wheel_tags import PLATFORMS, UV_PLATFORMS, Platform, normalize, select
from .errors import (
    InvalidEnvironmentLockError,
    LockGenerationError,
    MissingOptionalDependencyError,
)
from .lock import (
    NAME_PATTERN,
    VERSION_PATTERN,
    EnvironmentLock,
    LockedPackage,
    LockedWheel,
)

TIMEOUT_SECONDS = 600


def source_tree(distribution: metadata.Distribution) -> bool:
    """Whether PEP 610 records an editable, directory or VCS install.

    Their versions do not identify files. Wheels installed from a file or URL
    (archive_info) keep the version their metadata declares.
    """
    record = distribution.read_text("direct_url.json")
    if record is None:
        return False
    try:
        value = json.loads(record)
    except ValueError:
        return True
    return not isinstance(value, dict) or "archive_info" not in value


def installed_distributions() -> dict[str, str | None]:
    """First distribution per normalized name on sys.path, as imports resolve it.

    Source-tree and local-version installs map to None: no index can supply
    their files, so they can never be pinned faithfully.
    """
    found: dict[str, str | None] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if not isinstance(name, str):
            continue
        name = normalize(name)
        if name in found or re.fullmatch(NAME_PATTERN, name) is None:
            continue
        version = distribution.version
        faithful = (
            isinstance(version, str)
            and re.fullmatch(VERSION_PATTERN, version) is not None
            and not source_tree(distribution)
        )
        found[name] = version if faithful else None
    return found


def roots(
    groups: Sequence[Sequence[str]], installed: Mapping[str, str | None]
) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for group in groups:
        present = [name for name in group if name in installed]
        if len(present) != 1:
            raise LockGenerationError(
                f"Exactly one of {', '.join(group)} must be installed to lock."
            )
        version = installed[present[0]]
        if version is None:
            raise LockGenerationError(
                f"{present[0]} is installed from a source tree or local version."
            )
        pinned[present[0]] = version
    return pinned


def _uv() -> str:
    try:
        from uv import find_uv_bin  # noqa: PLC0415
    except ImportError:
        raise MissingOptionalDependencyError("lock") from None
    return find_uv_bin()


def _compile(
    uv: str, work: Path, python_version: str, target: Platform
) -> list[dict[str, object]]:
    output = work / f"pylock.{target.replace('-', '_')}.toml"
    command = [
        uv, "pip", "compile", "requirements.in", "--constraints", "constraints.txt",
        "--python-version", python_version,
        "--python-platform", UV_PLATFORMS[target],
        "--only-binary", ":all:", "--no-header", "--quiet", "--color", "never",
        "--output-file", output.name,
    ]  # fmt: skip
    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise LockGenerationError("uv timed out resolving the lock.") from None
    if result.returncode != 0:
        detail = display(result.stderr.strip()[-2000:], limit=2048)
        raise LockGenerationError(f"uv could not lock {target}: {detail}")
    with output.open("rb") as stream:
        packages = tomllib.load(stream).get("packages", [])
    if not isinstance(packages, list):
        raise LockGenerationError("uv wrote an unexpected lock.")
    return packages


def _wheel(
    package: dict[str, object], target: Platform, minor: int
) -> tuple[str, str, LockedWheel]:
    name, version = package.get("name"), package.get("version")
    files = package.get("wheels")
    if not isinstance(name, str) or not isinstance(version, str):
        raise LockGenerationError("uv wrote an unexpected lock.")
    records = {}
    for file in files if isinstance(files, list) else []:
        url = file.get("url") if isinstance(file, dict) else None
        if isinstance(url, str):
            records[unquote(urlsplit(url).path.rpartition("/")[2])] = file
    chosen = select(list(records), target, minor)
    if chosen is None:
        raise LockGenerationError(
            f"{display(name)} {display(version)} has no single preferred CPython "
            f"wheel for {target}."
        )
    file = records[chosen]
    try:
        wheel = LockedWheel.model_validate(
            {
                "name": chosen,
                "url": file["url"],
                "size": file.get("size"),
                "sha256": (file.get("hashes") or {}).get("sha256"),
            }
        )
    except (ValidationError, AttributeError):
        raise LockGenerationError(
            f"{display(name)} has an unsupported wheel record."
        ) from None
    return normalize(name), version, wheel


def resolve_lock(
    python_version: str, pins: Mapping[str, str], constraints: Mapping[str, str]
) -> EnvironmentLock:
    """Resolve pinned roots per target with uv, wheels only, and merge the results."""
    uv = _uv()
    minor = int(python_version.split(".")[1])
    found: dict[str, tuple[str, list[Platform], dict[str, LockedWheel]]] = {}
    with tempfile.TemporaryDirectory(prefix="mlbundle-lock-") as directory:
        work = Path(directory)
        lines = (f"{name}=={version}\n" for name, version in sorted(pins.items()))
        (work / "requirements.in").write_text("".join(lines), encoding="utf-8")
        lines = (
            f"{name}=={version}\n" for name, version in sorted(constraints.items())
        )
        (work / "constraints.txt").write_text("".join(lines), encoding="utf-8")
        for target in PLATFORMS:
            for package in _compile(uv, work, python_version, target):
                name, version, wheel = _wheel(package, target, minor)
                previous = found.setdefault(name, (version, [], {}))
                if previous[0] != version:
                    raise LockGenerationError(
                        f"{name} resolves to different versions per platform."
                    )
                previous[1].append(target)
                previous[2][wheel.name] = wheel
    try:
        return EnvironmentLock(
            python_version=python_version,
            packages=tuple(
                LockedPackage(
                    name=name,
                    version=version,
                    platforms=tuple(targets),
                    wheels=tuple(wheels[key] for key in sorted(wheels)),
                )
                for name, (version, targets, wheels) in sorted(found.items())
                if name != "mlbundle"
            ),
        )
    except ValidationError:
        raise LockGenerationError("The resolved environment is not lockable.") from None


def check_fidelity(
    lock: EnvironmentLock, installed: Mapping[str, str | None] | None = None
) -> None:
    """Every locked distribution present at training must be the installed one."""
    current = installed_distributions() if installed is None else installed
    if lock.python_version != platform.python_version():
        raise InvalidEnvironmentLockError("The lock's Python is not the running one.")
    problems = []
    for package in lock.packages:
        if package.name not in current:
            continue  # Needed only on other targets, such as NCCL on Linux.
        version = current[package.name]
        if version != package.version:
            problems.append(
                f"{package.name} locked {package.version}, installed "
                f"{'from a source tree' if version is None else display(version)}"
            )
    if problems:
        raise InvalidEnvironmentLockError(
            "The lock disagrees with the training environment: "
            + "; ".join(problems[:10])
            + ("; ..." if len(problems) > 10 else "")
        )


def generate_lock(groups: Sequence[Sequence[str]]) -> EnvironmentLock:
    """Lock the running interpreter's installed versions of the model's runtime."""
    installed = installed_distributions()
    pins = roots(groups, installed)
    constraints = {
        name: version
        for name, version in installed.items()
        if version is not None and name != "mlbundle"
    }
    return resolve_lock(platform.python_version(), pins, constraints)
