"""Canonical TOML for the lock: one spelling, so every TOML parser reads the same data."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any, TypedDict

from ._wheel_tags import MARKERS, PLATFORMS, Platform
from .errors import InvalidEnvironmentLockError

if TYPE_CHECKING:
    from .lock import LockedPackage


def marker(platforms: Sequence[Platform]) -> str:
    """The canonical marker of a proper subset of targets, in target order."""
    if len(platforms) == 1:
        return MARKERS[platforms[0]]
    return " or ".join(f"({MARKERS[platform]})" for platform in platforms)


_SUBSETS = {
    marker(subset): subset
    for size in range(1, len(PLATFORMS))
    for subset in combinations(PLATFORMS, size)
}
_TOP = {"lock-version", "created-by", "environments", "requires-python", "packages"}
_PACKAGE = {"name", "version", "marker", "wheels"}
_WHEEL = {"name", "url", "size", "hashes"}


class WheelRecord(TypedDict):
    name: str
    url: str
    size: int
    sha256: str


class PackageRecord(TypedDict):
    name: str
    version: str
    platforms: tuple[Platform, ...]
    wheels: list[WheelRecord]


def dump(python_version: str, packages: Sequence[LockedPackage]) -> bytes:
    """Emit validated values; their character sets need no TOML escapes."""
    lines = [
        'lock-version = "1.0"',
        'created-by = "mlbundle"',
        "environments = [",
        *(f'    "{MARKERS[platform]}",' for platform in PLATFORMS),
        "]",
        f'requires-python = "=={python_version}"',
    ]
    for package in packages:
        lines += ["", "[[packages]]", f'name = "{package.name}"']
        lines.append(f'version = "{package.version}"')
        if package.platforms != PLATFORMS:
            lines.append(f'marker = "{marker(package.platforms)}"')
        lines.append("wheels = [")
        lines += [
            f'    {{ name = "{wheel.name}", url = "{wheel.url}", size = {wheel.size}, '
            f'hashes = {{ sha256 = "{wheel.sha256}" }} }},'
            for wheel in package.wheels
        ]
        lines.append("]")
    return ("\n".join(lines) + "\n").encode("ascii")


def _invalid() -> InvalidEnvironmentLockError:
    return InvalidEnvironmentLockError("Invalid environment lock.")


def _table(value: object, keys: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= value.keys() <= keys:
        raise _invalid()
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise _invalid()
    return value


def _wheel(value: object) -> WheelRecord:
    wheel = _table(value, _WHEEL, _WHEEL)
    hashes = _table(wheel["hashes"], {"sha256"}, {"sha256"})
    size = wheel["size"]
    if type(size) is not int:
        raise _invalid()
    return {
        "name": _string(wheel["name"]),
        "url": _string(wheel["url"]),
        "size": size,
        "sha256": _string(hashes["sha256"]),
    }


def _package(value: object) -> PackageRecord:
    package = _table(value, _PACKAGE, _PACKAGE - {"marker"})
    platforms: tuple[Platform, ...] = PLATFORMS
    if "marker" in package:
        platforms = _SUBSETS.get(_string(package["marker"]), ())
        if not platforms:
            raise _invalid()
    wheels = package["wheels"]
    if not isinstance(wheels, list):
        raise _invalid()
    return {
        "name": _string(package["name"]),
        "version": _string(package["version"]),
        "platforms": platforms,
        "wheels": [_wheel(wheel) for wheel in wheels],
    }


def parse(data: bytes) -> tuple[str, list[PackageRecord]]:
    """Decode bounded bytes into plain records; the caller validates and re-emits."""
    try:
        document = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError):
        raise _invalid() from None
    top = _table(document, _TOP, _TOP)
    requires = _string(top["requires-python"])
    if (
        top["lock-version"] != "1.0"
        or top["created-by"] != "mlbundle"
        or top["environments"] != [MARKERS[platform] for platform in PLATFORMS]
        or not requires.startswith("==")
        or not isinstance(top["packages"], list)
    ):
        raise _invalid()
    return requires[2:], [_package(package) for package in top["packages"]]
