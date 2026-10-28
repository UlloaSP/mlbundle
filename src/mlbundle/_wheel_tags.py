"""Closed wheel filename and tag rules for the lock's fixed targets."""

import re
from typing import Literal

# Targets are named as sysconfig.get_platform() reports them on the host.
Platform = Literal["linux-x86_64", "linux-aarch64", "win-amd64"]
PLATFORMS: tuple[Platform, ...] = ("linux-x86_64", "linux-aarch64", "win-amd64")
# The oldest glibc every Linux target host must provide: manylinux_2_28 and older.
GLIBC = (2, 28)
# PEP 508 markers selecting each target, and uv's matching --python-platform.
MARKERS: dict[Platform, str] = {
    "linux-x86_64": "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "linux-aarch64": "sys_platform == 'linux' and platform_machine == 'aarch64'",
    "win-amd64": "sys_platform == 'win32' and platform_machine == 'AMD64'",
}
UV_PLATFORMS: dict[Platform, str] = {
    "linux-x86_64": "x86_64-manylinux_2_28",
    "linux-aarch64": "aarch64-manylinux_2_28",
    "win-amd64": "x86_64-pc-windows-msvc",
}

_WHEEL = re.compile(
    r"(?P<dist>[A-Za-z0-9_.]+)-(?P<version>[A-Za-z0-9_.!]+)"
    r"(?:-[0-9][A-Za-z0-9_.]*)?"
    r"-(?P<python>[A-Za-z0-9_.]+)-(?P<abi>[A-Za-z0-9_.]+)-(?P<platform>[A-Za-z0-9_.]+)"
    r"\.whl"
)
_MANYLINUX = re.compile(r"manylinux_([0-9]{1,3})_([0-9]{1,3})_(x86_64|aarch64)")
_LEGACY = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}
_CPYTHON = re.compile(r"cp3([0-9]{1,3})")


def normalize(name: str) -> str:
    """PEP 503 project name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_identity(filename: str) -> tuple[str, str] | None:
    """Normalized distribution name and version spelled by a wheel filename."""
    match = _WHEEL.fullmatch(filename)
    if match is None:
        return None
    return normalize(match["dist"]), match["version"]


def _platform_compatible(tag: str, platform: Platform) -> bool:
    if tag == "any":
        return True
    if platform == "win-amd64":
        return tag == "win_amd64"
    architecture = platform.removeprefix("linux-")
    match = _MANYLINUX.fullmatch(tag)
    if match is not None:
        glibc = (int(match[1]), int(match[2]))
        return match[3] == architecture and glibc <= GLIBC
    legacy, _, machine = tag.partition("_")
    return machine == architecture and legacy in _LEGACY


def _abi_rank(python: str, abi: str, minor: int) -> int | None:
    """Installer preference: exact CPython ABI, then the stable ABI, then none."""
    cpython = _CPYTHON.fullmatch(python)
    if abi == f"cp3{minor}" and python == f"cp3{minor}":
        return 0
    # Free-threaded builds use a distinct ABI tag (cp3Nt) and never match here.
    if abi == "abi3" and cpython is not None and 2 <= int(cpython[1]) <= minor:
        return 1
    if abi == "none" and python in ("py3", f"py3{minor}", f"cp3{minor}"):
        return 2
    return None


def rank(filename: str, platform: Platform, minor: int) -> tuple[int, int] | None:
    """Preference of a wheel on a target, lower first, as installers order tags.

    Platform wheels precede pure ones; None means the target cannot install it.
    """
    match = _WHEEL.fullmatch(filename)
    if match is None:
        return None
    ranks = [
        (int(tag == "any"), abi_rank)
        for python in match["python"].split(".")
        for abi in match["abi"].split(".")
        for tag in match["platform"].split(".")
        if (abi_rank := _abi_rank(python, abi, minor)) is not None
        and _platform_compatible(tag, platform)
    ]
    return min(ranks, default=None)


def select(filenames: list[str], platform: Platform, minor: int) -> str | None:
    """The one wheel an installer takes on the target, or None if none or tied."""
    ranked = sorted(
        (value, name)
        for name in filenames
        if (value := rank(name, platform, minor)) is not None
    )
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        return None
    return ranked[0][1]
