"""Compare a running interpreter with a bundle's lock, before loading its model."""

from __future__ import annotations

import os
import platform
import re
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata

from ._display import display
from ._lock_generation import source_tree
from ._wheel_tags import normalize
from .errors import EnvironmentMismatchError
from .results import LoadedBundle, MetadataView

_GLIBC = re.compile(r"glibc ([0-9]{1,3})\.([0-9]{1,3})")


@dataclass(frozen=True)
class RuntimeFacts:
    implementation: str
    python_version: str
    gil_disabled: bool
    # As sysconfig.get_platform() reports it, such as linux-x86_64 or win-amd64.
    platform: str
    glibc: tuple[int, int] | None
    # Installed versions; None marks duplicate and editable, directory or VCS installs.
    distributions: dict[str, str | None]

    @classmethod
    def current(cls) -> RuntimeFacts:
        distributions: dict[str, str | None] = {}
        for distribution in metadata.distributions():
            name = distribution.metadata["Name"]
            if not isinstance(name, str):
                continue
            key = normalize(name)
            distributions[key] = (
                None
                if key in distributions or source_tree(distribution)
                else distribution.version
            )
        try:
            libc = os.confstr("CS_GNU_LIBC_VERSION") or ""
        except (AttributeError, OSError, ValueError):
            libc = ""
        found = _GLIBC.fullmatch(libc)
        return cls(
            implementation=sys.implementation.name,
            python_version=platform.python_version(),
            gil_disabled=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            platform=sysconfig.get_platform(),
            glibc=None if found is None else (int(found[1]), int(found[2])),
            distributions=distributions,
        )


def environment_problems(
    bundle: MetadataView | LoadedBundle, facts: RuntimeFacts
) -> list[str]:
    lock = bundle.environment_lock
    problems = []
    if facts.implementation != "cpython" or facts.gil_disabled:
        problems.append("the interpreter is not CPython with the GIL")
    if facts.python_version != lock.python_version:
        problems.append(
            f"Python {display(facts.python_version)} is not {lock.python_version}"
        )
    if facts.platform not in lock.platforms:
        problems.append(f"platform {display(facts.platform)} is not a target")
    elif facts.platform.startswith("linux-") and (
        facts.glibc is None or facts.glibc < lock.glibc
    ):
        problems.append("glibc 2.28 or newer is required")
    expected = {
        package.name: package.version
        for package in lock.packages
        if facts.platform in package.platforms
    }
    expected["mlbundle"] = bundle.manifest.writer_version
    for name in sorted(expected.keys() | facts.distributions.keys()):
        wanted = expected.get(name)
        if name not in facts.distributions:
            problems.append(f"{name} {wanted} is not installed")
        elif wanted is None:
            problems.append(f"{display(name)} is installed but not locked")
        elif facts.distributions[name] is None:
            problems.append(f"{name} is duplicated or installed from a source tree")
        elif facts.distributions[name] != wanted:
            installed = display(facts.distributions[name])
            problems.append(f"{name} {installed} is installed, {wanted} is locked")
    return problems


def check_environment(bundle: MetadataView | LoadedBundle) -> None:
    """Require exactly the bundle's lock plus mlbundle at its writer version.

    Runtimes call this in the isolated process that will load the model. It reads
    installed distribution metadata only: it imports no model library and does
    not verify file hashes, which the installer checks against the lock.
    """
    problems = environment_problems(bundle, RuntimeFacts.current())
    if problems:
        raise EnvironmentMismatchError(
            "The environment differs from the bundle lock: "
            + "; ".join(problems[:10])
            + ("; ..." if len(problems) > 10 else "")
        )
