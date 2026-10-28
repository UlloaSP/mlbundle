"""The bundle's PEP 751 runtime lock: a closed canonical subset mlbundle never installs."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated

from pydantic import Field, StringConstraints, ValidationError, model_validator

from . import _lock_toml
from ._base import ContractModel
from ._wheel_tags import GLIBC, PLATFORMS, Platform, select, wheel_identity
from .canonical import canonical_dumps
from .contract import Library
from .errors import InvalidEnvironmentLockError
from .manifest import Manifest

MAX_PACKAGES = 512
# Normalized names and public versions: no local versions, which no index serves.
NAME_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*"
VERSION_PATTERN = (
    r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
)
PackageName = Annotated[
    str, StringConstraints(max_length=128, pattern=f"^{NAME_PATTERN}$")
]
PackageVersion = Annotated[
    str, StringConstraints(max_length=64, pattern=f"^{VERSION_PATTERN}$")
]
_PYTHON = re.compile(r"3\.([0-9]{1,3})\.[0-9]{1,3}(?:(?:a|b|rc)[0-9]{1,3})?")
# Quotes, backslashes, spaces and controls never occur, so TOML needs no escapes.
_URL = re.compile(r"https://[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]{1,2040}")

# Distributions a loaded model needs besides mlbundle itself, as in the extras.
# Each group needs exactly one member; xgboost-cpu is XGBoost without NCCL.
_BASE = (("pydantic",),)
_LIBRARY = {
    Library.SKLEARN: (("scikit-learn",), ("skops",)),
    Library.XGBOOST: (("xgboost", "xgboost-cpu"), ("scikit-learn",)),
    Library.CATBOOST: (("catboost",),),
    Library.ONNX: (("onnx",), ("onnxruntime",)),
}


class LockedWheel(ContractModel):
    """One wheel file: name, download URL, size and SHA-256 digest."""

    name: Annotated[str, StringConstraints(max_length=256)]
    url: Annotated[str, StringConstraints(max_length=2048)]
    size: int = Field(gt=0, le=2**40)
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def addressed_file(self) -> LockedWheel:
        if wheel_identity(self.name) is None:
            raise ValueError("Invalid wheel filename.")
        if _URL.fullmatch(self.url) is None or not self.url.endswith("/" + self.name):
            raise ValueError("Wheel URLs are HTTPS and end with the wheel filename.")
        return self


class LockedPackage(ContractModel):
    """A pinned distribution, the targets that install it and their wheels."""

    name: PackageName
    version: PackageVersion
    platforms: tuple[Platform, ...] = Field(min_length=1, max_length=len(PLATFORMS))
    wheels: tuple[LockedWheel, ...] = Field(min_length=1, max_length=len(PLATFORMS))

    @model_validator(mode="after")
    def consistent_files(self) -> LockedPackage:
        if list(self.platforms) != [p for p in PLATFORMS if p in self.platforms]:
            raise ValueError("Package platforms must be unique and ordered.")
        names = [wheel.name for wheel in self.wheels]
        if names != sorted(set(names)):
            raise ValueError("Wheels must be unique and sorted by name.")
        if any(wheel_identity(name) != (self.name, self.version) for name in names):
            raise ValueError("Wheel filenames must match the package and version.")
        return self


class EnvironmentLock(ContractModel):
    """Validated lock contents; an unauthenticated claim, never instructions here."""

    python_version: Annotated[str, StringConstraints(max_length=32)]
    packages: tuple[LockedPackage, ...] = Field(min_length=1, max_length=MAX_PACKAGES)

    @model_validator(mode="after")
    def installable(self) -> EnvironmentLock:
        version = _PYTHON.fullmatch(self.python_version)
        if version is None:
            raise ValueError("Invalid CPython version.")
        names = [package.name for package in self.packages]
        if names != sorted(set(names)):
            raise ValueError("Packages must be unique and sorted by name.")
        if "mlbundle" in names:
            raise ValueError("The writer_version pins mlbundle; the lock omits it.")
        minor = int(version[1])
        for package in self.packages:
            names = [wheel.name for wheel in package.wheels]
            chosen = {select(names, platform, minor) for platform in package.platforms}
            # Each target installs one unambiguous file; no file is unused.
            if None in chosen or chosen != set(names):
                raise ValueError("Each target needs one preferred wheel, all used.")
        return self

    @property
    def platforms(self) -> tuple[Platform, ...]:
        """Linux x86_64/aarch64 with glibc >= 2.28 and Windows x64, with the GIL."""
        return PLATFORMS

    @property
    def glibc(self) -> tuple[int, int]:
        return GLIBC

    @property
    def environment_id(self) -> str:
        """SHA-256 identity of Python and every file, independent of download URLs."""
        record = {
            "python": self.python_version,
            "packages": [
                [
                    p.name,
                    p.version,
                    list(p.platforms),
                    [[w.name, w.sha256] for w in p.wheels],
                ]
                for p in self.packages
            ],
        }
        return hashlib.sha256(canonical_dumps(record)).hexdigest()

    def version(self, name: str) -> str | None:
        for package in self.packages:
            if package.name == name:
                return package.version
        return None

    def to_bytes(self) -> bytes:
        return _lock_toml.dump(self.python_version, self.packages)

    @classmethod
    def from_bytes(cls, data: bytes) -> EnvironmentLock:
        """Parse the canonical TOML subset; other spellings are rejected."""
        python_version, packages = _lock_toml.parse(data)
        try:
            lock = cls(
                python_version=python_version,
                packages=tuple(
                    LockedPackage(
                        name=package["name"],
                        version=package["version"],
                        platforms=package["platforms"],
                        wheels=tuple(LockedWheel(**w) for w in package["wheels"]),
                    )
                    for package in packages
                ),
            )
        except (ValidationError, TypeError):
            raise InvalidEnvironmentLockError("Invalid environment lock.") from None
        if lock.to_bytes() != data:
            raise InvalidEnvironmentLockError("Environment lock is not canonical.")
        return lock


def runtime_requirements(
    library: Library, leaf: Library | None = None
) -> tuple[tuple[str, ...], ...]:
    """Distribution groups a loaded model needs: pydantic and the adapters' libraries."""
    groups = [*_BASE, *_LIBRARY[library]]
    if leaf is not None:
        groups.extend(group for group in _LIBRARY[leaf] if group not in groups)
    return tuple(groups)


def check_bundle_lock(lock: EnvironmentLock, manifest: Manifest) -> None:
    """Bind the lock to the manifest's Python and to the model's libraries."""
    environment = manifest.environment
    if environment.python_implementation != "CPython":
        raise InvalidEnvironmentLockError("Environment locks require CPython.")
    if lock.python_version != environment.python_version:
        raise InvalidEnvironmentLockError(
            "The lock's Python differs from the training Python."
        )
    pipeline = manifest.model.pipeline
    leaf = None if pipeline is None else pipeline.estimator.library
    names = {package.name: package for package in lock.packages}
    for group in runtime_requirements(manifest.model.library, leaf):
        present = [names[name] for name in group if name in names]
        if len(present) != 1 or present[0].platforms != PLATFORMS:
            raise InvalidEnvironmentLockError(
                f"The lock needs exactly one of {', '.join(group)} on every target."
            )
