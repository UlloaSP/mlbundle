"""Synthetic canonical locks: valid structure with placeholder files, for offline tests."""

import hashlib

from mlbundle import EnvironmentLock, LockedPackage, LockedWheel
from mlbundle._wheel_tags import PLATFORMS


def pure_wheel(name: str, version: str) -> LockedWheel:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    return LockedWheel(
        name=filename,
        url=f"https://files.example.org/packages/{filename}",
        size=1024,
        sha256=hashlib.sha256(filename.encode()).hexdigest(),
    )


def synthetic_lock(python_version: str, packages: dict[str, str]) -> EnvironmentLock:
    return EnvironmentLock(
        python_version=python_version,
        packages=tuple(
            LockedPackage(
                name=name,
                version=version,
                platforms=PLATFORMS,
                wheels=(pure_wheel(name, version),),
            )
            for name, version in sorted(packages.items())
        ),
    )


SKLEARN_PACKAGES = {"pydantic": "2.14.1", "scikit-learn": "1.9.1", "skops": "0.16.0"}


def sklearn_lock_bytes(python_version: str = "3.14.0") -> bytes:
    return synthetic_lock(python_version, SKLEARN_PACKAGES).to_bytes()


LIBRARY_PACKAGES = {
    "sklearn": SKLEARN_PACKAGES,
    "xgboost": {"pydantic": "2.14.1", "scikit-learn": "1.9.1", "xgboost": "3.4.1"},
    "catboost": {"pydantic": "2.14.1", "catboost": "1.2.10"},
    "onnx": {"pydantic": "2.14.1", "onnx": "1.23.0", "onnxruntime": "1.30.0"},
}


def library_lock_bytes(library: str, python_version: str = "3.14.0") -> bytes:
    return synthetic_lock(python_version, LIBRARY_PACKAGES[library]).to_bytes()
