"""The pylock.toml contract: canonical round trips, bundle binding and rejections."""

import hashlib
import tomllib

import pytest
from pydantic import ValidationError

from lock_helpers import sklearn_lock_bytes, synthetic_lock
from mlbundle import (
    EnvironmentLock,
    InvalidEnvironmentLockError,
    LockedPackage,
    LockedWheel,
)

URL = "https://files.pythonhosted.org/packages/ab/cd/"
PLATFORMS = ("linux-x86_64", "linux-aarch64", "win-amd64")
LINUX = ("linux-x86_64", "linux-aarch64")


def wheel(filename):
    return LockedWheel(
        name=filename,
        url=URL + filename,
        size=10,
        sha256=hashlib.sha256(filename.encode()).hexdigest(),
    )


def numpy_package(*filenames, platforms=PLATFORMS):
    return LockedPackage(
        name="numpy",
        version="2.5.4",
        platforms=platforms,
        wheels=tuple(wheel(name) for name in sorted(filenames)),
    )


X86 = "numpy-2.5.4-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
ARM = "numpy-2.5.4-cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
WIN = "numpy-2.5.4-cp314-cp314-win_amd64.whl"


def test_canonical_round_trip_identity_and_consumers():
    lock = synthetic_lock("3.14.7", {"pydantic": "2.14.1", "scikit-learn": "1.9.1"})
    lock = lock.model_copy(
        update={
            "packages": (
                *lock.packages,
                LockedPackage(
                    name="zeta-nccl",
                    version="2.0",
                    platforms=LINUX,
                    wheels=(
                        wheel("zeta_nccl-2.0-py3-none-manylinux_2_27_aarch64.whl"),
                        wheel("zeta_nccl-2.0-py3-none-manylinux_2_27_x86_64.whl"),
                    ),
                ),
                LockedPackage(
                    name="zeta-win",
                    version="1.0",
                    platforms=("win-amd64",),
                    wheels=(wheel("zeta_win-1.0-py3-none-any.whl"),),
                ),
            )
        }
    )
    data = lock.to_bytes()
    assert EnvironmentLock.from_bytes(data) == lock
    # Any TOML parser reads PEP 751 fields from the canonical spelling.
    document = tomllib.loads(data.decode())
    assert document["lock-version"] == "1.0"
    assert document["requires-python"] == "==3.14.7"
    assert document["environments"][-1] == (
        "sys_platform == 'win32' and platform_machine == 'AMD64'"
    )
    assert document["packages"][-2]["marker"] == (
        "(sys_platform == 'linux' and platform_machine == 'x86_64') or "
        "(sys_platform == 'linux' and platform_machine == 'aarch64')"
    )
    assert document["packages"][-1]["marker"] == document["environments"][-1]
    assert lock.version("scikit-learn") == "1.9.1" and lock.version("x") is None
    assert lock.platforms == PLATFORMS and lock.glibc == (2, 28)
    moved = lock.model_copy(
        update={
            "packages": tuple(
                p.model_copy(
                    update={
                        "wheels": tuple(
                            w.model_copy(
                                update={"url": "https://mirror.example/" + w.name}
                            )
                            for w in p.wheels
                        )
                    }
                )
                for p in lock.packages
            )
        }
    )
    assert moved.environment_id == lock.environment_id  # URLs are not identity.
    changed = synthetic_lock("3.14.8", {"pydantic": "2.14.1", "scikit-learn": "1.9.1"})
    assert (
        changed.environment_id
        != synthetic_lock(
            "3.14.7", {"pydantic": "2.14.1", "scikit-learn": "1.9.1"}
        ).environment_id
    )


def test_preferred_wheel_per_target():
    abi3 = "wcwidth-0.9.2-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
    windows = "wcwidth-0.9.2-cp310-abi3-win_amd64.whl"
    pure = "wcwidth-0.9.2-py3-none-any.whl"
    package = LockedPackage(
        name="wcwidth",
        version="0.9.2",
        platforms=PLATFORMS,
        wheels=(wheel(abi3), wheel(windows), wheel(pure)),
    )
    # Each target prefers its platform wheel; aarch64 uses the pure one.
    EnvironmentLock(python_version="3.14.7", packages=(package,))
    lock = EnvironmentLock(
        python_version="3.14.7", packages=(numpy_package(X86, ARM, WIN),)
    )
    assert len(lock.packages[0].wheels) == 3


@pytest.mark.parametrize(
    "files,platforms",
    [
        ((X86, ARM), PLATFORMS),  # Windows has no wheel.
        ((X86, ARM, WIN.replace("win_amd64", "win_arm64")), PLATFORMS),
        ((X86, WIN), PLATFORMS),  # aarch64 has no wheel.
        ((X86.replace("cp314-cp314", "cp314-cp314t"), ARM, WIN), PLATFORMS),
        ((WIN.replace("cp314-cp314", "cp314-cp314t"), X86, ARM), PLATFORMS),
        (("numpy-2.5.4-cp314-cp314-manylinux_2_31_x86_64.whl", ARM, WIN), PLATFORMS),
        ((X86, "numpy-2.5.4-cp314-cp314-musllinux_1_2_x86_64.whl"), ("linux-x86_64",)),
        ((X86, ARM), ("linux-x86_64",)),  # An unused wheel.
        ((X86.replace("cp314", "cp313"), ARM, WIN), PLATFORMS),
        ((WIN,), ("x86_64",)),
        (
            (X86, X86.replace("_x86_64.many", "_x86_64.linux_x86_64.many")),
            ("linux-x86_64",),
        ),
        ((X86, ARM), ("linux-aarch64", "linux-x86_64")),
    ],
)
def test_unresolvable_targets(files, platforms):
    with pytest.raises(ValidationError):
        EnvironmentLock(
            python_version="3.14.7",
            packages=(numpy_package(*files, platforms=platforms),),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Numpy"},
        {"name": "numpy_core"},
        {"version": "2.5.4+local"},
        {"version": "v2.5.4"},
        {"platforms": ()},
    ],
)
def test_invalid_package_records(change):
    fields = {
        "name": "numpy",
        "version": "2.5.4",
        "platforms": PLATFORMS,
        "wheels": (wheel(X86), wheel(ARM), wheel(WIN)),
    } | change
    with pytest.raises(ValidationError):
        LockedPackage(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"url": "http://files.example/" + ARM},
        {"url": URL + "other.whl"},
        {"url": URL + 'a"b/' + ARM},
        {"name": "numpy-2.5.4.tar.gz", "url": URL + "numpy-2.5.4.tar.gz"},
        {"sha256": "A" * 64},
        {"size": 0},
    ],
)
def test_invalid_wheel_records(fields):
    with pytest.raises(ValidationError):
        LockedWheel(
            **({"name": ARM, "url": URL + ARM, "size": 1, "sha256": "a" * 64} | fields)
        )


@pytest.mark.parametrize(
    "packages,python",
    [
        ({"mlbundle": "0.3.0"}, "3.14.7"),  # writer_version pins mlbundle.
        ({"pydantic": "2.14.1"}, "3.14"),
        ({"pydantic": "2.14.1"}, "2.7.18"),
    ],
)
def test_invalid_locks(packages, python):
    with pytest.raises(ValidationError):
        synthetic_lock(python, packages)


def test_empty_lock():
    with pytest.raises(ValidationError):
        EnvironmentLock(python_version="3.14.7", packages=())


def mutate(old, new):
    return sklearn_lock_bytes().replace(old, new, 1)


@pytest.mark.parametrize(
    "data",
    [
        mutate(b'created-by = "mlbundle"', b'created-by = "uv"'),
        mutate(b'lock-version = "1.0"', b'lock-version = "1.1"'),
        mutate(b'"==3.14.0"', b'">=3.14.0"'),
        mutate(b"[[packages]]", b"[tool.uv]\n\n[[packages]]"),
        mutate(b"[[packages]]\n", b'[[packages]]\nindex = "https://pypi.org/simple"\n'),
        mutate(b"size = 1024", b"size = true"),
        mutate(b"size = 1024", b'size = "1024"'),
        mutate(b"hashes = { sha256", b'hashes = { md5 = "x", sha256'),
        mutate(b"wheels = [", b"marker = \"os_name == 'nt'\"\nwheels = ["),
        mutate(b"wheels = [", b"marker = \"sys_platform == 'win32'\"\nwheels = ["),
        mutate(b'name = "pydantic"', b'name = "pydantic"\nsdist = { url = "x" }'),
        mutate(b'name = "skops"', b'name = "aaa"'),  # Unsorted, wheel mismatch.
        mutate(b"\n", b"\n# comment\n"),  # Valid TOML, not canonical.
        mutate(b"size = 1024", b"size=1024"),
        sklearn_lock_bytes() + b"\n",
        sklearn_lock_bytes().replace(b"\n", b"\r\n"),
        b"\xff\xfe",
        b"packages = " + b"[" * 2000 + b"]" * 2000,
        b"lock-version = ",
        b"",
    ],
)
def test_parser_rejections(data):
    with pytest.raises(InvalidEnvironmentLockError) as error:
        EnvironmentLock.from_bytes(data)
    assert str(error.value) in (
        "Invalid environment lock.",
        "Environment lock is not canonical.",
    )
