"""Runtime checks: exactly the lock plus mlbundle, on a GIL CPython target."""

import pytest

import mlbundle
from lock_helpers import sklearn_lock_bytes
from mlbundle import (
    EnvironmentLock,
    EnvironmentMismatchError,
    LockedPackage,
    Manifest,
    MemberName,
    MetadataView,
    _environment_check,
)
from mlbundle._environment_check import RuntimeFacts, environment_problems
from mlbundle.canonical import canonical_dumps
from mlbundle.lock import LockedWheel


def bundle(manifest_data, *extra):
    lock = EnvironmentLock.from_bytes(sklearn_lock_bytes("3.14.7"))
    lock = lock.model_copy(update={"packages": (*lock.packages, *extra)})
    manifest_data["environment"]["python_version"] = "3.14.7"
    return MetadataView(
        Manifest.from_bytes(canonical_dumps(manifest_data)), (MemberName.LOCK,), lock
    )


def facts(view, **changes):
    distributions = {p.name: p.version for p in view.environment_lock.packages}
    distributions["mlbundle"] = view.manifest.writer_version
    values = {
        "implementation": "cpython",
        "python_version": "3.14.7",
        "gil_disabled": False,
        "platform": "linux-x86_64",
        "glibc": (2, 36),
        "distributions": distributions,
    }
    return RuntimeFacts(**(values | changes))


NCCL = LockedPackage(
    name="nvidia-nccl-cu13",
    version="2.32.3",
    platforms=("linux-x86_64",),
    wheels=(
        LockedWheel(
            name="nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_x86_64.whl",
            url="https://files.example.org/"
            "nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_x86_64.whl",
            size=1,
            sha256="0" * 64,
        ),
    ),
)


def test_exact_environment_passes_per_target(manifest_data):
    view = bundle(manifest_data, NCCL)
    assert environment_problems(view, facts(view)) == []
    for platform, glibc in [("linux-aarch64", (2, 28)), ("win-amd64", None)]:
        other = facts(view, platform=platform, glibc=glibc)
        other.distributions.pop("nvidia-nccl-cu13")  # Locked only for x86_64.
        assert environment_problems(view, other) == []


@pytest.mark.parametrize(
    "changes,problem",
    [
        ({"implementation": "pypy"}, "not CPython with the GIL"),
        ({"gil_disabled": True}, "not CPython with the GIL"),
        ({"python_version": "3.14.8"}, 'Python "3.14.8" is not 3.14.7'),
        ({"platform": "win-arm64"}, 'platform "win-arm64" is not a target'),
        ({"platform": "macosx-14.0-arm64"}, "is not a target"),
        ({"platform": "linux-riscv64"}, "is not a target"),
        ({"glibc": None}, "glibc 2.28 or newer"),
        ({"glibc": (2, 27)}, "glibc 2.28 or newer"),
    ],
)
def test_interpreter_and_platform_mismatches(manifest_data, changes, problem):
    view = bundle(manifest_data)
    assert any(
        problem in item for item in environment_problems(view, facts(view, **changes))
    )


def test_distribution_mismatches(manifest_data):
    view = bundle(manifest_data)
    current = facts(view)
    current.distributions.update(
        {"skops": "0.15.0", "pip": "26.0", "scikit-learn": None}
    )
    del current.distributions["pydantic"]
    current.distributions["mlbundle"] = "0.2.1"
    problems = environment_problems(view, current)
    assert problems == [
        f'mlbundle "0.2.1" is installed, {view.manifest.writer_version} is locked',
        '"pip" is installed but not locked',
        "pydantic 2.14.1 is not installed",
        "scikit-learn is duplicated or installed from a source tree",
        'skops "0.15.0" is installed, 0.16.0 is locked',
    ]


def test_check_environment_reports_bounded_problems(manifest_data, monkeypatch):
    view = bundle(manifest_data)
    many = facts(view)
    many.distributions.update({f"extra-{index}": "1.0" for index in range(20)})
    monkeypatch.setattr(RuntimeFacts, "current", classmethod(lambda cls: many))
    with pytest.raises(EnvironmentMismatchError) as error:
        mlbundle.check_environment(view)
    assert str(error.value).endswith("; ...")
    assert str(error.value).count("is installed but not locked") == 10
    monkeypatch.setattr(RuntimeFacts, "current", classmethod(lambda cls: facts(view)))
    mlbundle.check_environment(view)


def test_current_facts_read_this_interpreter(monkeypatch):
    current = RuntimeFacts.current()
    assert current.implementation == "cpython"
    # None for an editable checkout, the version for the installed wheel in CI.
    assert current.distributions["mlbundle"] in (None, mlbundle.__version__)
    monkeypatch.setattr(
        _environment_check.os, "confstr", lambda _: "glibc 2.39", raising=False
    )
    assert RuntimeFacts.current().glibc == (2, 39)

    def unsupported(_):
        raise ValueError

    monkeypatch.setattr(_environment_check.os, "confstr", unsupported, raising=False)
    assert RuntimeFacts.current().glibc is None
