"""Training-side lock generation: installed pins, per-target uv output and its errors."""

import hashlib
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from mlbundle import (
    LockGenerationError,
    MissingOptionalDependencyError,
    _lock_generation,
)
from mlbundle._wheel_tags import PLATFORMS, UV_PLATFORMS, normalize
from mlbundle.contract import MODEL_MEMBERS, Library, ModelKind
from mlbundle.lock import runtime_requirements
from mlbundle.manifest import ModelInfo


@pytest.fixture
def offline_lock():
    """This module exercises the real resolver; the shared fixture replaces it."""


def model_info(library: Library) -> ModelInfo:
    return ModelInfo(
        library=library,
        serialization=MODEL_MEMBERS[library][0],
        estimator_class="Model",
        kind=ModelKind.REGRESSOR,
        hyperparameters={},
        n_features=1,
        n_classes=None,
        n_outputs=1,
    )


class Distribution:
    def __init__(self, name, version, direct=None):
        self.metadata = {"Name": name}
        self.version = version
        self.direct = direct

    def read_text(self, filename):
        return self.direct if filename == "direct_url.json" else None


def test_installed_distributions(monkeypatch):
    found = [
        Distribution("Scikit_Learn", "1.9.1"),
        Distribution("scikit-learn", "1.0"),  # Shadowed by the first on sys.path.
        Distribution("mlbundle", "0.3.0", direct='{"dir_info": {"editable": true}}'),
        Distribution(
            "skops", "0.16.0", direct='{"url": "file:///w.whl", "archive_info": {}}'
        ),
        Distribution(
            "scipy", "1.18.1", direct='{"url": "git+https://x", "vcs_info": {}}'
        ),
        Distribution("numpy", "2.5.4", direct="not json"),
        Distribution("torch", "2.0+cpu"),
        Distribution("--index-url", "1.0"),
        Distribution(None, "1.0"),
    ]
    monkeypatch.setattr(_lock_generation.metadata, "distributions", lambda: found)
    # A wheel installed from a file keeps its version; source trees cannot be pinned.
    assert _lock_generation.installed_distributions() == {
        "scikit-learn": "1.9.1",
        "mlbundle": None,
        "skops": "0.16.0",
        "scipy": None,
        "numpy": None,
        "torch": None,
    }


def test_roots_pick_one_installed_distribution_per_group():
    groups = runtime_requirements(model_info(Library.XGBOOST))
    installed = {"pydantic": "2.14.1", "scikit-learn": "1.9.1", "xgboost-cpu": "3.4.1"}
    assert _lock_generation.roots(groups, installed) == installed
    for broken, message in [
        (installed | {"xgboost": "3.4.1"}, "Exactly one of xgboost, xgboost-cpu"),
        ({"pydantic": "2.14.1", "scikit-learn": "1.9.1"}, "Exactly one of xgboost"),
        (installed | {"pydantic": None}, "pydantic is installed from a source tree"),
    ]:
        with pytest.raises(LockGenerationError, match=message):
            _lock_generation.roots(groups, broken)


def test_runtime_groups_follow_the_package_extras():
    extras: dict[str | None, set[str]] = {}
    for line in metadata.requires("mlbundle") or []:
        name = re.match(r"[A-Za-z0-9_.-]+", line)
        extra = re.search(r"extra == [\"']([a-z]+)", line)
        assert name is not None
        extras.setdefault(extra and extra[1], set()).add(normalize(name[0]))
    for library in Library:
        groups = runtime_requirements(model_info(library))
        assert {group[0] for group in groups} == extras[None] | extras[library.value]


def uv_record(filename, size=10):
    digest = hashlib.sha256(filename.encode()).hexdigest()
    url = f"https://files.pythonhosted.org/packages/aa/{filename}"
    return f'{{ url = "{url}", size = {size}, hashes = {{ sha256 = "{digest}" }} }}'


def uv_output(packages):
    lines = ['lock-version = "1.0"', 'created-by = "uv"', 'requires-python = ">=3.14"']
    for name, version, files in packages:
        wheels = ", ".join(uv_record(file) for file in files)
        lines += ["[[packages]]", f'name = "{name}"', f'version = "{version}"']
        lines.append(f"wheels = [{wheels}]")
    return "\n".join(lines) + "\n"


def fake_uv(monkeypatch, outputs, returncode=0, stderr=""):
    """Write per-target uv output; outputs are keyed by mlbundle target names."""
    calls = []
    targets = {uv: target for target, uv in UV_PLATFORMS.items()}

    def run(command, *, cwd, **_):
        calls.append(command)
        target = targets[command[command.index("--python-platform") + 1]]
        if returncode == 0:
            out = Path(cwd) / command[command.index("--output-file") + 1]
            out.write_text(uv_output(outputs.get(target, [])), encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    monkeypatch.setattr(_lock_generation, "_uv", lambda: "uv")
    monkeypatch.setattr(_lock_generation.subprocess, "run", run)
    return calls


PURE = ("pydantic", "2.14.1", ["pydantic-2.14.1-py3-none-any.whl"])
NUMPY = {
    "linux-x86_64": "numpy-2.5.4-cp314-cp314-manylinux_2_28_x86_64.whl",
    "linux-aarch64": "numpy-2.5.4-cp314-cp314-manylinux_2_28_aarch64.whl",
    "win-amd64": "numpy-2.5.4-cp314-cp314-win_amd64.whl",
}
WCWIDTH = {
    "linux-x86_64": "wcwidth-0.9.2-cp310-abi3-manylinux2014_x86_64.whl",
    "linux-aarch64": "wcwidth-0.9.2-py3-none-any.whl",
    "win-amd64": "wcwidth-0.9.2-cp310-abi3-win_amd64.whl",
}
NCCL = ("nvidia-nccl-cu13", "2.32.3", ["nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_x86_64.whl"])  # fmt: skip


def test_resolve_merges_targets(monkeypatch):
    outputs = {
        target: [
            PURE,
            ("mlbundle", "0.3.0", ["mlbundle-0.3.0-py3-none-any.whl"]),
            ("numpy", "2.5.4", [NUMPY[target], "numpy-2.5.4.tar.gz"]),
            # uv also lists the pure fallback; installers take the platform wheel.
            ("wcwidth", "0.9.2", [WCWIDTH[target], "wcwidth-0.9.2-py3-none-any.whl"]),
        ]
        + ([NCCL] if target == "linux-x86_64" else [])
        for target in PLATFORMS
    }
    calls = fake_uv(monkeypatch, outputs)
    lock = _lock_generation.resolve_lock(
        "3.14.7", {"pydantic": "2.14.1"}, {"numpy": "2.5.4"}
    )
    assert [call[call.index("--python-platform") + 1] for call in calls] == [
        "x86_64-manylinux_2_28",
        "aarch64-manylinux_2_28",
        "x86_64-pc-windows-msvc",
    ]
    assert all("--only-binary" in call and ":all:" in call for call in calls)
    assert [p.name for p in lock.packages] == [
        "numpy",
        "nvidia-nccl-cu13",
        "pydantic",
        "wcwidth",
    ]
    numpy, nccl, _, wcwidth = lock.packages
    assert [w.name for w in numpy.wheels] == sorted(NUMPY.values())
    assert [w.name for w in wcwidth.wheels] == sorted(WCWIDTH.values())
    assert nccl.platforms == ("linux-x86_64",)
    assert [len(p.wheels) for p in lock.packages] == [3, 1, 1, 3]


@pytest.mark.parametrize(
    "outputs,message",
    [
        (
            {"linux-x86_64": [PURE], "win-amd64": [("pydantic", "2.14.0", PURE[2])]},
            "different versions",
        ),
        (
            {"linux-x86_64": [("numpy", "2.5.4", ["numpy-2.5.4.tar.gz"])]},
            "no single preferred CPython wheel",
        ),
        (
            {"win-amd64": [("numpy", "2.5.4", [NUMPY["linux-x86_64"]])]},
            "no single preferred CPython wheel for win-amd64",
        ),
        (
            {"linux-x86_64": [("Bad Name", "1.0", ["bad-1.0-py3-none-any.whl"])]},
            "not lockable",
        ),
    ],
)
def test_resolve_rejects_unlockable_environments(monkeypatch, outputs, message):
    fake_uv(monkeypatch, outputs)
    with pytest.raises(LockGenerationError, match=message):
        _lock_generation.resolve_lock("3.14.7", {"pydantic": "2.14.1"}, {})


def test_uv_failures(monkeypatch):
    fake_uv(monkeypatch, {}, returncode=1, stderr="No solution\x1b[31m found")
    with pytest.raises(LockGenerationError) as error:
        _lock_generation.resolve_lock("3.14.7", {"pydantic": "2.14.1"}, {})
    assert "linux-x86_64" in str(error.value) and "\x1b" not in str(error.value)

    def timeout(command, **_):
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(_lock_generation.subprocess, "run", timeout)
    with pytest.raises(LockGenerationError, match="timed out"):
        _lock_generation.resolve_lock("3.14.7", {"pydantic": "2.14.1"}, {})


def test_missing_uv(monkeypatch):
    monkeypatch.setitem(sys.modules, "uv", None)
    with pytest.raises(MissingOptionalDependencyError) as error:
        _lock_generation.resolve_lock("3.14.7", {"pydantic": "2.14.1"}, {})
    assert error.value.extra == "lock"


@pytest.mark.network
def test_generate_locks_the_installed_runtime_with_uv():
    """Resolve the sklearn runtime of this interpreter through the package index."""
    lock = _lock_generation.generate_lock(
        runtime_requirements(model_info(Library.SKLEARN))
    )
    installed = _lock_generation.installed_distributions()
    assert lock.python_version == platform.python_version()
    for name in ("pydantic", "scikit-learn", "skops", "numpy", "scipy"):
        assert lock.version(name) == installed[name]
    _lock_generation.check_fidelity(lock, installed)
    assert all(
        wheel.url.startswith("https://") for p in lock.packages for wheel in p.wheels
    )
