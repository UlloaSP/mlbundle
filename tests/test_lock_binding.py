"""The lock inside bundles: manifest binding, container bounds and export fidelity."""

import pytest
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from lock_helpers import pure_wheel, sklearn_lock_bytes, synthetic_lock
from mlbundle import (
    InvalidContainerError,
    InvalidEnvironmentLockError,
    LimitExceededError,
    LockedPackage,
    MemberName,
)
from mlbundle.container import read_container
from mlbundle.lock import runtime_requirements
from mlbundle.writing import write_container


def bundle_parts(manifest_data, lock):
    from conftest import schema_for_manifest
    from mlbundle.canonical import canonical_dumps

    manifest_data.pop("integrity")
    members = {
        MemberName.SKOPS: b"native model bytes",
        MemberName.SCHEMA: canonical_dumps(schema_for_manifest(manifest_data)),
    }
    if lock is not None:
        members[MemberName.LOCK] = lock
    return members


@pytest.mark.parametrize(
    "packages,python,implementation",
    [
        ({"pydantic": "2.14.1", "scikit-learn": "1.9.1"}, "3.14.0", "CPython"),
        ({"pydantic": "2.14.1", "skops": "0.16.0"}, "3.14.0", "CPython"),
        ({"scikit-learn": "1.9.1", "skops": "0.16.0"}, "3.14.0", "CPython"),
        (None, "3.14.1", "CPython"),
        (None, "3.14.0", "PyPy"),
    ],
)
def test_lock_binds_to_manifest(manifest_data, packages, python, implementation):
    manifest_data["environment"]["python_implementation"] = implementation
    lock = (
        sklearn_lock_bytes(python)
        if packages is None
        else synthetic_lock(python, packages).to_bytes()
    )
    with pytest.raises(InvalidEnvironmentLockError):
        write_container(manifest_data, bundle_parts(manifest_data, lock))


def test_xgboost_cpu_alternative_and_platform_coverage(manifest_data):
    manifest_data["model"].update(library="xgboost", serialization="ubjson")
    base = {"pydantic": "2.14.1", "scikit-learn": "1.9.1"}
    for name in ("xgboost", "xgboost-cpu"):
        members = bundle_parts(dict(manifest_data, integrity={}), None)
        members[MemberName.UBJSON] = members.pop(MemberName.SKOPS)
        members[MemberName.LOCK] = synthetic_lock(
            "3.14.0", base | {name: "3.4.1"}
        ).to_bytes()
        assert write_container(manifest_data, members)
    both = synthetic_lock("3.14.0", base | {"xgboost": "3.4.1", "xgboost-cpu": "3.4.1"})
    members[MemberName.LOCK] = both.to_bytes()
    with pytest.raises(InvalidEnvironmentLockError):
        write_container(manifest_data, members)
    partial = synthetic_lock("3.14.0", base).model_copy(
        update={
            "packages": (
                *synthetic_lock("3.14.0", base).packages,
                LockedPackage(
                    name="xgboost",
                    version="3.4.1",
                    platforms=("linux-x86_64", "linux-aarch64"),
                    wheels=(pure_wheel("xgboost", "3.4.1"),),
                ),
            )
        }
    )
    members[MemberName.LOCK] = partial.to_bytes()
    with pytest.raises(InvalidEnvironmentLockError):
        write_container(manifest_data, members)


def test_container_requires_and_bounds_the_lock(parts):
    from conftest import build_zip

    with pytest.raises(InvalidContainerError, match="predates the required pylock"):
        read_container(build_zip(parts[:3]))
    with pytest.raises(LimitExceededError):
        read_container(
            build_zip(parts), limits=mlbundle.ArchiveLimits(max_lock_bytes=64)
        )
    tampered = [*parts[:3], ("pylock.toml", parts[3][1].replace(b"1024", b"1025", 1))]
    with pytest.raises(mlbundle.IntegrityMismatchError):
        read_container(build_zip(tampered))
    assert read_container(build_zip(parts))[1][MemberName.LOCK] == parts[3][1]


def test_export_with_explicit_lock_and_fidelity(tmp_path):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    generated = mlbundle.export(model, tmp_path / "a.mlbundle").environment_lock
    for source in (generated.to_bytes(), tmp_path / "pylock.toml"):
        (tmp_path / "pylock.toml").write_bytes(generated.to_bytes())
        view = mlbundle.export(
            model, tmp_path / "b.mlbundle", environment_lock=source, overwrite=True
        )
        assert view.environment_lock == generated
        assert mlbundle.load(tmp_path / "b.mlbundle").environment_lock == generated
    stale = generated.model_copy(
        update={
            "packages": tuple(
                p.model_copy(
                    update={"version": "1.0", "wheels": (pure_wheel(p.name, "1.0"),)}
                )
                if p.name == "scikit-learn"
                else p
                for p in generated.packages
            )
        }
    )
    with pytest.raises(InvalidEnvironmentLockError, match=r"scikit-learn locked 1.0"):
        mlbundle.export(
            model, tmp_path / "c.mlbundle", environment_lock=stale.to_bytes()
        )
    other = generated.model_copy(update={"python_version": "3.14.99"})
    with pytest.raises(InvalidEnvironmentLockError, match="Python"):
        mlbundle.export(
            model, tmp_path / "c.mlbundle", environment_lock=other.to_bytes()
        )
    with pytest.raises(LimitExceededError):
        mlbundle.export(
            model,
            tmp_path / "c.mlbundle",
            environment_lock=generated.to_bytes(),
            limits=mlbundle.ArchiveLimits(max_lock_bytes=16),
        )
    assert not (tmp_path / "c.mlbundle").exists()


def _relocked(path, lock, change=lambda text: text):
    """Rebuild a bundle around other lock bytes, as an older writer could have."""
    from conftest import build_zip
    from mlbundle._container_rules import MEMBER_ORDER
    from mlbundle.canonical import canonical_dumps
    from mlbundle.integrity import hash_member

    manifest, members = read_container(path)
    members[MemberName.LOCK] = lock
    value = manifest.model_dump(mode="json", by_alias=True)
    value["integrity"]["pylock.toml"] = hash_member(lock).model_dump()
    members[MemberName.MANIFEST] = change(canonical_dumps(value))
    return build_zip(
        [(name, members[name]) for name in MEMBER_ORDER if name in members]
    )


@pytest.mark.parametrize("kind", ["classifier", "composite"])
def test_by_name_selection_locks_pandas(kind, tmp_path):
    from importlib import metadata

    from record_helpers import named_pipeline

    model, _ = named_pipeline(kind)
    path = tmp_path / "named.mlbundle"
    lock = mlbundle.export(model, path).environment_lock
    assert lock.version("pandas") == metadata.version("pandas")
    assert mlbundle.load(path).environment_lock == lock
    lacking = synthetic_lock(
        lock.python_version,
        {p.name: p.version for p in lock.packages if p.name != "pandas"},
    ).to_bytes()
    with pytest.raises(InvalidEnvironmentLockError, match="exactly one of pandas"):
        mlbundle.export(model, tmp_path / "x.mlbundle", environment_lock=lacking)
    assert not (tmp_path / "x.mlbundle").exists()
    stale = _relocked(path, lacking)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidEnvironmentLockError, match="exactly one of pandas"):
            reader(stale)


def test_hidden_name_selection_fails_verification(tmp_path):
    from record_helpers import named_pipeline

    model, _ = named_pipeline("classifier")
    path = tmp_path / "named.mlbundle"
    lock = mlbundle.export(model, path).environment_lock
    lacking = synthetic_lock(
        lock.python_version,
        {p.name: p.version for p in lock.packages if p.name != "pandas"},
    ).to_bytes()
    # The derivation reads hyperparameters, a claim that load recomputes.
    hidden = _relocked(path, lacking, lambda text: text.replace(b'["a"]', b"[0]"))
    assert mlbundle.inspect(hidden).environment_lock.version("pandas") is None
    with pytest.raises(mlbundle.ClaimMismatchError, match="hyperparameters"):
        mlbundle.load(hidden)


def test_other_models_lock_no_pandas(tmp_path):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline

    from pipeline_helpers import pipeline

    frame = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    positional = make_pipeline(
        ColumnTransformer([("select", "passthrough", [0])]), DecisionTreeClassifier()
    ).fit(frame, [0, 1])
    for name, model in [
        ("positional", positional),
        ("tree", DecisionTreeClassifier().fit(frame, [0, 1])),
        ("catboost", pipeline("catboost")[0]),
    ]:
        view = mlbundle.export(model, tmp_path / f"{name}.mlbundle")
        groups = runtime_requirements(view.manifest.model)
        assert ("pandas",) not in groups
        assert view.environment_lock.version("pandas") is None
    assert [group[0] for group in groups] == [
        "pydantic",
        "scikit-learn",
        "skops",
        "catboost",
    ]
