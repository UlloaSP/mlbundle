"""Version hints diagnose existing rejection paths without adding a load gate."""

import importlib.metadata

import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import mlbundle
from lock_helpers import pure_wheel, sklearn_lock_bytes
from mlbundle import _load_diagnostics
from mlbundle._sklearn import Adapter, transformers
from mlbundle.canonical import canonical_dumps
from mlbundle.container import read_container
from mlbundle.writing import write_container


def failed_load(_self, _data):
    raise mlbundle.InvalidModelError(
        "Skops state does not survive native construction."
    )


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "version.mlbundle"
    mlbundle.export(DecisionTreeClassifier().fit([[0], [1]], [0, 1]), path)
    return path


def declared_package(path, declared):
    """Rewrite the locked scikit-learn version, the declaration diagnostics read."""
    manifest, members = read_container(path)
    lock = mlbundle.EnvironmentLock.from_bytes(members[mlbundle.MemberName.LOCK])
    packages = tuple(
        package.model_copy(
            update={
                "version": declared,
                "wheels": (pure_wheel(package.name, declared),),
            }
        )
        if package.name == "scikit-learn"
        else package
        for package in lock.packages
    )
    members[mlbundle.MemberName.LOCK] = lock.model_copy(
        update={"packages": packages}
    ).to_bytes()
    del members[mlbundle.MemberName.MANIFEST]
    return write_container(manifest, members)


def test_environment_mismatch_after_failure(path, monkeypatch):
    data = declared_package(path, "1.9.1")
    monkeypatch.setattr(_load_diagnostics, "version", lambda _: "1.9.0")
    assert mlbundle.load(data).predict_proba([[0]]) == [[1.0, 0.0]]
    monkeypatch.setattr(Adapter, "deserialize", failed_load)
    with pytest.raises(mlbundle.EnvironmentMismatchError) as error:
        mlbundle.load(data)
    assert isinstance(error.value, mlbundle.InvalidModelError)
    assert str(error.value) == (
        'Bundle declares scikit-learn "1.9.1"; installed version is "1.9.0". '
        'The bundle must be re-exported with scikit-learn "1.9.0" or loaded '
        'where scikit-learn "1.9.1" is installed.'
    )


def test_equal_environment_keeps_original(path, monkeypatch):
    data = declared_package(path, "1.9.0")
    monkeypatch.setattr(_load_diagnostics, "version", lambda _: "1.9.0")
    monkeypatch.setattr(Adapter, "deserialize", failed_load)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        mlbundle.load(data)
    assert type(error.value) is mlbundle.InvalidModelError
    assert str(error.value) == "Skops state does not survive native construction."


@pytest.mark.parametrize(
    "declared", ["private\\n", "1.9.1 ", "9" * 65, "v1.9.1", "1.9.1;evil", "1.9+x"]
)
def test_unsafe_declared_versions_never_reach_diagnostics(declared):
    # The lock rejects such versions before any reader can display them.
    data = sklearn_lock_bytes().replace(b'"1.9.1"', f'"{declared}"'.encode())
    with pytest.raises(mlbundle.InvalidEnvironmentLockError):
        mlbundle.EnvironmentLock.from_bytes(data)


def test_unavailable_installed_metadata_keeps_original(path, monkeypatch):
    def absent(_):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(_load_diagnostics, "version", absent)
    monkeypatch.setattr(Adapter, "deserialize", failed_load)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        mlbundle.load(declared_package(path, "1.9.1"))
    assert type(error.value) is mlbundle.InvalidModelError


def test_claim_mismatch_is_not_environment_failure(path, monkeypatch):
    data = declared_package(path, "0.0.1")

    def claims(*_):
        raise mlbundle.ClaimMismatchError("n_features")

    monkeypatch.setattr(Adapter, "extract_claims", claims)
    with pytest.raises(mlbundle.ClaimMismatchError):
        mlbundle.load(data)


@pytest.mark.parametrize("sklearn_method", ["deserialize", "claims"])
@pytest.mark.parametrize(
    "native_method", ["deserialize", "extract_claims", "extract_structure"]
)
def test_composite_skops_only_failures(
    tmp_path, monkeypatch, sklearn_method, native_method
):
    path = tmp_path / "composite.mlbundle"
    model = make_pipeline(
        StandardScaler(), XGBClassifier(n_estimators=1, n_jobs=1)
    ).fit([[0], [1]], [0, 1])
    mlbundle.export(model, path)
    data = declared_package(path, "0.0.1")
    assert mlbundle.load(data).predict_record({"0": 0})
    from mlbundle._xgboost import Adapter as XGBAdapter

    with monkeypatch.context() as patch:
        patch.setattr(XGBAdapter, native_method, failed_load)
        with pytest.raises(mlbundle.InvalidModelError) as error:
            mlbundle.load(data)
        assert type(error.value) is mlbundle.InvalidModelError
    monkeypatch.setattr(transformers.Adapter, sklearn_method, failed_load)
    with pytest.raises(mlbundle.EnvironmentMismatchError):
        mlbundle.load(data)


@pytest.mark.parametrize(
    "writer", ["0.1.0a1", "0.1.0a2", "1.0rc1", "1.0.post2.dev3+local"]
)
def test_old_manifest_diagnosis_shared_by_inspect_and_load(manifest_data, writer):
    from conftest import build_zip, schema_for_manifest

    manifest_data["writer_version"] = writer
    manifest_data["target"]["name"] = manifest_data["target"].pop("names")
    data = build_zip(
        [
            ("manifest.json", canonical_dumps(manifest_data)),
            ("model.skops", b"native model bytes"),
            ("schema.json", canonical_dumps(schema_for_manifest(manifest_data))),
            ("pylock.toml", sklearn_lock_bytes()),
        ]
    )
    expected = (
        f'Invalid mlbundle manifest. Bundle was written by mlbundle "{writer}"; '
        f'this reader is "{mlbundle.__version__}". Format 1 is not frozen; '
        "the model must be re-exported."
    )
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(mlbundle.InvalidManifestError) as error:
            reader(data)
        assert str(error.value) == expected


@pytest.mark.parametrize(
    "writer", [None, 1, [], "", "evil\n\x1b", "9" * 65, "1.0 ", mlbundle.__version__]
)
def test_manifest_without_safe_different_writer_keeps_message(manifest_data, writer):
    manifest_data["unexpected"] = True
    manifest_data["writer_version"] = writer
    with pytest.raises(mlbundle.InvalidManifestError) as error:
        mlbundle.Manifest.from_bytes(canonical_dumps(manifest_data))
    assert str(error.value) == "Invalid mlbundle manifest."


def test_malformed_json_and_unknown_format_keep_errors():
    with pytest.raises(
        mlbundle.InvalidManifestError, match=r"^Invalid mlbundle manifest\.$"
    ):
        mlbundle.Manifest.from_bytes(b"not JSON")
    with pytest.raises(mlbundle.UnsupportedFormatVersionError):
        mlbundle.Manifest.from_bytes(
            b'{"format_name":"mlbundle","format_version":"99","writer_version":"0.1.0a1"}'
        )
