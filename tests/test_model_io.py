"""Library-independent pipelines, dependency failures and export-time verification."""

import pytest
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle import (
    ArchiveLimits,
    ClaimMismatchError,
    InvalidManifestError,
    Library,
    MemberName,
    MissingOptionalDependencyError,
    UnsupportedModelError,
    registry,
)
from mlbundle._sklearn import Adapter
from mlbundle.canonical import canonical_dumps
from mlbundle.container import read_container
from mlbundle.writing import write_container


def test_basic_export_errors_and_options(tmp_path, monkeypatch):
    target = tmp_path / "tree.mlbundle"
    for model, match in [
        (object(), "Unsupported"),
        (DecisionTreeClassifier(), "fitted"),
    ]:
        with pytest.raises(UnsupportedModelError, match=match):
            mlbundle.export(model, target)
    model = DecisionTreeClassifier().fit([[0, 1], [1, 0]], [0, 1])
    with pytest.raises(mlbundle.LimitExceededError):
        mlbundle.export(model, target, limits=ArchiveLimits(max_member_bytes=1))
    assert not target.exists()
    for names in [["only-one"], ["same", "same"]]:
        with pytest.raises(InvalidManifestError):
            mlbundle.export(model, target, feature_names=names)
    mlbundle.export(model, target, feature_names=["a", "b"])
    with pytest.raises(FileExistsError):
        mlbundle.export(model, target)
    mlbundle.export(model, target, overwrite=True)
    assert mlbundle.load(target).feature_origin == "generated"

    def missing(path):
        raise ModuleNotFoundError(name="skops" if path == "mlbundle._sklearn" else path)

    monkeypatch.setattr(registry, "import_module", missing)
    for action in [
        lambda: mlbundle.export(model, target),
        lambda: mlbundle.load(target),
    ]:
        with pytest.raises(
            MissingOptionalDependencyError, match='pip install "mlbundle\\[sklearn\\]"'
        ):
            action()


@pytest.mark.parametrize("structure", [False, True])
def test_export_roundtrip_rejects_loss(tmp_path, monkeypatch, structure):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    original = Adapter.deserialize

    def changed(self, data):
        restored = original(self, data)
        if structure:
            restored.tree_.threshold[0] += 1
        else:
            restored.max_depth = 99
        return restored

    monkeypatch.setattr(Adapter, "deserialize", changed)
    with pytest.raises(
        ClaimMismatchError, match="structure" if structure else "hyperparameters"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("library", list(Library))
def test_closed_adapter_pipeline_is_library_agnostic(
    library, tmp_path, monkeypatch, manifest_data
):
    from types import SimpleNamespace

    from mlbundle import Manifest, OnnxInfo
    from mlbundle.contract import MODEL_MEMBERS

    claims = Manifest.from_bytes(canonical_dumps(manifest_data)).verified_claims()
    graph = (
        OnnxInfo(
            opsets=[mlbundle.Opset(domain="", version=21)],
            ir_version=10,
            producer_name="test",
            producer_version="1",
        )
        if library == Library.ONNX
        else None
    )
    claims = claims.model_copy(update={"onnx": graph})

    class FakeAdapter:
        model_member = MODEL_MEMBERS[library][1]

        def recognizes(self, model):
            return model == "native input"

        def serialize(self, model):
            return b"native format bytes"

        def deserialize(self, data):
            assert data == b"native format bytes"
            return "native input"

        def extract_claims(self, model):
            return claims

        def extract_structure(self, model):
            return None

        def predictor(self, model):
            return SimpleNamespace(predict_proba=lambda data: [[0.2, 0.8]])

    FakeAdapter.library = library

    def resolve(path):
        if path != f"mlbundle._{library.value}":
            raise ModuleNotFoundError(name=path)
        return SimpleNamespace(Adapter=FakeAdapter)

    monkeypatch.setattr(registry, "import_module", resolve)
    path = tmp_path / "bundle.mlbundle"
    mlbundle.export("native input", path)
    loaded = mlbundle.load(path)
    assert loaded.library == library and loaded.native_model == "native input"
    assert loaded.manifest.environment.onnx == graph
    assert loaded.predict_proba([[1, 2]]) == [[0.2, 0.8]]
    if graph is not None:
        manifest, members = read_container(path)
        parts = manifest.model_dump(mode="json", by_alias=True)
        parts["environment"]["onnx"]["ir_version"] = 11
        del members[MemberName.MANIFEST]
        with pytest.raises(ClaimMismatchError, match="onnx"):
            mlbundle.load(write_container(parts, members))
