"""Closed lazy resolution using a complete in-test adapter."""

from types import SimpleNamespace

import pytest

from mlbundle import (
    AdapterNotImplementedError,
    Library,
    Manifest,
    MemberName,
    MissingOptionalDependencyError,
    UnsupportedModelError,
    registry,
)
from mlbundle.canonical import canonical_dumps, canonical_loads


class FakePredictor:
    def predict(self, data):
        return [float(row[0]) for row in data]

    def predict_proba(self, data):
        return [[0.25, 0.75] for _ in data]


class FakeAdapter:
    library = Library.SKLEARN
    model_member = MemberName.SKOPS

    def __init__(self, claims):
        self.claims = claims

    def recognizes(self, model):
        return isinstance(model, dict) and "weights" in model

    def serialize(self, model):
        return canonical_dumps(model)

    def deserialize(self, data):
        return canonical_loads(data)

    def extract_claims(self, model):
        return self.claims

    def extract_structure(self, model):
        return {"family": "linear", "coef": model["weights"], "intercept": [0.0]}

    def predictor(self, model):
        return FakePredictor()


def test_fake_adapter_contract(monkeypatch, manifest_data):
    claims = Manifest.from_bytes(canonical_dumps(manifest_data)).verified_claims()
    fake = FakeAdapter(claims)
    paths = []

    def import_module(path):
        paths.append(path)
        return SimpleNamespace(Adapter=lambda: fake)

    monkeypatch.setattr(registry, "import_module", import_module)
    adapter = registry.get_adapter(Library.SKLEARN)
    assert paths == ["mlbundle._sklearn"]
    native = {"weights": [1.0, 2.0]}
    assert adapter.recognizes(native)
    assert not adapter.recognizes(object())
    restored = adapter.deserialize(adapter.serialize(native))
    assert restored == native
    assert adapter.extract_claims(restored) == claims
    assert adapter.extract_structure(restored)["coef"] == [1.0, 2.0]
    assert adapter.predictor(restored).predict([[2]]) == [2.0]
    assert adapter.predictor(restored).predict_proba([[2]]) == [[0.25, 0.75]]


@pytest.mark.parametrize("library", list(Library))
def test_missing_adapter_module(library):
    with pytest.raises(AdapterNotImplementedError, match=library.value):
        registry.get_adapter(library)


@pytest.mark.parametrize(
    "library,dependency",
    [
        (Library.SKLEARN, "skops"),
        (Library.XGBOOST, "sklearn"),
        (Library.ONNX, "onnxruntime"),
    ],
)
def test_missing_optional_dependency(monkeypatch, library, dependency):
    def missing(path):
        raise ModuleNotFoundError(name=dependency)

    monkeypatch.setattr(registry, "import_module", missing)
    with pytest.raises(MissingOptionalDependencyError) as error:
        registry.get_adapter(library)
    assert error.value.extra == library.value


def test_internal_import_failure_is_not_hidden(monkeypatch):
    def missing(path):
        raise ModuleNotFoundError(name="mlbundle._mistake")

    monkeypatch.setattr(registry, "import_module", missing)
    with pytest.raises(ModuleNotFoundError):
        registry.get_adapter(Library.ONNX)


def test_missing_dependency_during_adapter_construction(monkeypatch):
    def construct():
        raise ModuleNotFoundError(name="onnxruntime")

    monkeypatch.setattr(
        registry, "import_module", lambda path: SimpleNamespace(Adapter=construct)
    )
    with pytest.raises(MissingOptionalDependencyError, match="mlbundle\\[onnx\\]"):
        registry.get_adapter(Library.ONNX)


def test_unknown_library_cannot_supply_module_path(monkeypatch):
    def unexpected(path):
        pytest.fail("Invalid input must never reach import_module")

    monkeypatch.setattr(registry, "import_module", unexpected)
    with pytest.raises(UnsupportedModelError):
        registry.get_adapter("attacker.module")


@pytest.mark.parametrize(
    "library,member",
    [(Library.ONNX, MemberName.SKOPS), (Library.SKLEARN, MemberName.ONNX)],
)
def test_mismatched_adapter_identity(monkeypatch, library, member):
    fake = SimpleNamespace(library=library, model_member=member)
    monkeypatch.setattr(
        registry, "import_module", lambda path: SimpleNamespace(Adapter=lambda: fake)
    )
    with pytest.raises(UnsupportedModelError):
        registry.get_adapter(Library.SKLEARN)
