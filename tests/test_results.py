"""Consumer access to metadata, predictions and the original native object."""

from dataclasses import FrozenInstanceError

import pytest

from lock_helpers import sklearn_lock_bytes
from mlbundle import (
    EnvironmentLock,
    FeatureOrigin,
    Library,
    LoadedBundle,
    Manifest,
    MemberName,
    MetadataView,
    ModelKind,
)
from mlbundle.canonical import canonical_dumps

LOCK = EnvironmentLock.from_bytes(sklearn_lock_bytes())


class Predictor:
    def predict(self, data):
        return [float(row[0]) for row in data]

    def predict_proba(self, data):
        return [[0.1, 0.9] for _ in data]


def test_metadata_view(manifest_data):
    view = MetadataView(
        Manifest.from_bytes(canonical_dumps(manifest_data)),
        (MemberName.MANIFEST, MemberName.SKOPS),
        LOCK,
    )
    assert view.environment_lock.version("skops") == "0.16.0"
    assert not hasattr(view, "native_model")
    with pytest.raises(FrozenInstanceError):
        view.members = ()


def test_loaded_classifier(manifest_data):
    native = object()
    bundle = LoadedBundle(
        Manifest.from_bytes(canonical_dumps(manifest_data)),
        (MemberName.MANIFEST, MemberName.SKOPS),
        LOCK,
        native,
        Predictor(),
    )
    assert bundle.native_model is native
    assert bundle.kind == ModelKind.CLASSIFIER
    assert bundle.library == Library.SKLEARN
    assert bundle.class_name == "DecisionTreeClassifier"
    assert bundle.feature_names == ("a", "b")
    assert bundle.feature_origin == FeatureOrigin.MODEL
    assert bundle.class_labels == [[1, "positive"]]
    bundle.class_labels[0].append("mutated copy")
    assert bundle.class_labels == [[1, "positive"]]
    assert bundle.predict_proba([[1, 2]]) == [[0.1, 0.9]]
    with pytest.raises(ValueError, match="regressor"):
        bundle.predict([[1]])


def test_loaded_regressor(manifest_data):
    manifest_data["model"].update(kind="regressor", n_classes=None)
    manifest_data["target"]["class_labels"] = None
    bundle = LoadedBundle(
        Manifest.from_bytes(canonical_dumps(manifest_data)),
        (),
        LOCK,
        object(),
        Predictor(),
    )
    assert bundle.class_labels is None
    assert bundle.predict([[2]]) == [2.0]
    with pytest.raises(ValueError, match="classifier"):
        bundle.predict_proba([[2]])
