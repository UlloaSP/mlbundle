"""Composite decisions transform the original row and call the native leaf."""

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mlbundle
from pipeline_helpers import pipeline
from record_helpers import threshold_classifier

pytestmark = [
    pytest.mark.filterwarnings("ignore:X does not have valid feature names"),
    pytest.mark.filterwarnings("ignore:X has feature names"),
]


@pytest.mark.parametrize("library", ["xgboost", "catboost"])
@pytest.mark.parametrize("prefix", [False, True])
def test_pipeline_native_class_decisions(library, prefix, tmp_path):
    model, frame = pipeline(library, False, prefix)
    path = tmp_path / "composite.mlbundle"
    labels = [["cold", "warm"]] if library == "xgboost" else None
    mlbundle.export(model, path, class_labels=labels)
    loaded = mlbundle.load(path)
    expected = model.predict(frame).reshape(-1)
    classes = model.steps[-1][1].classes_.tolist()
    assert loaded.predictor.predict_class_indices(frame) == [
        [classes.index(value)] for value in expected
    ]
    for row, label in zip(frame.to_numpy(), expected, strict=True):
        keys = (
            loaded.feature_names
            if loaded.feature_origin == "model"
            else map(str, range(len(row)))
        )
        result = loaded.predict_record(dict(zip(keys, row, strict=True)))[0]
        assert result.predicted_label == (
            labels[0][int(label)] if labels else str(label)
        )


def test_composite_preserves_catboost_threshold(tmp_path):
    model = make_pipeline(StandardScaler().fit([[0], [1]]), threshold_classifier())
    path = tmp_path / "threshold-pipeline.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    result = loaded.predict_record({"0": 1})[0]
    assert result.predicted_label == str(model.predict([[1]])[0]) == "negative"
    assert result.probabilities[1] > result.probabilities[0]


def test_composite_decision_errors_are_bounded(tmp_path, monkeypatch):
    model, frame = pipeline()
    path = tmp_path / "invalid.mlbundle"
    mlbundle.export(model, path)
    predictor = mlbundle.load(path).predictor
    monkeypatch.setattr(predictor.native.model, "predict", lambda _: np.array([999]))
    with pytest.raises(mlbundle.InvalidModelError) as error:
        predictor.predict_class_indices(frame.iloc[:1])
    assert len(str(error.value)) < 160
    regressor, frame = pipeline(regression=True)
    mlbundle.export(regressor, path, overwrite=True)
    with pytest.raises(ValueError, match="classifier"):
        mlbundle.load(path).predictor.predict_class_indices(frame)
