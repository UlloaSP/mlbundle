"""CatBoost class calls preserve native thresholds, labels and bounded errors."""

import catboost as cb
import numpy as np
import pytest

import mlbundle
from mlbundle._catboost import Adapter
from record_helpers import threshold_classifier


@pytest.mark.parametrize(
    "labels", [["cold", "warm"], [10, 20], [0.5, 1.5], ["a", "b", "c"]]
)
def test_native_decisions(labels, tmp_path):
    rows = [[index] for index in range(9)]
    targets = [labels[index % len(labels)] for index in range(9)]
    model = cb.CatBoostClassifier(
        iterations=2, depth=1, verbose=False, thread_count=1, allow_writing_files=False
    ).fit(rows, targets)
    path = tmp_path / "classes.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    raw = np.asarray(model.predict(rows)).reshape(-1)
    assert loaded.predictor.predict_class_indices(rows) == [
        [model.classes_.tolist().index(value)] for value in raw
    ]
    assert [
        loaded.predict_record({"0": row[0]})[0].predicted_label for row in rows
    ] == [str(value) for value in raw]


def test_threshold_is_native_after_roundtrip(tmp_path):
    model = threshold_classifier()
    path = tmp_path / "threshold.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    result = loaded.predict_record({"0": 1})[0]
    assert loaded.native_model.get_probability_threshold() == 0.8
    assert 0.5 < result.probabilities[1] < 0.8
    assert result.predicted_label == str(model.predict([[1]])[0]) == "negative"
    assert loaded.predictor.predict_class_indices([[0], [1]]) == [[0], [0]]


@pytest.mark.parametrize(
    "raw",
    [
        [],
        ["private"],
        [["negative", "positive"]],
        [None],
        [np.nan],
        [np.inf],
        [object()],
        ["x" * 1025],
    ],
)
def test_invalid_decision_output(raw, monkeypatch):
    model = threshold_classifier()
    predictor = Adapter().predictor(model)
    monkeypatch.setattr(model, "predict", lambda *a, **kw: raw)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        predictor.predict_class_indices([[1]])
    assert len(str(error.value)) < 160 and "private" not in str(error.value)


def test_class_call_and_failure_boundaries(monkeypatch):
    model = threshold_classifier()
    predictor = Adapter().predictor(model)
    original = model.predict
    seen = []

    def predict(data, **kwargs):
        seen.append(kwargs)
        return original(data, **kwargs)

    monkeypatch.setattr(model, "predict", predict)
    monkeypatch.setattr(model, "predict_proba", lambda *a: pytest.fail("argmax"))
    assert predictor.predict_class_indices([[1]]) == [[0]]
    assert seen == [{"prediction_type": "Class", "thread_count": 1}]

    def fail(*a, **kw):
        raise ValueError("private native decision")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(
        mlbundle.InvalidModelError, match=r"^CatBoost prediction failed\.$"
    ):
        predictor.predict_class_indices([[1]])
    regressor = cb.CatBoostRegressor(
        iterations=1, verbose=False, allow_writing_files=False
    ).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="classifier"):
        Adapter().predictor(regressor).predict_class_indices([[1]])
