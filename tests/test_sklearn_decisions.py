"""Native sklearn class decisions, including calibrated SVC disagreement."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mlbundle
from mlbundle._sklearn import Adapter
from record_helpers import disagreement_classifier


@pytest.mark.parametrize("outputs", [1, 2])
def test_native_decisions_and_output_order(outputs, monkeypatch):
    rows = [[0], [1], [2], [3]]
    labels = (
        ["cold", "cold", "warm", "warm"]
        if outputs == 1
        else [[0, 2], [0, 3], [1, 2], [1, 3]]
    )
    model = DecisionTreeClassifier().fit(rows, labels)
    predictor = Adapter().predictor(model)
    raw = np.asarray(model.predict(rows)).reshape(len(rows), outputs)
    classes = [model.classes_] if outputs == 1 else model.classes_
    expected = [
        [
            values.tolist().index(value)
            for values, value in zip(classes, row, strict=True)
        ]
        for row in raw
    ]
    monkeypatch.setattr(
        model, "predict_proba", lambda *a: pytest.fail("probability decision")
    )
    assert predictor.predict_class_indices(rows) == expected
    with pytest.raises(ValueError, match="classifier"):
        Adapter().predictor(
            DecisionTreeRegressor().fit(rows, [0, 0, 1, 1])
        ).predict_class_indices(rows)


def test_svc_uses_predict_with_disagreeing_probabilities(tmp_path):
    model = disagreement_classifier()
    path = tmp_path / "svc.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    row = [[3.5]]
    native = int(model.predict(row)[0])
    assert loaded.predictor.predict_class_indices(row) == [[native]]
    result = loaded.predict_record({"0": 3.5})[0]
    assert result.predicted_label == str(native)
    assert result.probabilities.index(max(result.probabilities)) != native


@pytest.mark.parametrize(
    "raw",
    [
        [],
        ["cold"],
        [["cold", "warm"]],
        ["private", "warm"],
        [None, "warm"],
        [np.nan, "warm"],
        [complex(1), "warm"],
        [{"private": 1}, "warm"],
        ["x" * 1025, "warm"],
    ],
)
def test_invalid_decision_output_is_bounded(raw, monkeypatch):
    model = DecisionTreeClassifier().fit([[0], [1]], ["cold", "warm"])
    predictor = Adapter().predictor(model)
    monkeypatch.setattr(model, "predict", lambda _: raw)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        predictor.predict_class_indices([[0], [1]])
    assert "private" not in str(error.value) and len(str(error.value)) < 160


def test_native_failure_is_bounded(monkeypatch):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    predictor = Adapter().predictor(model)

    def fail(_):
        raise ValueError("private decision")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(
        mlbundle.InvalidModelError, match=r"^Sklearn prediction failed\.$"
    ):
        predictor.predict_class_indices([[0]])
