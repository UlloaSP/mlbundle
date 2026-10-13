"""Native encoded XGBoost decisions map through the bound original labels."""

from decimal import Decimal

import numpy as np
import pytest
from xgboost import XGBClassifier, XGBRegressor

import mlbundle
from mlbundle._xgboost import Adapter


@pytest.mark.parametrize("classes", [2, 3])
def test_native_wrapper_decisions_and_bound_labels(classes, tmp_path):
    rows = [[index] for index in range(9)]
    model = XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1).fit(
        rows, [index % classes for index in range(9)]
    )
    names = ["cold", "warm", "hot"][:classes]
    path = tmp_path / "bound.mlbundle"
    mlbundle.export(model, path, class_labels=[names])
    loaded = mlbundle.load(path)
    native = model.predict(rows)
    assert loaded.predictor.predict_class_indices(rows) == [
        [int(index)] for index in native
    ]
    assert [
        loaded.predict_record({"0": row[0]})[0].predicted_label for row in rows
    ] == [names[int(index)] for index in native]


@pytest.mark.parametrize(
    "raw",
    [
        [],
        [2],
        [-1],
        [0.5],
        [np.nan],
        [np.inf],
        [[0, 1]],
        ["private"],
        ["0"],
        [Decimal("1")],
        np.array([1 + 4j]),
    ],
)
def test_invalid_encoded_decision(raw, monkeypatch):
    model = XGBClassifier(n_estimators=1, n_jobs=1).fit([[0], [1]], [0, 1])
    predictor = Adapter().predictor(model)
    monkeypatch.setattr(model, "predict", lambda _: raw)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        predictor.predict_class_indices([[0]])
    assert "private" not in str(error.value) and len(str(error.value)) < 160


def test_native_call_without_probability_inference(monkeypatch):
    rows = [[0], [1], [2], [3]]
    model = XGBClassifier(n_estimators=2, n_jobs=1).fit(rows, [0, 1, 0, 1])
    predictor = Adapter().predictor(model)
    expected = [[int(index)] for index in model.predict(rows)]
    monkeypatch.setattr(model, "predict_proba", lambda *a: pytest.fail("argmax"))
    assert predictor.predict_class_indices(rows) == expected

    def fail(_):
        raise ValueError("private prediction")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(
        mlbundle.InvalidModelError, match=r"^XGBoost prediction failed\.$"
    ):
        predictor.predict_class_indices(rows)
    with pytest.raises(ValueError, match="classifier"):
        Adapter().predictor(
            XGBRegressor(n_estimators=1, n_jobs=1).fit(rows, [0, 1, 0, 1])
        ).predict_class_indices(rows)
