"""All adapters bound invalid inputs and remaining native prediction failures."""

from types import SimpleNamespace

import pytest
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression

import mlbundle
from mlbundle import MLBundleError
from mlbundle._onnx import Adapter as OnnxAdapter
from mlbundle._sklearn import Adapter as SklearnAdapter
from mlbundle._xgboost import Adapter as XgbAdapter
from onnx_helpers import classifier, regressor


@pytest.fixture(params=["sklearn", "xgboost", "onnx"])
def predictors(request):
    x = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y = [0, 0, 1, 1]
    if request.param == "sklearn":
        adapter = SklearnAdapter()
        return adapter.predictor(LinearRegression().fit(x, y)), adapter.predictor(
            LogisticRegression().fit(x, y)
        )
    if request.param == "xgboost":
        adapter = XgbAdapter()
        return adapter.predictor(
            xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, y)
        ), adapter.predictor(xgb.XGBClassifier(n_estimators=2, n_jobs=1).fit(x, y))
    adapter = OnnxAdapter()
    return adapter.predictor(
        adapter.deserialize(regressor().SerializeToString())
    ), adapter.predictor(adapter.deserialize(classifier().SerializeToString()))


@pytest.mark.parametrize(
    "data",
    [
        [["private-cell", 0]],
        [[float("inf"), 0]],
        [[float("-inf"), 0]],
        [[1e1000, 0]],
        [[0]],
        [0, 0],
        [[0, 0], [0]],
        [[complex(1), 0]],
    ],
)
def test_prediction_input_errors_are_bounded(data, predictors):
    regression, classification = predictors
    for method in [regression.predict, classification.predict_proba]:
        with pytest.raises(MLBundleError) as error:
            method(data)
        assert len(str(error.value)) < 160
        assert "private-cell" not in str(error.value)


def test_native_prediction_errors_are_bounded(predictors, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("private-cell " + "x" * 10000)

    regression, classification = predictors
    for predictor, name in [(regression, "predict"), (classification, "predict_proba")]:
        if hasattr(predictor, "session"):
            predictor.session = SimpleNamespace(run=fail)
        else:
            monkeypatch.setattr(predictor.model, name, fail)
        with pytest.raises(MLBundleError) as error:
            getattr(predictor, name)([[0, 0]])
        assert "private-cell" not in str(error.value) and len(str(error.value)) < 160


def test_sklearn_name_errors_do_not_echo_names(tmp_path):
    import pandas as pd

    path = tmp_path / "named.mlbundle"
    model = LogisticRegression().fit(
        pd.DataFrame([[0], [1]], columns=["private-feature"]), [0, 1]
    )
    mlbundle.export(model, path)
    with pytest.raises(MLBundleError) as error:
        mlbundle.load(path).predict_proba(pd.DataFrame([[0]], columns=["private-cell"]))
    assert "private" not in str(error.value)
