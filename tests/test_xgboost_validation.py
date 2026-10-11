"""XGBoost validation contracts and rejection paths."""

import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
import xgboost as xgb

import mlbundle
from mlbundle import InvalidModelError, UnsupportedModelError
from mlbundle._xgboost import Adapter
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.declarations import export_declarations
from xgboost_helpers import classifier


@pytest.mark.parametrize(
    "labels",
    [
        [[0, 0, 1]],
        [[0, 1]],
        [[0, 1, 2, 3]],
        [["a", 1, "c"]],
        [[None, None, None]],
        [[float("nan"), 1, 2]],
        [[{}, {}, {}]],
    ],
)
def test_label_rejections(labels, xgb_data, tmp_path):
    x, y = xgb_data
    with pytest.raises(UnsupportedModelError):
        mlbundle.export(
            classifier(x, y), tmp_path / "bad.mlbundle", class_labels=labels
        )
    assert not list(tmp_path.iterdir())


def test_unsupported_models(xgb_data, tmp_path):
    x, y = xgb_data

    def custom(labels, predictions):
        return predictions - labels, np.ones_like(predictions)

    models = [
        xgb.XGBClassifier(),
        xgb.XGBRegressor(),
        xgb.XGBRanker(n_estimators=2, n_jobs=1).fit(x, y, qid=np.zeros(len(y))),
        xgb.XGBRegressor(n_estimators=2, n_jobs=1, objective=custom).fit(x, y),
        classifier(x, y).get_booster(),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        models.extend(
            [
                xgb.XGBRFClassifier(n_estimators=2, n_jobs=1).fit(x, y),
                xgb.XGBRFRegressor(n_estimators=2, n_jobs=1).fit(x, y),
            ]
        )
    for model in models:
        with pytest.raises(UnsupportedModelError):
            mlbundle.export(model, tmp_path / "bad.mlbundle")
    regressor = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, y)
    with pytest.raises(UnsupportedModelError, match="Regressors"):
        mlbundle.export(regressor, tmp_path / "bad.mlbundle", class_labels=[[0, 1, 2]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        linear = xgb.XGBRegressor(booster="gblinear", n_estimators=2, n_jobs=1).fit(
            x, y
        )
        with pytest.raises(UnsupportedModelError, match="gbtree"):
            mlbundle.export(linear, tmp_path / "bad.mlbundle")
    multi = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, np.column_stack([y, y]))
    with pytest.raises(UnsupportedModelError, match="single target"):
        mlbundle.export(multi, tmp_path / "bad.mlbundle")


def test_lazy_import_and_missing_extra(xgb_data, tmp_path, monkeypatch):
    x, y = xgb_data
    path = tmp_path / "model.mlbundle"
    model = classifier(x, y)
    mlbundle.export(model, path)
    code = "import sys,mlbundle; mlbundle.inspect(sys.argv[1]); assert not {'sklearn','skops','numpy','pandas','xgboost','onnx','onnxruntime'} & sys.modules.keys(); print('lazy import and inspect: OK')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lazy import and inspect: OK" in result.stdout
    original = mlbundle.registry.import_module

    def missing(name):
        if name == "mlbundle._xgboost":
            raise ModuleNotFoundError(name="xgboost")
        return original(name)

    monkeypatch.setattr(mlbundle.registry, "import_module", missing)
    for action in [lambda: mlbundle.export(model, path), lambda: mlbundle.load(path)]:
        with pytest.raises(
            mlbundle.MissingOptionalDependencyError, match="mlbundle\\[xgboost\\]"
        ):
            action()


def test_prediction_validation(xgb_data, monkeypatch):
    x, y = xgb_data
    model = classifier(x, y)
    predictor = Adapter().predictor(model)
    with pytest.raises(ValueError, match="width"):
        predictor.predict_proba([[0]])
    with pytest.raises(ValueError, match="regressor"):
        predictor.predict(x)
    for output in [[[float("nan")] * 3], [[0.5] * 3], [[1.1, -0.1, 0]], [[1.0]]]:
        monkeypatch.setattr(
            model, "predict_proba", lambda values, output=output: output
        )
        with pytest.raises(InvalidModelError):
            predictor.predict_proba([x[0]])
    regressor = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, y)
    predictor = Adapter().predictor(regressor)
    with pytest.raises(ValueError, match="classifier"):
        predictor.predict_proba(x)
    monkeypatch.setattr(
        regressor, "predict", lambda values: [float("inf")] * len(values)
    )
    with pytest.raises(InvalidModelError):
        predictor.predict(x)


def test_binding_validation_and_limits(xgb_data, tmp_path):
    x, y = xgb_data
    model = Adapter().prepare_export(classifier(x, y), export_declarations(None))
    binding = canonical_loads(model.get_booster().attr("mlbundle").encode())
    for change in [
        lambda v: v.update(estimator_class="os.system"),
        lambda v: v.update(class_labels=None),
        lambda v: v["prediction"].update(extra="bad"),
        lambda v: v["prediction"].update(enable_categorical="yes"),
    ]:
        value = json.loads(json.dumps(binding))
        change(value)
        model.get_booster().set_attr(mlbundle=canonical_dumps(value).decode())
        with pytest.raises(mlbundle.MLBundleError):
            Adapter().deserialize(bytes(model.get_booster().save_raw(raw_format="ubj")))
    for limits in [
        mlbundle.ArchiveLimits(max_member_bytes=1),
        mlbundle.ArchiveLimits(max_structure_bytes=1),
    ]:
        with pytest.raises(mlbundle.LimitExceededError):
            mlbundle.export(classifier(x, y), tmp_path / "bad.mlbundle", limits=limits)
    assert not list(tmp_path.iterdir())
