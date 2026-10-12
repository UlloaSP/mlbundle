"""XGBoost native round trips, labels and prediction settings."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import mlbundle
from mlbundle._xgboost import Adapter
from mlbundle._xgboost.native import REGRESSORS
from mlbundle.canonical import canonical_dumps
from mlbundle.declarations import export_declarations
from xgboost_helpers import classifier


@pytest.mark.parametrize(
    "kind,frame,names",
    [
        ("binary", False, None),
        ("multiclass", True, None),
        ("regression", False, list("abcd")),
        ("multiclass", False, list("abcd")),
    ],
)
def test_roundtrip(kind, frame, names, xgb_data, tmp_path):
    x, y = xgb_data
    inputs = pd.DataFrame(x, columns=list("abcd")) if frame else x
    labels = [["setosa", "versicolor", "virginica"]] if kind == "multiclass" else None
    model = (
        xgb.XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1).fit(inputs, y)
        if kind == "regression"
        else classifier(inputs, y % 2 if kind == "binary" else y)
    )
    before = bytes(model.get_booster().save_raw(raw_format="ubj"))
    path = tmp_path / "model.mlbundle"
    view = mlbundle.export(model, path, feature_names=names, class_labels=labels)
    loaded = mlbundle.load(path)
    assert type(loaded.native_model) is type(model)
    assert view.manifest == loaded.manifest
    assert bytes(model.get_booster().save_raw(raw_format="ubj")) == before
    prepared = Adapter().prepare_export(model, export_declarations(labels))
    assert canonical_dumps(
        loaded.manifest.verified_claims().model_dump(mode="json")
    ) == canonical_dumps(Adapter().extract_claims(prepared).model_dump(mode="json"))
    assert canonical_dumps(loaded.structure) == canonical_dumps(
        Adapter().extract_structure(prepared)
    )
    method = "predict" if kind == "regression" else "predict_proba"
    for values in [inputs, x, x.tolist()]:
        assert np.array_equal(
            getattr(loaded, method)(values), getattr(model, method)(values)
        )
    assert np.array_equal(loaded.native_model.predict(inputs), model.predict(inputs))
    assert loaded.class_labels == (None if kind == "regression" else labels or [[0, 1]])
    assert loaded.feature_origin == (
        "model" if frame else "declared" if names else "generated"
    )
    if frame:
        assert loaded.manifest.features.model_dtypes == ["float"] * 4


def test_wrapper_facts_and_in_memory_native_persistence(xgb_data, tmp_path):
    x, y = xgb_data
    for labels in [np.array(["a", "b", "c"])[y], np.array([10, 20, 30])[y]]:
        with pytest.raises(ValueError, match="Invalid classes inferred"):
            classifier(x, labels)
    frame = pd.DataFrame(x, columns=list("abcd"))
    for cls in [xgb.XGBClassifier, xgb.XGBRegressor]:
        model = cls(
            n_estimators=3, max_depth=2, learning_rate=0.2, random_state=7, n_jobs=1
        ).fit(frame, y)
        booster = model.get_booster()
        booster.set_attr(probe='["red","green","blue"]', best_iteration="1")
        restored = cls()
        restored.load_model(bytearray(booster.save_raw(raw_format="ubj")))
        assert booster.attributes() == restored.get_booster().attributes()
        assert booster.feature_names == restored.get_booster().feature_names
        assert booster.feature_types == restored.get_booster().feature_types
        assert np.array_equal(model.feature_names_in_, restored.feature_names_in_)
        assert (
            model.n_features_in_ == restored.n_features_in_
            and model.best_iteration == restored.best_iteration
        )
        assert model.get_params() != restored.get_params()
        before = json.loads(booster.save_config())["learner"]
        after = json.loads(restored.get_booster().save_config())["learner"]
        assert before != after
        assert before["learner_model_param"] == after["learner_model_param"]
        assert before["objective"] == after["objective"]
        assert np.array_equal(model.predict(frame), restored.predict(frame))
        if cls is xgb.XGBClassifier:
            assert model.n_classes_ == restored.n_classes_ == 3
            assert model.classes_.tolist() == restored.classes_.tolist() == [0, 1, 2]
        path = tmp_path / f"{cls.__name__}.ubj"
        model.save_model(path)
        disk = cls()
        disk.load_model(path)
        assert np.array_equal(model.predict(frame), disk.predict(frame))
        raw = xgb.Booster()
        raw.load_model(bytearray(booster.save_raw(raw_format="ubj")))
        assert raw.attributes() == booster.attributes()
        assert np.array_equal(
            raw.predict(xgb.DMatrix(frame)), booster.predict(xgb.DMatrix(frame))
        )


@pytest.mark.parametrize(
    "labels", [[[10, 20, 30]], [[1.0, 2.0, 3.0]], [["", "b", "c"]]]
)
def test_original_label_types(labels, xgb_data, tmp_path):
    x, y = xgb_data
    path = tmp_path / "labels.mlbundle"
    mlbundle.export(classifier(x, y), path, class_labels=labels)
    assert mlbundle.load(path).class_labels == labels


@pytest.mark.parametrize("objective", sorted(REGRESSORS))
def test_supported_regression_objectives(objective, xgb_data, tmp_path):
    x, y = xgb_data
    target = y / 3 + 0.1 if objective == "reg:logistic" else y + 1.0
    model = xgb.XGBRegressor(objective=objective, n_estimators=2, n_jobs=1).fit(
        x, target
    )
    path = tmp_path / "regression.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(mlbundle.load(path).predict(x), model.predict(x))


def test_prediction_settings_and_dropout(xgb_data, tmp_path):
    x, y = xgb_data
    x = x.copy()
    x[0, 0] = -99
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = classifier(
            x,
            y,
            booster="dart",
            rate_drop=0.3,
            missing=-99,
            importance_type="weight",
            random_state=7,
        )
    model.get_booster().set_attr(best_iteration="1")
    path = tmp_path / "dropout.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.native_model.missing == -99
    assert loaded.native_model.importance_type == "weight"
    assert loaded.native_model.best_iteration == 1
    assert np.array_equal(loaded.predict_proba(x), model.predict_proba(x))
    assert np.array_equal(
        loaded.structure["feature_importances"], model.feature_importances_
    )
