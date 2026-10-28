"""CatBoost native round trips, every admitted loss, labels, and publication."""

import subprocess
import sys
from pathlib import Path

import catboost as cb
import numpy as np
import pytest

import mlbundle
from catboost_helpers import classifier, data, regressor
from mlbundle._catboost import Adapter
from mlbundle._catboost.scope import REGRESSORS
from mlbundle.canonical import canonical_dumps


@pytest.mark.parametrize(
    "labels", [[0, 1], [10, 20], ["cold", "warm"], [10, 20, 30], ["a", "b", "c"]]
)
def test_classifier_roundtrip(labels, tmp_path):
    x, y = data()
    model = classifier(labels=np.asarray(labels)[y % len(labels)])
    before = model._serialize_model()
    path = tmp_path / "classifier.mlbundle"
    view = mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert type(loaded.native_model) is cb.CatBoostClassifier
    assert view.manifest == loaded.manifest
    assert model._serialize_model() == before
    assert loaded.library == "catboost"
    assert view.manifest.model.serialization == "cbm"
    assert view.environment_lock.version("catboost") == cb.__version__
    assert loaded.class_labels == [labels]
    assert list(loaded.feature_names) == list("abc")
    assert loaded.feature_origin == "model"
    assert loaded.native_model.n_features_in_ == 3
    assert loaded.native_model.get_all_params() == model.get_all_params()
    for inputs in [x, x.to_numpy(), x.to_numpy().tolist()]:
        assert np.array_equal(loaded.predict_proba(inputs), model.predict_proba(inputs))
    assert np.array_equal(loaded.native_model.predict(x), model.predict(x))
    assert canonical_dumps(loaded.structure) == canonical_dumps(
        Adapter().extract_structure(model)
    )
    for command in ("inspect", "verify"):
        executable = Path(sys.executable).with_name(
            "mlbundle.exe" if sys.platform == "win32" else "mlbundle"
        )
        result = subprocess.run(
            [executable, command, str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not result.stderr


@pytest.mark.parametrize("loss", ["Logloss", "CrossEntropy", "MultiClass"])
def test_classifier_losses(loss, tmp_path):
    x, y = data()
    model = classifier(labels=y if loss == "MultiClass" else y % 2, loss_function=loss)
    path = tmp_path / "loss.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(mlbundle.load(path).predict_proba(x), model.predict_proba(x))


@pytest.mark.parametrize("loss", sorted(REGRESSORS))
def test_regressor_losses(loss, tmp_path):
    options = {"Huber": "Huber:delta=1.0", "Quantile": "Quantile:alpha=0.7"}
    model = regressor(options.get(loss, loss))
    x, _ = data()
    path = tmp_path / "regressor.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert type(loaded.native_model) is cb.CatBoostRegressor
    assert loaded.class_labels is None
    assert np.array_equal(loaded.predict(x), model.predict(x))
    assert np.array_equal(loaded.native_model.predict(x), model.predict(x))
    assert loaded.native_model.get_all_params() == model.get_all_params()


def test_calibration_truncation_and_best_iteration(tmp_path):
    x, y = data()
    model = cb.CatBoostClassifier(
        iterations=9,
        depth=2,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        use_best_model=False,
    ).fit(x, y % 2, eval_set=(x, y % 2))
    model.shrink(4)
    model.set_scale_and_bias(1.2, 0.3)
    model.set_probability_threshold(0.65)
    path = tmp_path / "calibrated.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    params = loaded.manifest.model.hyperparameters
    assert params["tree_count"] == 4
    assert params["best_iteration"] == model.get_best_iteration()
    assert params["scale_and_bias"] == [1.2, 0.3]
    assert params["probability_threshold"] == 0.65
    assert np.array_equal(loaded.predict_proba(x), model.predict_proba(x))
    assert np.array_equal(loaded.native_model.predict(x), model.predict(x))


@pytest.mark.parametrize("nan_mode", ["Min", "Max", "Forbidden"])
def test_nan_handling(nan_mode, tmp_path):
    x, y = data()
    if nan_mode != "Forbidden":
        x.iloc[0, 0] = np.nan
    model = classifier(frame=x, labels=y % 2, nan_mode=nan_mode)
    path = tmp_path / "nan.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(mlbundle.load(path).predict_proba(x), model.predict_proba(x))


def test_schema_samples_and_lowered_limits(tmp_path):
    x, _ = data()
    model = classifier()
    path = tmp_path / "schema.mlbundle"
    mlbundle.export(model, path, training_data=x, sample_rows=2)
    loaded = mlbundle.load(path)
    assert loaded.samples == x.head(2).to_dict("records")
    assert loaded.schema["reports"][0]["labels"] == ["0", "1", "2"]
    assert set(loaded.members) == {
        mlbundle.MemberName.MANIFEST,
        mlbundle.MemberName.CBM,
        mlbundle.MemberName.STRUCTURE,
        mlbundle.MemberName.SCHEMA,
        mlbundle.MemberName.SAMPLES,
        mlbundle.MemberName.LOCK,
    }
    for limits in [
        mlbundle.ArchiveLimits(max_member_bytes=1),
        mlbundle.ArchiveLimits(max_structure_bytes=1),
    ]:
        with pytest.raises(mlbundle.LimitExceededError):
            mlbundle.export(model, tmp_path / "bad.mlbundle", limits=limits)
    assert not (tmp_path / "bad.mlbundle").exists()


def test_native_importance_cache_and_export_immutability(tmp_path):
    x, y = data()
    model = regressor()
    model.get_feature_importance(
        cb.Pool(x, label=y + 1.0), type="LossFunctionChange", thread_count=1
    )
    model._canonized_params = None
    before = model._serialize_model()
    cache = model._prediction_values_change
    path = tmp_path / "cache.mlbundle"
    mlbundle.export(model, path)
    assert model._canonized_params is None
    assert model._prediction_values_change is cache
    assert model._serialize_model() == before
    assert np.array_equal(mlbundle.load(path).predict(x), model.predict(x))
