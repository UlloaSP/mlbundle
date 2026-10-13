"""Real Python callback training is rejected; native metric strings round-trip."""

import warnings

import catboost as cb
import numpy as np
import pytest

import mlbundle
from catboost_helpers import data
from mlbundle._catboost import Adapter


class Objective:
    def calc_ders_range(self, approxes, targets, weights):
        return [
            (float(target - approx), -1.0)
            for approx, target in zip(approxes, targets, strict=True)
        ]


class Metric:
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, targets, weights):
        return float(np.square(np.asarray(approxes[0]) - targets).sum()), float(
            len(targets)
        )

    def get_final_error(self, error, weight):
        return error / weight


@pytest.mark.parametrize("custom", ["objective", "metric", None])
def test_real_callbacks_and_native_metrics(custom, tmp_path):
    x, y = data()
    options = (
        {"loss_function": Objective(), "eval_metric": "RMSE"}
        if custom == "objective"
        else {"loss_function": "RMSE", "eval_metric": Metric() if custom else "MAE"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = cb.CatBoostRegressor(
            iterations=2,
            depth=2,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **options,
        ).fit(x, y)
    path = tmp_path / "metrics.mlbundle"
    if custom:
        with pytest.raises(mlbundle.UnsupportedModelError, match="Custom Python"):
            mlbundle.export(model, path)
        with pytest.raises(mlbundle.MLBundleError):
            Adapter().deserialize(model._serialize_model())
        assert not path.exists()
    else:
        mlbundle.export(model, path)
        assert np.array_equal(mlbundle.load(path).predict(x), model.predict(x))
