"""Supported-scope failures, native prediction errors, and optional dependency."""

import subprocess
import sys

import catboost as cb
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import mlbundle
from catboost_helpers import classifier, data, regressor
from mlbundle._catboost import Adapter
from mlbundle.declarations import export_declarations


def test_exact_classes_fitted_state_and_labels(tmp_path):
    class CustomClassifier(cb.CatBoostClassifier):
        pass

    for model in [
        cb.CatBoostClassifier(),
        cb.CatBoostRegressor(),
        cb.CatBoost(),
        cb.CatBoostRanker(),
        CustomClassifier(),
    ]:
        with pytest.raises(mlbundle.UnsupportedModelError):
            mlbundle.export(model, tmp_path / "bad.mlbundle")
    for model, labels in [(classifier(), [[0, 1, 2]]), (regressor(), [[0, 1]])]:
        with pytest.raises(
            mlbundle.UnsupportedModelError, match="class_labels is unsupported"
        ):
            mlbundle.export(model, tmp_path / "bad.mlbundle", class_labels=labels)
    assert not list(tmp_path.iterdir())


def test_pipeline_rejection_lists_types_and_direct_export(tmp_path):
    model = classifier()
    with pytest.raises(mlbundle.UnsupportedModelError) as error:
        mlbundle.export(Pipeline([("est", model)]), tmp_path / "pipeline.mlbundle")
    assert "catboost.core.CatBoostClassifier" in error.value.untrusted_types
    assert "Export the estimator directly" in str(error.value)
    assert 'pipeline.named_steps["est"]' in str(error.value)
    mlbundle.export(model, tmp_path / "estimator.mlbundle")


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
def test_non_symmetric_trees_rejected(policy, tmp_path):
    model = classifier(grow_policy=policy)
    with pytest.raises(mlbundle.UnsupportedModelError, match="SymmetricTree"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")


@pytest.mark.parametrize(
    "loss",
    [
        "MultiClassOneVsAll",
        "MultiRMSE",
        "RMSEWithUncertainty",
        "Tweedie:variance_power=1.5",
    ],
)
def test_other_losses_and_multi_target_rejected(loss, tmp_path):
    x, y = data()
    cls = (
        cb.CatBoostClassifier if loss == "MultiClassOneVsAll" else cb.CatBoostRegressor
    )
    target = np.column_stack([y, y]) if loss == "MultiRMSE" else y + 1.0
    model = cls(
        iterations=2,
        depth=2,
        loss_function=loss,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    ).fit(x, target)
    with pytest.raises(mlbundle.UnsupportedModelError, match="single-target losses"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")


@pytest.mark.parametrize("feature", ["cat", "text", "embedding"])
def test_feature_types_rejected(feature, tmp_path):
    x = pd.DataFrame(
        {
            "number": np.arange(30),
            "special": ["red apple", "blue pear", "green plum"] * 10,
        }
    )
    options = {
        "cat": {"cat_features": [1]},
        "text": {
            "text_features": [1],
            "text_processing": {
                "tokenizers": [
                    {
                        "tokenizer_id": "Space",
                        "separator_type": "ByDelimiter",
                        "delimiter": " ",
                    }
                ],
                "dictionaries": [
                    {
                        "dictionary_id": "Word",
                        "occurrence_lower_bound": "1",
                        "gram_order": "1",
                    }
                ],
                "feature_processing": {
                    "default": [
                        {
                            "dictionaries_names": ["Word"],
                            "feature_calcers": ["BoW"],
                            "tokenizers_names": ["Space"],
                        }
                    ]
                },
            },
        },
        "embedding": {"embedding_features": [1]},
    }
    if feature == "embedding":
        x["special"] = [[float(i), float(i % 3)] for i in range(30)]
    model = cb.CatBoostClassifier(
        iterations=2,
        depth=2,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
        **options[feature],
    ).fit(x, np.arange(30) % 2)
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="features are unsupported"
    ):
        mlbundle.export(model, tmp_path / "bad.mlbundle")


@pytest.mark.parametrize("key", ["loss_function", "eval_metric", "custom_metric"])
def test_custom_python_parameters_fail_before_publication(key, tmp_path):
    model = classifier()
    model._init_params[key] = [object()] if key == "custom_metric" else object()
    with pytest.raises(mlbundle.UnsupportedModelError, match="Custom Python"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert not list(tmp_path.iterdir())


def test_lazy_inspect_and_missing_extra(tmp_path, monkeypatch):
    model = classifier()
    path = tmp_path / "catboost.mlbundle"
    mlbundle.export(model, path)
    code = "import sys,mlbundle; mlbundle.inspect(sys.argv[1]); assert not {'catboost','sklearn','skops','numpy','pandas','xgboost','onnx','onnxruntime'} & sys.modules.keys(); print('lazy import and inspect: OK')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lazy import and inspect: OK" in result.stdout
    original = mlbundle.registry.import_module

    def missing(name):
        if name == "mlbundle._catboost":
            raise ModuleNotFoundError(name="catboost")
        return original(name)

    monkeypatch.setattr(mlbundle.registry, "import_module", missing)
    for action in [lambda: mlbundle.export(model, path), lambda: mlbundle.load(path)]:
        with pytest.raises(
            mlbundle.MissingOptionalDependencyError, match="mlbundle\\[catboost\\]"
        ):
            action()


def test_input_output_and_native_error_boundaries(monkeypatch):
    x, _ = data()
    model = classifier()
    predictor = Adapter().predictor(model)
    for bad in [[[0]], np.empty((0, 3)), x[list("cba")]]:
        with pytest.raises(mlbundle.PredictionInputError):
            predictor.predict_proba(bad)
    for bad in [[["secret", 0, 1]], [[float("inf"), 0, 1]], [[1j, 0, 1]]]:
        with pytest.raises(mlbundle.InvalidModelError) as error:
            predictor.predict_proba(bad)
        assert "secret" not in str(error.value)
    with pytest.raises(ValueError, match="regressor"):
        predictor.predict(x)
    for values in [[[np.nan] * 3], [[0.5] * 3], [[1.1, -0.1, 0]], [[1.0]]]:
        monkeypatch.setattr(
            model, "predict_proba", lambda *a, values=values, **kw: values
        )
        with pytest.raises(mlbundle.InvalidModelError):
            predictor.predict_proba([x.iloc[0].tolist()])

    def native_failure(*a, **kw):
        raise cb.CatBoostError("secret native error")

    monkeypatch.setattr(model, "predict_proba", native_failure)
    with pytest.raises(
        mlbundle.InvalidModelError, match="CatBoost prediction failed"
    ) as error:
        predictor.predict_proba(x)
    assert "secret" not in str(error.value)
    model = regressor()
    predictor = Adapter().predictor(model)
    with pytest.raises(ValueError, match="classifier"):
        predictor.predict_proba(x)
    monkeypatch.setattr(model, "predict", lambda *a, **kw: [np.inf] * len(x))
    with pytest.raises(mlbundle.InvalidModelError, match="regression prediction"):
        predictor.predict(x)


def test_export_failures_are_bounded(tmp_path, monkeypatch):
    model = classifier()

    def failure(*a, **kw):
        raise cb.CatBoostError("secret native error")

    monkeypatch.setattr(cb.CatBoost, "save_model", failure)
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="CBM serialization failed"
    ):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    monkeypatch.setattr(cb.CatBoost, "get_all_params", failure)
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="export preparation failed"
    ):
        Adapter().prepare_export(model, export_declarations(None))


@pytest.mark.parametrize(
    "override", ["predict", "predict_proba", "_base_predict", "review_extra"]
)
def test_unknown_wrapper_state_rejected(override, tmp_path):
    model = classifier()
    setattr(model, override, lambda *args: [[1.0, 0.0]])
    with pytest.raises(mlbundle.UnsupportedModelError, match="wrapper attributes"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cache", [None, {"loss_function": "RMSE"}])
def test_poisson_wrapper_default_must_survive(cache, tmp_path):
    trained = regressor("Poisson")
    model = cb.CatBoostRegressor(loss_function=None).load_model(
        blob=trained._serialize_model()
    )
    model._canonized_params = cache
    with pytest.raises(mlbundle.UnsupportedModelError, match="default prediction"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert not list(tmp_path.iterdir())


def test_non_poisson_cache_cannot_change_prediction_mode(tmp_path):
    model = regressor()
    model._canonized_params = {"loss_function": "Poisson"}
    with pytest.raises(mlbundle.UnsupportedModelError, match="default prediction"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")


def test_native_object_proxy_is_rejected(tmp_path):
    model = classifier()
    native = model._object

    class Proxy:
        def __getattr__(self, name):
            return getattr(native, name)

    model._object = Proxy()
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="native CatBoost model object"
    ):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert not list(tmp_path.iterdir())
