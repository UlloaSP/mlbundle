"""Exact wrappers, verified losses and numeric symmetric trees only."""

from typing import Any

import catboost as cb

from mlbundle.canonical import canonical_dumps
from mlbundle.errors import InvalidJSONError, UnsupportedModelError
from mlbundle.limits import HYPERPARAMETER_BYTES

CLASSIFIERS = frozenset({"Logloss", "CrossEntropy", "MultiClass"})
REGRESSORS = frozenset({"RMSE", "MAE", "Quantile", "Huber", "Poisson", "MAPE"})
CLASSES = (cb.CatBoostClassifier, cb.CatBoostRegressor)
NATIVE_CLASS = type(vars(cb.CatBoostClassifier())["_object"])
MAX_TREES = 100_000
MAX_DEPTH = 16
WRAPPER_FIELDS = frozenset(
    {
        "_init_params",
        "_canonized_params",
        "_object",
        "_is_fitted_",
        "_random_seed",
        "_learning_rate",
        "_tree_count",
        "_n_features_in",
        "_prediction_values_change",
        "_loss_value_change",
    }
)


def _parameters(params: dict[str, Any]) -> None:
    for key in ("loss_function", "objective", "eval_metric"):
        if params.get(key) is not None and not isinstance(params[key], str):
            raise UnsupportedModelError(
                reason="Custom Python CatBoost objectives and metrics are unsupported."
            )
    metrics = params.get("custom_metric", [])
    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is not None and any(not isinstance(metric, str) for metric in metrics):
        raise UnsupportedModelError(
            reason="Custom Python CatBoost objectives and metrics are unsupported."
        )
    try:
        canonical_dumps(params, max_bytes=HYPERPARAMETER_BYTES)
    except InvalidJSONError:
        raise UnsupportedModelError(
            reason="CatBoost parameters must contain only JSON values."
        ) from None


def _wrapper(model: Any, loss: str) -> None:
    cached = model._canonized_params
    params = model.get_params() if cached is None else cached
    if not isinstance(params, dict):
        raise UnsupportedModelError(reason="Invalid CatBoost wrapper parameter cache.")
    _parameters(params)
    if type(model) is cb.CatBoostRegressor:
        declared = params.get("loss_function", params.get("objective"))
        mode = "RawFormulaVal"
        if isinstance(declared, str):
            if declared.startswith(("Poisson", "Tweedie")):
                mode = "Exponent"
            elif declared == "RMSEWithUncertainty":
                mode = "RMSEWithUncertainty"
        if mode != ("Exponent" if loss == "Poisson" else "RawFormulaVal"):
            raise UnsupportedModelError(
                reason="CatBoost wrapper default prediction does not survive CBM loading."
            )


def validate(model: Any) -> None:
    if type(model) not in CLASSES:
        raise UnsupportedModelError(
            reason="Only exact CatBoostClassifier and CatBoostRegressor are supported."
        )
    if set(vars(model)) - WRAPPER_FIELDS:
        raise UnsupportedModelError(
            reason="Unknown CatBoost wrapper attributes are unsupported."
        )
    if type(vars(model).get("_object")) is not NATIVE_CLASS:
        raise UnsupportedModelError(
            reason="Only the native CatBoost model object is supported."
        )
    if not model.is_fitted():
        raise UnsupportedModelError(reason="CatBoost estimator must be fitted.")
    _parameters(model.get_params())
    if model.get_cat_feature_indices():
        raise UnsupportedModelError(
            reason="CatBoost categorical features are unsupported."
        )
    if model.get_text_feature_indices() or model.get_embedding_feature_indices():
        raise UnsupportedModelError(
            reason="CatBoost text and embedding features are unsupported."
        )
    native = model.get_all_params()
    loss = native["loss_function"].split(":", 1)[0]
    allowed = CLASSIFIERS if type(model) is cb.CatBoostClassifier else REGRESSORS
    if loss not in allowed:
        raise UnsupportedModelError(
            reason="Unsupported CatBoost loss; only closed single-target losses are supported."
        )
    if native.get("grow_policy") != "SymmetricTree":
        raise UnsupportedModelError(
            reason="Only CatBoost SymmetricTree is supported; Depthwise/Lossguide are unsupported."
        )
    _wrapper(model, loss)
