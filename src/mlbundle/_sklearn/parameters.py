"""Deterministic sklearn parameters; these records never select loading code."""

from typing import cast

import numpy as np
from pydantic import JsonValue
from sklearn.base import BaseEstimator

from mlbundle.canonical import canonical_value
from mlbundle.errors import UnsupportedModelError


def _parameter(value: object, ancestors: frozenset[int]) -> JsonValue:
    if id(value) in ancestors:
        raise UnsupportedModelError(reason="Cyclic hyperparameters are unsupported.")
    path = ancestors | {id(value)}
    if isinstance(value, BaseEstimator):
        if not type(value).__module__.startswith("sklearn."):
            raise UnsupportedModelError(
                (f"{type(value).__module__}.{type(value).__name__}",)
            )
        return {
            "$estimator": {
                "class": type(value).__name__,
                "params": parameters(value, path),
            }
        }
    if isinstance(value, np.random.RandomState):
        return {"$random_state": canonical_value(value.get_state())}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise UnsupportedModelError(reason="Hyperparameter keys must be strings.")
        return {key: _parameter(item, path) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_parameter(item, path) for item in value]
    if callable(value):
        raise UnsupportedModelError(
            reason="Callable and type hyperparameters are unsupported."
        )
    return canonical_value(value)


def parameters(
    model: BaseEstimator, ancestors: frozenset[int] = frozenset()
) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            key: _parameter(value, ancestors)
            for key, value in model.get_params(deep=True).items()
        },
    )
