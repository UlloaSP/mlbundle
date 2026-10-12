"""Deterministic sklearn parameters; these records never select loading code."""

from typing import cast

import numpy as np
from pydantic import JsonValue
from sklearn.base import BaseEstimator

from mlbundle.canonical import canonical_value
from mlbundle.errors import LimitExceededError, UnsupportedModelError
from mlbundle.limits import HYPERPARAMETER_BYTES

MAX_VISITS = 100_000


def _parameter(
    value: object, ancestors: frozenset[int], budget: list[int] | None = None
) -> JsonValue:
    budget = [0] if budget is None else budget
    budget[0] += 1
    if budget[0] > MAX_VISITS or len(ancestors) > 16:
        raise LimitExceededError("Sklearn parameter work exceeds the limit.")
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
                "params": parameters(value, path, budget),
            }
        }
    if isinstance(value, np.random.RandomState):
        return {"$random_state": canonical_value(value.get_state())}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise UnsupportedModelError(reason="Hyperparameter keys must be strings.")
        return {key: _parameter(item, path, budget) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_parameter(item, path, budget) for item in value]
    if callable(value):
        raise UnsupportedModelError(
            reason="Callable and type hyperparameters are unsupported."
        )
    return canonical_value(value, max_bytes=HYPERPARAMETER_BYTES, max_depth=16)


def parameters(
    model: BaseEstimator,
    ancestors: frozenset[int] = frozenset(),
    budget: list[int] | None = None,
) -> dict[str, JsonValue]:
    budget = [0] if budget is None else budget
    return cast(
        dict[str, JsonValue],
        {
            key: _parameter(value, ancestors, budget)
            for key, value in model.get_params(deep=True).items()
        },
    )
