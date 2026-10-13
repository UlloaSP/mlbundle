"""Deterministic sklearn parameters; these records never select loading code."""

from typing import cast

import numpy as np
from pydantic import JsonValue
from sklearn.base import BaseEstimator

from mlbundle.canonical import canonical_value
from mlbundle.errors import LimitExceededError, UnsupportedModelError
from mlbundle.limits import HYPERPARAMETER_BYTES

from .trusted import dtype_identifier, reference_identifier

MAX_VISITS = 100_000


def _parameter(
    value: object,
    ancestors: frozenset[int],
    budget: list[int] | None = None,
    *,
    estimator_ref: bool = False,
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
        return _estimator(value, path, budget, estimator_ref)
    if isinstance(value, np.random.RandomState):
        return {"$random_state": canonical_value(value.get_state())}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise UnsupportedModelError(reason="Hyperparameter keys must be strings.")
        return {key: _parameter(item, path, budget) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [
            _parameter(item, path, budget, estimator_ref=estimator_ref)
            for item in value
        ]
    return _leaf(value)


def _leaf(value: object) -> JsonValue:
    if isinstance(value, np.dtype):
        return {"$dtype": dtype_identifier(value)}
    if callable(value):
        identifier = reference_identifier(value)
        if identifier is not None:
            return {"$type" if isinstance(value, type) else "$function": identifier}
        raise UnsupportedModelError(
            reason="Callable and type hyperparameters are unsupported."
        )
    return canonical_value(value, max_bytes=HYPERPARAMETER_BYTES, max_depth=16)


def _estimator(value, ancestors, budget, reference) -> JsonValue:
    # Named deep keys own step parameters; constructor lists retain routing/order.
    if reference:
        return {"$estimator_ref": type(value).__name__}
    return {
        "$estimator": {
            "class": type(value).__name__,
            "params": parameters(value, ancestors, budget),
        }
    }


def parameters(
    model: BaseEstimator,
    ancestors: frozenset[int] = frozenset(),
    budget: list[int] | None = None,
) -> dict[str, JsonValue]:
    budget = [0] if budget is None else budget
    return cast(
        dict[str, JsonValue],
        {
            key: _parameter(
                value,
                ancestors,
                budget,
                estimator_ref=key in ("steps", "transformers", "transformer_list"),
            )
            for key, value in model.get_params(deep=True).items()
        },
    )
