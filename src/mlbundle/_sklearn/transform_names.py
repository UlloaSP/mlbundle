"""Input and inner display names, separately from native estimator names."""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures

from mlbundle.claims import ModelClaims
from mlbundle.errors import InvalidModelError, UnsupportedModelError

from ._width_checks import input_width, width
from .polynomial_names import polynomial_names
from .transform_width import identity


def _first_boundary(prefix: Pipeline) -> tuple[int, list[str] | None] | None:
    for _, step in prefix.steps:
        if identity(step):
            continue
        if type(step) is Pipeline:
            result = _first_boundary(step)
            if result is not None:
                return result
            continue
        count = getattr(step, "n_features_in_", None)
        if count is not None:
            count = width(count)
            input_width(step, count)
            names = getattr(step, "feature_names_in_", None)
            return count, None if names is None else list(names)
        if type(step) is not FunctionTransformer:
            raise UnsupportedModelError(
                reason="Composite input width cannot be recovered from fitted state."
            )
    return None


def input_boundary(
    prefix: Pipeline | None, leaf: ModelClaims
) -> tuple[int, list[str] | None]:
    result = None if prefix is None else _first_boundary(prefix)
    return (leaf.n_features, leaf.feature_names) if result is None else result


def output_names(model: Any, names: list[str]) -> list[str] | None:
    if identity(model):
        return names
    native = getattr(model, "feature_names_in_", None)
    if native is not None and list(native) != names:
        raise InvalidModelError("Invalid transformed feature names.")
    if type(model) is Pipeline:
        for _, step in model.steps:
            result = output_names(step, names)
            if result is None:
                return None
            names = result
        return names
    return leaf_names(model, names)


def leaf_names(model: Any, names: list[str]) -> list[str] | None:
    if type(model) is FunctionTransformer and model.feature_names_out is None:
        return names
    if type(model) is PolynomialFeatures:
        return checked_names(polynomial_names(model, names))
    if type(model) is ColumnTransformer:
        return column_names(model, names)
    if type(model) is OneHotEncoder:
        # Check string conversion/concatenation bounds before sklearn allocates names.
        for categories in model.categories_:
            for category in categories:
                if len(str(category)) > 1024:
                    raise InvalidModelError("Invalid transformed feature names.")
    method = getattr(model, "get_feature_names_out", None)
    if method is None:
        return None
    return checked_names(method(names))


def checked_names(value: Any) -> list[str]:
    result = np.asarray(value)
    if (
        result.ndim != 1
        or any(
            type(name) not in (str, np.str_) or not 1 <= len(name) <= 1024
            for name in result
        )
        or len(set(result)) != len(result)
    ):
        raise InvalidModelError("Invalid transformed feature names.")
    return [str(name) for name in result]


def column_names(model: ColumnTransformer, names: list[str]) -> list[str] | None:
    if type(model.verbose_feature_names_out) is not bool:
        raise InvalidModelError("Invalid transformed feature names.")
    result = []
    for name, transformer, _ in model.transformers_:
        selected = model._transformer_to_input_indices[name]
        if not selected or (type(transformer) is str and transformer == "drop"):
            continue
        branch = output_names(transformer, [names[i] for i in selected])
        if branch is None:
            return None
        result.extend(
            f"{name}__{item}" if model.verbose_feature_names_out else item
            for item in branch
        )
    return checked_names(result)
