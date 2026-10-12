"""Bounded native-placement discovery and closed Pipeline options."""

from typing import Any

import numpy as np
from sklearn import get_config
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from mlbundle._pipeline_contract import OPTIONS, valid_step_names
from mlbundle.errors import InvalidModelError, LimitExceededError, UnsupportedModelError


def objects(root: Any):
    pending = [root]
    seen = set()
    visits = 0
    while pending:
        value = pending.pop()
        visits += 1
        if visits > 100_000:
            raise LimitExceededError("Pipeline state work exceeds the limit.")
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
            children = value.flat
        else:
            yield value
            children = vars(value).values() if isinstance(value, BaseEstimator) else ()
        if len(pending) + len(children) > 100_000:
            raise LimitExceededError("Pipeline state work exceeds the limit.")
        pending.extend(children)


def native_types(root: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"{type(value).__module__}.{type(value).__name__}"
                for value in objects(root)
                if {base.__module__.split(".")[0] for base in type(value).__mro__}
                & {"xgboost", "catboost"}
            }
        )
    )


def composite_candidate(model: Any) -> bool:
    found = native_types(model)
    if not found:
        return False
    if (
        isinstance(model, Pipeline)
        and model.steps
        and {base.__module__.split(".")[0] for base in type(model.steps[-1][1]).__mro__}
        & {"xgboost", "catboost"}
    ):
        if any(native_types(step) for _, step in model.steps[:-1]):
            reject_placement(found)
        validate_options(model)
        return True
    reject_placement(found)
    return False


def reject_placement(types: tuple[str, ...]) -> None:
    raise UnsupportedModelError(
        types,
        reason=(
            "Native estimators are supported only as the last step of the outermost Pipeline. "
            + str(UnsupportedModelError(types))
        ),
    )


def validate_options(model: Any) -> None:
    if (
        type(model) is not Pipeline
        or not isinstance(model.steps, (list, tuple))
        or not 1 <= len(model.steps) <= 128
    ):
        raise UnsupportedModelError(
            reason="Composite models require an exact outer Pipeline with 1 to 128 steps."
        )
    if "_metadata_request" in vars(model) or model.transform_input is not None:
        raise UnsupportedModelError(
            reason="Composite Pipeline transform_input and metadata routing are unsupported."
        )
    if set(vars(model)) != OPTIONS:
        raise InvalidModelError("Unknown outer Pipeline state.")
    if any(
        not isinstance(item, (list, tuple)) or len(item) != 2 for item in model.steps
    ):
        raise UnsupportedModelError(
            reason="Pipeline step names must be unique bounded sklearn names."
        )
    names = [name for name, _ in model.steps]
    if any(
        type(name) is not str or not 1 <= len(name) <= 256 for name in names
    ) or not valid_step_names(names):
        raise UnsupportedModelError(
            reason="Pipeline step names must be unique bounded sklearn names."
        )
    if model.memory is not None:
        raise UnsupportedModelError(
            reason="Composite Pipeline memory must be None; caching is unsupported."
        )
    if type(model.verbose) is not bool:
        raise UnsupportedModelError(
            reason="Composite Pipeline verbose must be a boolean."
        )


def validate_components(model: Pipeline, *, exporting: bool = False) -> None:
    validate_options(model)
    if exporting and get_config()["transform_output"] != "default":
        raise UnsupportedModelError(
            reason="Composite export requires default global transform_output; use per-step set_output."
        )
    for value in objects(model):
        if isinstance(value, BaseEstimator):
            if "_metadata_request" in vars(value):
                raise UnsupportedModelError(
                    reason="Composite Pipeline transform_input and metadata routing are unsupported."
                )
            config = vars(value).get("_sklearn_output_config", {})
            if (
                not isinstance(config, dict)
                or set(config) - {"transform"}
                or config.get("transform", "default") not in ("default", "pandas")
            ):
                raise UnsupportedModelError(
                    reason="Composite transformers support only default or pandas set_output."
                )
            if isinstance(value, Pipeline) and value is not model:
                validate_options(value)
