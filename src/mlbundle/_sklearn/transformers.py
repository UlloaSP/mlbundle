"""Transformer-only component using the estimator's skops/state/parameter owners."""

from typing import Any, cast

import skops.io as sio
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mlbundle._pipeline_contract import TransformerInfo
from mlbundle.errors import UnsupportedModelError

from .parameters import parameters
from .payload import validate_payload
from .pipeline_scope import native_types, reject_placement, validate_components
from .serialization import deserialize as load_skops, serialize as dump_skops
from .structure import validate_tree_storage


def validate(model: Any) -> None:
    if type(model) is not Pipeline or not model.steps:
        raise UnsupportedModelError(
            reason="Transformer member must be a nonempty exact Pipeline."
        )
    validate_components(model)
    found = native_types(model)
    if found:
        reject_placement(found)
    for _, step in model.steps:
        if step is None or (type(step) is str and step == "passthrough"):
            continue
        if not callable(getattr(step, "transform", None)):
            raise UnsupportedModelError(
                reason="Transformer component contains a non-transformer."
            )
        try:
            check_is_fitted(step)
        except NotFittedError:
            raise UnsupportedModelError(
                reason="Composite transformers must be fitted."
            ) from None
    validate_tree_storage(model)
    parameters(model)


def serialize(model: Pipeline) -> bytes:
    if type(model) is not Pipeline or not model.steps:
        raise UnsupportedModelError(
            reason="Transformer member must be a nonempty exact Pipeline."
        )
    validate_components(model)
    return dump_skops(model, validate)


def deserialize(data: bytes) -> Pipeline:
    return cast(Pipeline, load_skops(data, validate))


def claims(model: Pipeline) -> TransformerInfo:
    validate(model)
    return TransformerInfo(
        member="model.skops",
        estimator_class="Pipeline",
        hyperparameters=parameters(model),
        state_sha256=validate_payload(sio.dumps(model)),
    )
