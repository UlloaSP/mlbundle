"""Transformer-only component using the estimator's skops/state/parameter owners."""

from typing import Any, cast

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mlbundle._pipeline_contract import TransformerInfo
from mlbundle.errors import UnsupportedModelError

from .parameters import parameters
from .payload import validate_payload
from .pipeline_scope import native_types, reject_placement, validate_components
from .serialization import SkopsIO
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


class Adapter:
    """Transformer snapshot owner for one composite export/load."""

    def __init__(self) -> None:
        self._io = SkopsIO(validate, validate_payload)

    def serialize(self, model: Pipeline) -> bytes:
        if type(model) is not Pipeline or not model.steps:
            raise UnsupportedModelError(
                reason="Transformer member must be a nonempty exact Pipeline."
            )
        validate_components(model)
        return self._io.serialize(model)

    def deserialize(self, data: bytes) -> Pipeline:
        return cast(Pipeline, self._io.deserialize(data))

    def claims(self, model: Pipeline) -> TransformerInfo:
        validate(model)
        return TransformerInfo(
            member="model.skops",
            estimator_class="Pipeline",
            hyperparameters=parameters(model),
            state_sha256=self._io.state(model),
        )
