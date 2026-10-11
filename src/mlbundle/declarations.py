"""Export declarations and resolution for models without native class labels."""

from collections.abc import Sequence
from typing import cast

from pydantic import ValidationError

from ._base import ContractModel
from .canonical import canonical_dumps
from .claims import ClassLabels, Label, ModelClaims, validate_labels
from .contract import ModelKind
from .errors import UnsupportedModelError


class ExportDeclarations(ContractModel):
    class_labels: ClassLabels | None = None


def export_declarations(
    labels: Sequence[Sequence[Label]] | None,
) -> ExportDeclarations:
    try:
        if labels is not None and any(
            isinstance(output, (str, bytes)) for output in labels
        ):
            raise ValueError
        value = None if labels is None else [list(output) for output in labels]
        declarations = ExportDeclarations(class_labels=value)
        for output in declarations.class_labels or []:
            if len({type(label) for label in output}) != 1:
                raise ValueError
        return declarations
    except (ValidationError, TypeError, ValueError):
        raise UnsupportedModelError(
            reason="Invalid class_labels declaration."
        ) from None


def resolve_labels(
    claims: ModelClaims, declarations: ExportDeclarations
) -> ModelClaims:
    """Native labels win; otherwise verify declarations against native class counts."""
    labels = claims.class_labels
    if claims.kind == ModelKind.REGRESSOR and declarations.class_labels is not None:
        raise UnsupportedModelError(reason="Regressors cannot accept class_labels.")
    if (
        labels is not None
        and declarations.class_labels is not None
        and canonical_dumps(labels) != canonical_dumps(declarations.class_labels)
    ):
        raise UnsupportedModelError(
            reason="Declared labels disagree with native class labels."
        )
    if labels is None and claims.kind == ModelKind.CLASSIFIER:
        labels = declarations.class_labels
        if labels is None:
            labels = cast(
                ClassLabels, [list(range(count)) for count in claims.n_classes or []]
            )
    try:
        validate_labels(claims, labels)
        return ModelClaims.model_validate(
            {**claims.model_dump(), "class_labels": labels}
        )
    except (ValidationError, ValueError):
        raise UnsupportedModelError(
            reason="Class labels must match native counts and kind."
        ) from None
