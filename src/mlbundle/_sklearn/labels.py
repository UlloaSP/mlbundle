"""One label/count check for every fitted classifier, including nested models."""

from typing import Any, cast

import numpy as np
from pydantic import ConfigDict, TypeAdapter
from sklearn.neural_network import MLPClassifier

from mlbundle.canonical import canonical_dumps, canonical_value
from mlbundle.claims import ClassLabels, validate_label_sets
from mlbundle.errors import InvalidModelError

# The pinned pre-commit Pyright predates Annotated aliases as TypeForm arguments.
_LABELS = TypeAdapter(
    cast(Any, ClassLabels), config=ConfigDict(strict=True, allow_inf_nan=False)
)


def validate_classes(model: Any) -> None:
    classes = model.classes_ if isinstance(model.classes_, list) else [model.classes_]
    if not classes or any(np.asarray(labels).ndim != 1 for labels in classes):
        raise InvalidModelError("Invalid classifier label dimensions.")
    counts = np.asarray([len(labels) for labels in classes])
    labels = _LABELS.validate_python(canonical_value(classes))
    validate_label_sets(labels)
    binarizer = getattr(model, "_label_binarizer", None)
    if binarizer is not None and canonical_dumps(model.classes_) != canonical_dumps(
        binarizer.classes_
    ):
        raise InvalidModelError("Classifier and label binarizer classes disagree.")
    _outputs(model, len(classes))
    if hasattr(model, "n_classes_"):
        native = np.asarray(model.n_classes_)
        if (
            native.dtype.kind not in "iu"
            or native.ndim != (0 if len(classes) == 1 else 1)
            or not np.array_equal(np.atleast_1d(native), counts)
        ):
            raise InvalidModelError(
                "Classifier labels and native class counts disagree."
            )
    if hasattr(model, "tree_") and not np.array_equal(model.tree_.n_classes, counts):
        raise InvalidModelError("Classifier labels and tree class counts disagree.")


def _outputs(model: Any, count: int) -> None:
    if (
        not isinstance(model, MLPClassifier)
        and hasattr(model, "n_outputs_")
        and model.n_outputs_ != count
    ):
        raise InvalidModelError("Classifier labels and native output counts disagree.")
