"""Strict fitted dimensions shared by the closed width validators."""

from typing import Any

import numpy as np

from mlbundle.errors import InvalidModelError
from mlbundle.limits import MAX_FEATURES


def width(value: Any, *, zero: bool = False) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not (0 if zero else 1) <= value <= MAX_FEATURES
    ):
        raise InvalidModelError("Invalid transformer feature width.")
    return int(value)


def dimensions(condition: bool) -> None:
    if not condition:
        raise InvalidModelError("Transformer fitted dimensions disagree.")


def vector(
    model: Any, name: str, count: int, *, required: bool = True
) -> np.ndarray | None:
    value = getattr(model, name, None)
    if value is None:
        dimensions(not required)
        return None
    result = np.asarray(value)
    dimensions(result.shape == (count,))
    return result


def input_width(model: Any, count: int) -> None:
    native = getattr(model, "n_features_in_", None)
    if native is not None and width(native) != count:
        raise InvalidModelError("Transformer input widths disagree.")
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        values = np.asarray(names)
        dimensions(
            values.shape == (count,)
            and all(type(x) is str and 1 <= len(x) <= 1024 for x in values)
            and len(set(values)) == count
        )


def indices(value: Any, count: int) -> list[int]:
    array = np.asarray(value)
    dimensions(array.ndim == 1 and array.dtype.kind in "iu" and len(array) <= count)
    result = [int(item) for item in array]
    dimensions(
        all(0 <= item < count for item in result) and len(set(result)) == len(result)
    )
    return result
