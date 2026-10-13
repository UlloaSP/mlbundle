"""Bound native class results to output-major labels without inventing decisions."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .claims import Label
from .errors import InvalidModelError


def _scalar(value: Any) -> Any:
    normalized = value.item() if isinstance(value, np.generic) else value
    if type(normalized) is str:
        if len(normalized) > 1024:
            raise ValueError
    elif type(normalized) not in (int, float, bool) or not math.isfinite(normalized):
        raise ValueError
    return normalized


def class_indices(
    raw: object, labels: Sequence[Sequence[Label]], rows: int
) -> list[list[int]]:
    try:
        values = np.asarray(raw, dtype=object)
        if len(labels) == 1 and values.shape == (rows,):
            values = values.reshape(rows, 1)
        if values.shape != (rows, len(labels)):
            raise ValueError
        mappings = [
            {label: index for index, label in enumerate(output)} for output in labels
        ]
        if any(
            len(mapping) != len(output)
            for mapping, output in zip(mappings, labels, strict=True)
        ):
            raise ValueError
        result = []
        for row in values:
            indices = []
            for value, mapping in zip(row, mappings, strict=True):
                indices.append(mapping[_scalar(value)])
            result.append(indices)
        return result
    except Exception:
        raise InvalidModelError(
            "Invalid native class prediction dimensions or labels."
        ) from None
