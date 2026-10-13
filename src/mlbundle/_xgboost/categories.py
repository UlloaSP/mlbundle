"""Native categorical split bounds before XGBoost allocates split bitsets."""

from typing import Any

import numpy as np

from mlbundle.errors import InvalidModelError

from .arrays import array as _array

MAX_CATEGORY = 1_000_000


def validate_encodings(cats: dict[str, Any], width: int) -> None:
    encodings = cats["enc"]
    segments, sorted_indices = (
        _array(cats[key]) for key in ("feature_segments", "sorted_idx")
    )
    if not encodings:
        if segments.size or sorted_indices.size:
            raise InvalidModelError("Invalid XGBoost empty category encoding.")
        return
    if (
        len(encodings) != width
        or segments.shape != (width + 1,)
        or segments.dtype.kind not in "iu"
        or sorted_indices.ndim != 1
        or sorted_indices.dtype.kind not in "iu"
        or segments[0] != 0
        or segments[-1] != len(sorted_indices)
        or np.any(segments[1:] < segments[:-1])
    ):
        raise InvalidModelError("Invalid XGBoost category feature segments.")
    for index, encoding in enumerate(encodings):
        count = _encoding_count(encoding)
        if count > MAX_CATEGORY + 1 or segments[index + 1] - segments[index] != count:
            raise InvalidModelError("Invalid XGBoost category encoding dimensions.")
        order = sorted_indices[int(segments[index]) : int(segments[index + 1])]
        if not np.array_equal(np.sort(order), np.arange(count)):
            raise InvalidModelError("Invalid XGBoost category sort indices.")


def validate_categories(tree: dict[str, Any], nodes: int) -> int:
    default, split = (_array(tree[key]) for key in ("default_left", "split_type"))
    if any(
        value.dtype.kind not in "iu" or np.any(value > 1) or np.any(value < 0)
        for value in (default, split)
    ):
        raise InvalidModelError("Invalid XGBoost default direction or split type.")
    categories, indices, segments, sizes = (
        _array(tree[key])
        for key in (
            "categories",
            "categories_nodes",
            "categories_segments",
            "categories_sizes",
        )
    )
    if any(
        value.ndim != 1 or (value.size and value.dtype.kind not in "iu")
        for value in (categories, indices, segments, sizes)
    ):
        raise InvalidModelError("Invalid XGBoost categorical array types.")
    if (
        indices.shape != segments.shape
        or indices.shape != sizes.shape
        or len(set(indices.tolist())) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= nodes)
        or np.any(categories < 0)
        or np.any(categories > MAX_CATEGORY)
        or not np.array_equal(np.sort(indices), np.flatnonzero(split))
    ):
        raise InvalidModelError("Invalid XGBoost categorical node storage.")
    end = storage = 0
    for start, size in zip(segments, sizes, strict=True):
        if start != end or size < 1 or size > len(categories) - end:
            raise InvalidModelError("Invalid XGBoost categorical segments.")
        stop = end + int(size)
        storage += (int(max(categories[end:stop])) // 32 + 1) * 4
        end = stop
    if end != len(categories):
        raise InvalidModelError("Unreferenced XGBoost categorical values.")
    return storage


def _encoding_count(encoding: dict[str, Any]) -> int:
    values = _array(encoding["values"])
    if values.ndim != 1 or (values.size and values.dtype.kind not in "biuf"):
        raise InvalidModelError("Invalid XGBoost category encoding values.")
    count = len(values)
    if "offsets" in encoding:
        offsets = _array(encoding["offsets"])
        if offsets.size:
            if (
                offsets.ndim != 1
                or offsets.dtype.kind not in "iu"
                or offsets[0] != 0
                or offsets[-1] != len(values)
                or np.any(offsets[1:] < offsets[:-1])
                or values.dtype.kind not in "iu"
                or np.any(values < 0)
                or np.any(values > 255)
            ):
                raise InvalidModelError("Invalid XGBoost category string offsets.")
        elif values.size:
            raise InvalidModelError("Invalid XGBoost empty category string offsets.")
        count = max(0, len(offsets) - 1)
    return count
