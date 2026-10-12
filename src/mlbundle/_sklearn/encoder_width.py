"""Validate fitted category/drop/grouping dimensions before deriving width."""

from typing import Any

import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from mlbundle.canonical import canonical_dumps

from ._width_checks import dimensions, indices, width


def encoder_width(model: Any, count: int) -> int:
    categories = model.categories_
    dimensions(isinstance(categories, list) and len(categories) == count)
    lengths = []
    for values in categories:
        dimensions(type(values) is np.ndarray)
        array = np.asarray(values)
        dimensions(array.ndim == 1 and 1 <= len(array) <= 100_000)
        encoded = [canonical_dumps(item) for item in array]
        dimensions(len(set(encoded)) == len(encoded))
        lengths.append(len(array))
    enabled = model._infrequent_enabled
    dimensions(
        type(enabled) is bool
        and enabled
        == (model.min_frequency is not None or model.max_categories is not None)
    )
    missing = model._missing_indices if type(model) is OrdinalEncoder else {}
    dimensions(
        isinstance(missing, dict)
        and all(type(i) is int and 0 <= i < count for i in missing)
    )
    for i, index in missing.items():
        dimensions(width(index, zero=True) == lengths[i] - 1)
        dimensions(bool(np.asarray(categories[i][-1] != categories[i][-1])))
    grouped = grouped_counts(model, lengths, missing, count)
    if type(model) is not OneHotEncoder:
        return count
    dropped_counts(model, lengths, grouped, count)
    cached = model._n_features_outs
    dimensions(isinstance(cached, list) and len(cached) == count)
    dimensions([width(item, zero=True) for item in cached] == grouped)
    return width(sum(grouped))


def grouped_counts(
    model: Any, lengths: list[int], missing: dict, count: int
) -> list[int]:
    grouped = lengths.copy()
    enabled = model._infrequent_enabled
    if enabled:
        infrequent = model._infrequent_indices
        mappings = model._default_to_infrequent_mappings
        dimensions(
            isinstance(infrequent, list)
            and isinstance(mappings, list)
            and len(infrequent) == len(mappings) == count
        )
        for i, (selection, mapping) in enumerate(
            zip(infrequent, mappings, strict=True)
        ):
            if selection is None:
                dimensions(mapping is None)
                continue
            size = lengths[i] - int(i in missing)
            selected = indices(selection, size)
            dimensions(bool(selected) and selected == sorted(selected))
            grouped[i] -= len(selected) - 1
            array = np.asarray(mapping)
            dimensions(array.shape == (size,) and array.dtype.kind in "iu")
            expected = np.empty(size, dtype=int)
            frequent = sorted(set(range(size)) - set(selected))
            expected[frequent] = np.arange(len(frequent))
            expected[selected] = len(frequent)
            dimensions(np.array_equal(array, expected))
    return grouped


def dropped_counts(
    model: Any, lengths: list[int], grouped: list[int], count: int
) -> None:
    enabled = model._infrequent_enabled
    drop = model.drop_idx_
    after = model._drop_idx_after_grouping
    dimensions((drop is None) == (after is None) == (model.drop is None))
    if drop is not None:
        dimensions(
            type(drop) is np.ndarray
            and type(after) is np.ndarray
            and drop.dtype.kind == after.dtype.kind == "O"
        )
        dimensions(np.asarray(drop).shape == np.asarray(after).shape == (count,))
        for i, (raw_original, raw_transformed) in enumerate(
            zip(drop, np.asarray(after), strict=True)
        ):
            dimensions((raw_original is None) == (raw_transformed is None))
            if raw_original is None:
                dimensions(model.drop == "if_binary" and grouped[i] != 2)
                continue
            original = width(raw_original, zero=True)
            transformed = width(raw_transformed, zero=True)
            dimensions(original < lengths[i] and transformed < grouped[i])
            mapping = model._default_to_infrequent_mappings[i] if enabled else None
            dimensions(
                transformed == (original if mapping is None else int(mapping[original]))
            )
            if type(model.drop) is str:
                dimensions(
                    transformed == 0
                    and (
                        model.drop == "first"
                        or (model.drop == "if_binary" and grouped[i] == 2)
                    )
                )
            else:
                dimensions(np.asarray(model.drop).shape == (count,))
                dimensions(
                    canonical_dumps(np.asarray(model.drop)[i])
                    == canonical_dumps(model.categories_[i][original])
                )
            grouped[i] -= 1
