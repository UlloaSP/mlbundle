"""Cross-check ColumnTransformer selectors, fitted clones and output slices."""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from mlbundle.canonical import canonical_dumps
from mlbundle.errors import InvalidModelError

from ._width_checks import width
from .parameters import parameters


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidModelError(message)


def selector(value: Any, count: int, names: list[str] | None) -> list[int]:
    message = "Invalid ColumnTransformer input mapping."

    def position(item: Any) -> int:
        if type(item) in (str, np.str_):
            require(names is not None and item in names, message)
            return (names or []).index(item)
        require(
            isinstance(item, (int, np.integer))
            and not isinstance(item, (bool, np.bool_)),
            message,
        )
        result = int(item)
        result = result + count if result < 0 else result
        require(0 <= result < count, message)
        return result

    if isinstance(value, slice):
        if type(value.start) is str or type(value.stop) is str:
            require(value.step is None and names is not None, message)
            start = 0 if value.start is None else position(value.start)
            stop = count if value.stop is None else position(value.stop) + 1
            return list(range(start, stop))
        require(
            all(
                item is None or type(item) is int
                for item in (value.start, value.stop, value.step)
            )
            and value.step != 0,
            message,
        )
        return list(range(count))[value]
    if type(value) in (str, np.str_) or isinstance(value, (int, np.integer)):
        return [position(value)]
    require(isinstance(value, (list, tuple, np.ndarray)), message)
    array = np.asarray(value)
    require(array.ndim == 1 and len(array) <= 100_000, message)
    if array.dtype.kind == "b":
        require(len(array) == count, message)
        return np.flatnonzero(array).tolist()
    return [position(item) for item in value]


def same_transformer(declared: Any, fitted: Any) -> bool:
    if type(declared) is str:
        if declared == "drop":
            return type(fitted) is str and fitted == "drop"
        return (
            declared == "passthrough"
            and type(fitted) is FunctionTransformer
            and fitted.func is None
            and fitted.inverse_func is None
            and fitted.feature_names_out == "one-to-one"
            and fitted.validate is False
        )
    return type(declared) is type(fitted) and canonical_dumps(
        parameters(declared)
    ) == canonical_dumps(parameters(fitted))


def column_width(
    model: Any, count: int, depth: int, output_width: Callable[..., int]
) -> int:

    mapping_error = "Invalid ColumnTransformer input mapping."
    clone_error = "ColumnTransformer declared and fitted transformers disagree."
    output_error = "Invalid ColumnTransformer output mapping."
    declared = model.transformers
    fitted = model.transformers_
    require(
        isinstance(declared, (list, tuple))
        and isinstance(fitted, list)
        and len(declared) <= 128
        and len(model._columns) == len(declared),
        mapping_error,
    )
    require(
        all(
            isinstance(item, (tuple, list)) and len(item) == 3
            for item in [*declared, *fitted]
        ),
        mapping_error,
    )
    declared_names = [item[0] for item in declared]
    require(
        all(
            type(name) is str
            and 1 <= len(name) <= 256
            and name != "remainder"
            and "__" not in name
            for name in declared_names
        )
        and len(set(declared_names)) == len(declared_names),
        mapping_error,
    )
    names = getattr(model, "feature_names_in_", None)
    names = None if names is None else list(names)
    mapping = model._transformer_to_input_indices
    require(
        isinstance(mapping, dict) and set(mapping) == {*declared_names, "remainder"},
        mapping_error,
    )
    for selected in mapping.values():
        require(
            isinstance(selected, list)
            and all(type(i) is int and 0 <= i < count for i in selected),
            mapping_error,
        )
    used = set()
    for (name, _, columns), cached in zip(declared, model._columns, strict=True):
        resolved = selector(columns, count, names)
        require(
            resolved == selector(cached, count, names) and resolved == mapping[name],
            mapping_error,
        )
        used.update(resolved)
    remaining = sorted(set(range(count)) - used)
    require(mapping["remainder"] == remaining, mapping_error)
    remainder = model._remainder
    require(
        isinstance(remainder, tuple)
        and len(remainder) == 3
        and remainder[0] == "remainder"
        and selector(remainder[2], count, names) == remaining,
        mapping_error,
    )
    left, right = model.remainder, remainder[1]
    require(
        type(left) is type(right)
        and (
            canonical_dumps(left) == canonical_dumps(right)
            if type(left) is str
            else canonical_dumps(parameters(left)) == canonical_dumps(parameters(right))
        ),
        clone_error,
    )
    expected = [*declared, *((remainder,) if remaining else ())]
    require(len(fitted) == len(expected), clone_error)
    require(type(model.sparse_output_) is bool, output_error)
    outputs = model.output_indices_
    require(
        isinstance(outputs, dict) and set(outputs) == {*declared_names, "remainder"},
        output_error,
    )
    total = 0
    for (name, original, columns), (actual_name, transformer, actual_columns) in zip(
        expected, fitted, strict=True
    ):
        require(
            name == actual_name
            and selector(columns, count, names)
            == selector(actual_columns, count, names),
            mapping_error,
        )
        selected = mapping[name]
        if not selected:
            require(
                type(original) is type(transformer)
                or same_transformer(original, transformer),
                clone_error,
            )
            require(outputs[name] == slice(0, 0), output_error)
            continue
        require(same_transformer(original, transformer), clone_error)
        if type(transformer) is str and transformer == "drop":
            require(outputs[name] == slice(0, 0), output_error)
            continue
        require(
            not isinstance(columns, (str, np.str_, int, np.integer))
            and not isinstance(actual_columns, (str, np.str_, int, np.integer)),
            mapping_error,
        )
        result = output_width(transformer, len(selected), depth=depth + 1)
        require(outputs[name] == slice(total, total + result), output_error)
        total = width(total + result)
    if not remaining:
        require(outputs["remainder"] == slice(0, 0), output_error)
    return width(total)
