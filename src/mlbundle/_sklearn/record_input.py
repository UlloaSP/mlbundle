"""Preserve fitted name-based selectors only when a record needs a DataFrame."""

from importlib import import_module
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

from mlbundle.errors import MissingOptionalDependencyError


def _named(columns: Any) -> bool:
    if isinstance(columns, str):
        return True
    if isinstance(columns, slice):
        return isinstance(columns.start, str) or isinstance(columns.stop, str)
    return isinstance(columns, (list, tuple, np.ndarray)) and any(
        isinstance(column, str) for column in columns
    )


def record_input(model: Any, row: list[float], names: list[str]) -> object:
    pending = [model]
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, ColumnTransformer):
            for _, step, columns in value.transformers_:
                if step != "drop" and _named(columns):
                    try:
                        pandas = import_module("pandas")
                    except ImportError:
                        raise MissingOptionalDependencyError("schema") from None
                    return pandas.DataFrame([row], columns=names)
                pending.append(step)
        elif isinstance(value, Pipeline):
            pending.extend(step for _, step in value.steps)
        elif isinstance(value, FeatureUnion):
            pending.extend(step for _, step in value.transformer_list)
    return [row]
