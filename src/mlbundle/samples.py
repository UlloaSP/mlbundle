"""Explicit feature-row selection and bounded pandas/NumPy JSON conversion."""

from __future__ import annotations

from datetime import date, datetime
from types import ModuleType
from typing import TYPE_CHECKING

from .canonical import canonical_dumps
from .errors import InvalidJSONError, InvalidSamplesError, LimitExceededError
from .limits import JSON_DEPTH

if TYPE_CHECKING:
    from pandas import DataFrame


def _value(value: object, pandas: ModuleType, depth: int = 0) -> object:
    if depth >= JSON_DEPTH:
        raise LimitExceededError("Sample nesting exceeds the depth limit.")
    if isinstance(value, dict):
        return {key: _value(item, pandas, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item, pandas, depth + 1) for item in value]
    if value is None or value is pandas.NA or value is pandas.NaT:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if (
        type(value).__module__.split(".")[0] == "numpy"
        and getattr(getattr(value, "dtype", None), "kind", None) == "M"
    ):
        import numpy as np  # noqa: PLC0415

        if isinstance(value, np.datetime64):
            return (
                None
                if np.isnat(value)
                else str(np.datetime_as_string(value, unit="auto"))
            )
        raise InvalidSamplesError("Unsupported sample value type.")
    if (
        isinstance(value, (str, bool, int, float))
        or type(value).__module__.split(".")[0] == "numpy"
    ):
        return None if bool(pandas.isna(value)) else value
    raise InvalidSamplesError("Unsupported sample value type.")


def sample_bytes(frame: DataFrame, count: int, cap: int) -> bytes:
    import pandas as pd  # noqa: PLC0415

    if type(count) is not int or count <= 0 or count > len(frame):
        raise InvalidSamplesError("sample_rows must be positive and within the frame.")
    content = bytearray(b"[")
    try:
        for values in frame.iloc[:count].itertuples(index=False, name=None):
            row = dict(zip(frame.columns, values, strict=True))
            encoded = canonical_dumps(
                _value(row, pd),
                max_bytes=cap - len(content) - 1,
                max_depth=JSON_DEPTH - 1,
            )
            if len(content) > 1:
                content.extend(b",")
            content.extend(encoded)
            if len(content) + 1 > cap:
                raise LimitExceededError("Samples exceed the byte limit.")
    except (InvalidJSONError, TypeError, ValueError):
        raise InvalidSamplesError("Unsupported sample value type.") from None
    content.extend(b"]")
    return bytes(content)
