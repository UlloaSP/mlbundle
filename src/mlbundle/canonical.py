"""Canonical UTF-8 JSON with explicit non-finite numbers and bounded parsing."""

import json
import math
from typing import cast

from pydantic import JsonValue

from .errors import InvalidJSONError, LimitExceededError
from .limits import JSON_DEPTH, MANIFEST_BYTES

_FLOAT_TAG = "$float"
_INTEGER_CEILING = 10**4096


def _float(value: float, depth: int, max_depth: int) -> JsonValue:
    if math.isfinite(value):
        return value
    if depth >= max_depth:
        raise LimitExceededError("JSON nesting exceeds the depth limit.")
    token = "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
    return {_FLOAT_TAG: token}


def _numpy(value: object, depth: int, max_depth: int) -> JsonValue:
    # Writer-side numpy conversion needs no import in the base package.
    if type(value).__module__.split(".")[0] == "numpy":
        array = type(value).__name__ == "ndarray"
        kind = getattr(getattr(value, "dtype", None), "kind", None)
        if kind not in ("b", "i", "u", "f", "U") and not (array and kind == "O"):
            raise InvalidJSONError
        method = "tolist" if array else "item"
        convert = getattr(value, method, None)
        if callable(convert):
            converted = convert()
            if type(converted) is not type(value):
                return _normalize(converted, depth, max_depth)
    raise InvalidJSONError


def _normalize(value: object, depth: int, max_depth: int) -> JsonValue:
    if isinstance(value, (dict, list, tuple)) and depth >= max_depth:
        raise LimitExceededError("JSON nesting exceeds the depth limit.")
    if type(value) is int and abs(value) >= _INTEGER_CEILING:
        raise InvalidJSONError
    if value is None or type(value) in (bool, int, str):
        return cast(JsonValue, value)
    if type(value) is float:
        return _float(value, depth, max_depth)
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise InvalidJSONError
        if _FLOAT_TAG in value and (
            len(value) != 1 or value[_FLOAT_TAG] not in ("nan", "+inf", "-inf")
        ):
            raise InvalidJSONError
        return {
            key: _normalize(item, depth + 1, max_depth) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, depth + 1, max_depth) for item in value]
    return _numpy(value, depth, max_depth)


def canonical_dumps(
    value: object, *, max_bytes: int = MANIFEST_BYTES, max_depth: int = JSON_DEPTH
) -> bytes:
    try:
        encoded = json.dumps(
            _normalize(value, 0, max_depth),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise InvalidJSONError from None
    if len(encoded) > max_bytes:
        raise LimitExceededError("JSON exceeds the byte limit.")
    return encoded


def _check_depth(text: str, max_depth: int) -> None:
    depth = 0
    quoted = escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > max_depth:
                raise LimitExceededError("JSON nesting exceeds the depth limit.")
        elif character in "]}":
            depth -= 1


def _object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidJSONError
        result[key] = value
    if _FLOAT_TAG in result and (
        len(result) != 1 or result[_FLOAT_TAG] not in ("nan", "+inf", "-inf")
    ):
        raise InvalidJSONError
    return result


def _finite_float(token: str) -> float:
    result = float(token)
    if not math.isfinite(result):
        raise InvalidJSONError
    return result


def _constant(_token: str) -> JsonValue:
    raise InvalidJSONError


def _integer(token: str) -> int:
    if len(token.lstrip("-")) > 4096:
        raise InvalidJSONError
    return int(token)


def canonical_loads(
    data: bytes, *, max_bytes: int = MANIFEST_BYTES, max_depth: int = JSON_DEPTH
) -> JsonValue:
    if len(data) > max_bytes:
        raise LimitExceededError("JSON exceeds the byte limit.")
    try:
        text = data.decode("utf-8")
        _check_depth(text, max_depth)
        value = json.loads(
            text,
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_float=_finite_float,
            parse_int=_integer,
        )
        # Reject invalid Unicode and enforce the encoder's bounds too.
        canonical_dumps(value, max_bytes=max_bytes, max_depth=max_depth)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise InvalidJSONError from None
    return cast(JsonValue, value)
