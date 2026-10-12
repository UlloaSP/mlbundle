"""Canonical UTF-8 JSON with explicit non-finite numbers and bounded parsing."""

import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from pydantic import JsonValue

from ._canonical_work import bound_work
from .errors import InvalidJSONError, LimitExceededError
from .limits import JSON_DEPTH, MANIFEST_BYTES

_FLOAT_TAG = "$float"
_INTEGER_CEILING = 10**4096
_DELIMITERS = re.compile(r'["\[\]{}]')
_STRING = re.compile(r'"[^"\\]*+(?:\\.[^"\\]*+)*+"?', re.DOTALL)
_ESCAPES = re.compile(r"\\(?:u([a-fA-F0-9]{4})|.)", re.DOTALL)


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
        return _sequence(value, depth + 1, max_depth)
    return _numpy(value, depth, max_depth)


def _sequence(value, depth: int, max_depth: int) -> JsonValue:
    if len(value) >= 64:
        kinds = set(map(type, value))
        if (kinds == {float} and all(map(math.isfinite, value))) or (
            kinds == {int} and max(map(abs, value)) < _INTEGER_CEILING
        ):
            return value if type(value) is list else list(value)
    return list(_items(value, depth, max_depth))


def _items(values, depth: int, max_depth: int):
    # Keep flat native numeric arrays out of the recursive dispatch path.
    for item in values:
        kind = type(item)
        if (
            (kind is float and math.isfinite(item))
            or item is None
            or kind in (bool, str)
            or (kind is int and abs(item) < _INTEGER_CEILING)
        ):
            yield item
        else:
            yield _normalize(item, depth, max_depth)


def canonical_dumps(
    value: object, *, max_bytes: int = MANIFEST_BYTES, max_depth: int = JSON_DEPTH
) -> bytes:
    bound_work(value, max_bytes, max_depth)
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


def _check_surrogates(text: str, start: int, end: int) -> None:
    pending = None
    for match in _ESCAPES.finditer(text, start, end):
        token = match[1]
        code = int(token, 16) if token is not None else 0
        if pending is not None:
            if match.start() != pending or not 0xDC00 <= code <= 0xDFFF:
                raise InvalidJSONError
            pending = None
        elif 0xD800 <= code <= 0xDBFF:
            pending = match.end()
        elif 0xDC00 <= code <= 0xDFFF:
            raise InvalidJSONError
    if pending is not None:
        raise InvalidJSONError


def _check_depth(text: str, max_depth: int) -> None:
    depth = 0
    valid_unicode = True
    # The C regex skips complete strings and numeric runs; only containers
    # need Python work. Possessive runs keep malformed strings linear too.
    position = 0
    while match := _DELIMITERS.search(text, position):
        character = text[match.start()]
        position = match.end()
        if character in "[{":
            depth += 1
            if depth > max_depth:
                raise LimitExceededError("JSON nesting exceeds the depth limit.")
        elif character in "]}":
            depth -= 1
        else:
            string = _STRING.match(text, match.start())
            assert string is not None
            position = string.end()
            if text.find(r"\u", string.start(), position) != -1:
                valid_unicode &= _valid_surrogates(text, string.start(), position)
    if not valid_unicode:
        raise InvalidJSONError


def _valid_surrogates(text: str, start: int, end: int) -> bool:
    try:
        _check_surrogates(text, start, end)
    except InvalidJSONError:
        return False
    return True


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


def _constant(_token: str) -> JsonValue:
    raise InvalidJSONError


def _integer(token: str) -> int:
    if len(token.lstrip("-")) > 4096:
        raise InvalidJSONError
    return int(token)


def _finite_array(value: list) -> bool:
    if len(value) < 64:
        return False
    kinds = set(map(type, value))
    if kinds == {float}:
        if not all(map(math.isfinite, value)):
            raise InvalidJSONError
        return True
    return not kinds.intersection((float, list, dict))


def _check_finite(value) -> None:
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidJSONError
        return
    if not isinstance(value, (list, dict)):
        return
    if isinstance(value, list) and _finite_array(value):
        return
    children = value.values() if isinstance(value, dict) else value
    for child in children:
        if type(child) is float:
            if not math.isfinite(child):
                raise InvalidJSONError
        elif isinstance(child, (list, dict)):
            _check_finite(child)


def _encode_json(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _bounded_encoding(
    value: JsonValue, stored_bytes: int, max_bytes: int
) -> bytes | None:
    # Only float spelling can grow: <=24 canonical bytes per >=3 token bytes.
    if stored_bytes * 8 <= max_bytes:
        _check_finite(value)
        return None
    encoded = _encode_json(value)
    if len(encoded) > max_bytes:
        raise LimitExceededError("JSON exceeds the byte limit.")
    return encoded


@dataclass(frozen=True)
class DecodedJSON:
    """Operation-local decoded value and its checked canonical comparison bytes."""

    value: JsonValue
    _encoded: bytes | None

    @cached_property
    def canonical(self) -> bytes:
        return self._encoded if self._encoded is not None else _encode_json(self.value)


def decode_json(
    data: bytes, *, max_bytes: int = MANIFEST_BYTES, max_depth: int = JSON_DEPTH
) -> DecodedJSON:
    if len(data) > max_bytes:
        raise LimitExceededError("JSON exceeds the byte limit.")
    try:
        text = data.decode("utf-8")
        _check_depth(text, max_depth)
        value = json.loads(
            text,
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_int=_integer,
        )
        # Stored byte/depth bounds also bound nodes/strings and expansion:
        # wire JSON has no shared references. Retain encoding only when needed.
        encoded = _bounded_encoding(value, len(data), max_bytes)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise InvalidJSONError from None
    return DecodedJSON(cast(JsonValue, value), encoded)


def canonical_loads(
    data: bytes, *, max_bytes: int = MANIFEST_BYTES, max_depth: int = JSON_DEPTH
) -> JsonValue:
    return decode_json(data, max_bytes=max_bytes, max_depth=max_depth).value


def canonical_value(
    value: object, *, max_bytes: int = MANIFEST_BYTES, max_depth: int = JSON_DEPTH
) -> JsonValue:
    """Project native scalars/arrays to the same bounded JSON used for comparison."""
    return canonical_loads(
        canonical_dumps(value, max_bytes=max_bytes, max_depth=max_depth),
        max_bytes=max_bytes,
        max_depth=max_depth,
    )
