"""Canonical numbers, numpy normalization and hostile JSON parsing."""

import math
import struct

import numpy as np
import pytest

from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.errors import InvalidJSONError, LimitExceededError


def test_determinism_and_unicode():
    assert canonical_dumps({"z": 0, "a": "á"}) == '{"a":"á","z":0}'.encode()
    assert canonical_dumps({"z": 0, "a": "á"}) == canonical_dumps({"a": "á", "z": 0})
    assert canonical_loads(b' { "z" : 0 } ') == {"z": 0}


@pytest.mark.parametrize(
    "number",
    [0.0, -0.0, 0.1, 1.2345678901234567, 5e-324, 1.7976931348623157e308, 1e-7, 1e20],
)
def test_float_round_trip(number):
    result = canonical_loads(canonical_dumps(number))
    assert struct.pack("!d", result) == struct.pack("!d", number)


def test_nonfinite_and_numpy():
    result = canonical_loads(canonical_dumps([math.nan, math.inf, -math.inf]))
    assert result == [{"$float": "nan"}, {"$float": "+inf"}, {"$float": "-inf"}]
    assert canonical_dumps(result) == canonical_dumps([math.nan, math.inf, -math.inf])
    values = {
        "ints": np.array([[1, 2]], dtype=np.int64),
        "real": np.float32(0.1),
        "bool": np.bool_(True),
    }
    assert canonical_loads(canonical_dumps(values)) == {
        "ints": [[1, 2]],
        "real": float(np.float32(0.1)),
        "bool": True,
    }


@pytest.mark.parametrize(
    "data",
    [
        b'{"a":1,"a":2}',
        b'{"x":{"a":1,"a":2}}',
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"1e999",
        b'{"$float":"wrong"}',
        b'{"$float":[]}',
        b'{"$float":"nan","x":1}',
        b'"\\ud800"',
        b"\xff",
        b"{",
    ],
)
def test_invalid_json(data):
    with pytest.raises(InvalidJSONError):
        canonical_loads(data)


@pytest.mark.parametrize(
    "value",
    [
        object(),
        {1: "value"},
        {"$float": "wrong"},
        complex(1, 2),
        np.complex128(1),
        "\ud800",
    ],
)
def test_invalid_encoding(value):
    with pytest.raises(InvalidJSONError):
        canonical_dumps(value)


def test_depth_bounds_before_parse_and_on_encode():
    data = b"[" * 1000 + b"0" + b"]" * 1000
    with pytest.raises(LimitExceededError):
        canonical_loads(data, max_depth=4)
    for value in ([[[]]], [[0]]):
        with pytest.raises(LimitExceededError):
            canonical_dumps(value, max_depth=1)
    assert canonical_loads(b'["[\\"{}]",{}]', max_depth=2) == ['["{}]', {}]
    assert canonical_loads(b"[[]]", max_depth=2) == [[]]
    assert canonical_dumps([[]], max_depth=2) == b"[[]]"


def test_size_bounds():
    with pytest.raises(LimitExceededError):
        canonical_loads(b'"long"', max_bytes=3)
    with pytest.raises(LimitExceededError):
        canonical_dumps("long", max_bytes=3)


def test_cycles_are_bounded():
    value = []
    value.append(value)
    with pytest.raises(LimitExceededError):
        canonical_dumps(value)


def test_integer_digit_bound():
    assert canonical_loads(canonical_dumps(10**4095)) == 10**4095
    with pytest.raises(InvalidJSONError):
        canonical_loads(b"1" + b"0" * 4096)
    with pytest.raises(InvalidJSONError):
        canonical_dumps(10**4096)


@pytest.mark.parametrize(
    "value",
    [
        np.datetime64("2024-01-01", "ns"),
        np.datetime64("NaT"),
        np.timedelta64(1, "ns"),
        np.timedelta64("NaT"),
        np.array(["2024-01-01"], dtype="datetime64[ns]"),
        np.array([1], dtype="timedelta64[ns]"),
        np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")]),
    ],
)
def test_numpy_semantics_are_not_erased(value):
    with pytest.raises(InvalidJSONError):
        canonical_dumps(value)


def test_numpy_object_arrays_preserve_valid_scalar_types():
    assert canonical_loads(canonical_dumps(np.array([1, "a", None], dtype=object))) == [
        1,
        "a",
        None,
    ]
    with pytest.raises(InvalidJSONError):
        canonical_dumps(np.array([np.datetime64("NaT")], dtype=object))


def test_tag_depth_and_malformed_unicode_escapes():
    with pytest.raises(LimitExceededError):
        canonical_dumps([math.inf], max_depth=1)
    with pytest.raises(InvalidJSONError):
        canonical_loads(b'"\\uXXXX"')


def test_shared_json_expansion_is_bounded_before_normalization(monkeypatch):
    from mlbundle import canonical

    value = [0]
    for _ in range(25):
        value = [value, value]
    monkeypatch.setattr(
        canonical,
        "_normalize",
        lambda *args: pytest.fail("Expansion must be rejected before normalization"),
    )
    with pytest.raises(LimitExceededError, match="Expanded JSON"):
        canonical_dumps(value, max_bytes=1024)


def test_shared_json_within_budget_is_preserved():
    child = [1, 2]
    assert canonical_dumps([child, child]) == b"[[1,2],[1,2]]"
