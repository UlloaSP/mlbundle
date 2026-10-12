"""Compare the optimized codec with the format-1 implementation at 13ce639."""

import json
import math
import random
import struct

import numpy as np
import pytest

import canonical_reference as old
from mlbundle import canonical
from mlbundle.errors import InvalidJSONError, LimitExceededError


def outcome(function, value, **limits):
    try:
        result = function(value, **limits)
        if isinstance(result, float):
            result = struct.pack("!d", result)
        return result
    except (InvalidJSONError, LimitExceededError) as error:
        return type(error)


def generated(rng, depth=0):
    atoms = [
        None,
        True,
        False,
        0,
        -1,
        10**80,
        -0.0,
        5e-324,
        math.inf,
        math.nan,
        "",
        "é",
        '\n\t\\"[{}]',
        "😀",
        "\u0000",
    ]
    if depth == 4 or rng.randrange(3) == 0:
        return rng.choice(atoms)
    children = [generated(rng, depth + 1) for _ in range(rng.randrange(5))]
    if rng.randrange(2):
        return children
    return {str(index): value for index, value in enumerate(children)}


def test_generated_values_and_wire_spellings_match():
    rng = random.Random(9)
    for _ in range(400):
        value = generated(rng)
        encoded = old.canonical_dumps(value)
        assert canonical.canonical_dumps(value) == encoded
        decoded = old.canonical_loads(encoded)
        for wire in (encoded, json.dumps(decoded, ensure_ascii=True).encode()):
            assert canonical.decode_json(wire).canonical == encoded
            assert canonical.canonical_loads(wire) == decoded
        for limits in ({"max_bytes": max(1, len(encoded) - 1)}, {"max_depth": 2}):
            assert outcome(canonical.canonical_dumps, value, **limits) == outcome(
                old.canonical_dumps, value, **limits
            )
            assert outcome(canonical.canonical_loads, encoded, **limits) == outcome(
                old.canonical_loads, encoded, **limits
            )


@pytest.mark.parametrize(
    "value",
    [
        [0.0] * 64,
        [math.inf] * 64,
        [math.nan] * 64,
        [10**4095] * 64,
        [10**4096] * 64,
        [True, None, 1, 2.0] * 32,
        tuple(range(64)),
        [np.float32(0.1)] * 64,
        np.array([[1.0, math.inf], [-0.0, math.nan]]),
        np.array([1, "é", None], dtype=object),
        np.array([np.datetime64("NaT")], dtype=object),
        {1: "a"},
        {"$float": []},
        object(),
        b"bytes",
        "\udfff",
    ],
)
def test_native_success_and_rejections_match(value):
    for limits in ({}, {"max_bytes": 32}, {"max_depth": 1}):
        assert outcome(canonical.canonical_dumps, value, **limits) == outcome(
            old.canonical_dumps, value, **limits
        )


@pytest.mark.parametrize(
    "wire",
    [
        b'{"a":1,"a":2}',
        b'{"x":[{"a":1,"a":2}]}',
        b'{"$float":"nan","x":0}',
        b'{"$float":false}',
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"1e309",
        b"-1e309",
        b'"\\ud800"',
        b'"\\udfff"',
        b'"\\ud800\\u0041"',
        b'"\\ud800x\\udc00"',
        b'"\\ud800\\ud800"',
        b'"\\udc00\\ud800"',
        b'"\\ud83d\\ude00"',
        b'"\\\\ud800"',
        b'"\\\\\\ud800"',
        b'{"\\ud800":1}',
        b'["\\udfff"]',
        b'"\\uXXXX"',
        b'"unfinished\\',
        b'"unfinished[[[[',
        b'"\\"[[[["',
        b'"a\\\\"[[[[',
        b"[[[0]]]",
        b"0e0",
        b"1e2",
        b"1e-5",
        '"é"'.encode(),
        b"1" + b"0" * 4096,
        b"0." + b"0" * 5000 + b"1",
        b"\xff",
    ],
)
def test_wire_success_and_rejections_match(wire):
    for limits in ({}, {"max_bytes": len(wire)}, {"max_depth": 2}):
        assert outcome(canonical.canonical_loads, wire, **limits) == outcome(
            old.canonical_loads, wire, **limits
        )


def test_numeric_bulk_paths_keep_work_and_type_bounds(monkeypatch):
    class Integer(int):
        pass

    class Sequence(list):
        pass

    class Mapping(dict):
        def items(self):
            return [("a", 3)]

    for value in ([Integer(1)] * 64, Sequence([0.0] * 64), Mapping(a=2)):
        assert outcome(canonical.canonical_dumps, value) == outcome(
            old.canonical_dumps, value
        )
    monkeypatch.setattr(canonical, "_normalize", lambda *args: pytest.fail("allocated"))
    for value in ([0.0] * 64, np.zeros(64), np.empty((1,) * 4)):
        with pytest.raises(LimitExceededError):
            canonical.canonical_dumps(value, max_bytes=32, max_depth=2)


def test_decode_depth_is_checked_before_json_parser(monkeypatch):
    monkeypatch.setattr(
        canonical.json, "loads", lambda *args, **kw: pytest.fail("parsed")
    )
    with pytest.raises(LimitExceededError):
        canonical.canonical_loads(b"[[[0]]]", max_depth=2)


def test_deferred_comparison_is_bounded_and_exact(monkeypatch):
    decoded = canonical.decode_json(b' {"z":1e2,"a":"\\ud83d\\ude00"} ')
    assert decoded.canonical == '{"a":"😀","z":100.0}'.encode()
    with pytest.raises(LimitExceededError):
        canonical.decode_json(b"1e2", max_bytes=3)
    monkeypatch.setattr(canonical, "_encode_json", lambda *args: pytest.fail("encoded"))
    assert canonical.canonical_loads(b"[1.0,2.0]") == [1.0, 2.0]


@pytest.mark.parametrize(
    "wire",
    [
        b"[" + b",".join([b"1e400"] * 64) + b"]",
        b"[" + b",".join([b"-1e400"] * 64) + b"]",
        b"[" + b",".join([b"1", b"1e400"] * 32) + b"]",
        b"[" + b",".join([b"[1e400]"] * 64) + b"]",
        b"[" + b",".join([b"1.0"] * 64) + b"]",
        b"[" + b",".join([b"1", b"null", b'"a"'] * 32) + b"]",
    ],
)
def test_decoded_bulk_finiteness_matches(wire):
    assert outcome(canonical.canonical_loads, wire) == outcome(
        old.canonical_loads, wire
    )
