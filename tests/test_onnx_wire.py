"""Batched scalar scans retain descriptor, duplicate, length and budget checks."""

import onnx
import pytest

from mlbundle import InvalidModelError, LimitExceededError
from mlbundle._onnx import protobuf

ATTRIBUTE = onnx.AttributeProto.DESCRIPTOR


def counted(data, descriptor=ATTRIBUTE):
    budget = [0]
    protobuf._message(data, 0, len(data), descriptor, depth=0, budget=budget)
    return budget[0]


@pytest.mark.parametrize(
    "unit",
    [
        b"\x40\x01",
        b"\x40\xff\x01",
        b"\x40" + b"\xff" * 9 + b"\x01",
        b"\x3d1234",
        b"\x4a\x04LEAF",
    ],
)
def test_unpacked_runs_count_across_chunks(unit):
    assert counted(unit * 40000) == 40000


@pytest.mark.parametrize(
    "tail",
    [
        b"\x40\x80",
        b"\x40" + b"\x80" * 10,
        b"\x40" + b"\xff" * 9 + b"\x02",
        b"\x3d123",
        b"\x4a\x05LEAF",
        b"\x3c",
        b"\xf8\x7f\x01",
        b"\x40\x01\x12\x00\x12\x00",
    ],
)
def test_invalid_tail_of_batched_fields_rejected(tail):
    with pytest.raises(InvalidModelError):
        protobuf.validate_wire(b"\x40\x01" * 40000 + tail, ATTRIBUTE)


def test_repeated_length_fields_preserve_boundaries_and_duplicates():
    # Equal bytes inside a string are content, never nested protobuf fields.
    assert counted(b"\x4a\x02\x40\x01" * 40000) == 40000
    assert counted(b"\x40\x01\x4a\x01a\x40\x02") == 3
    with pytest.raises(InvalidModelError, match="duplicate"):
        counted(b"\x0a\x01a" + b"\x40\x01" * 40000 + b"\x0a\x01b")


@pytest.mark.parametrize("wire", [b"\x42\x03\x01\x02\x03" * 3, b"\x3a\x041234" * 3])
def test_each_packed_header_and_scalar_counted(wire):
    assert counted(wire) == (12 if wire[0] == 66 else 6)


@pytest.mark.parametrize("unit", [b"\x40\x01", b"\x3d1234", b"\x4a\x04LEAF"])
def test_unpacked_batch_enforces_lowered_field_budget(unit, monkeypatch):
    monkeypatch.setattr(protobuf, "MAX_FIELDS", 100)
    with pytest.raises(LimitExceededError, match="field count"):
        protobuf.validate_wire(unit * 40000, ATTRIBUTE)


@pytest.mark.parametrize(
    "wire", [b"\x42\x02\x01\x80", b"\x42\x0a" + b"\xff" * 9 + b"\x02", b"\x3a\x03abc"]
)
def test_invalid_packed_tail_rejected(wire):
    with pytest.raises(InvalidModelError):
        protobuf.validate_wire(wire, ATTRIBUTE)


def test_noncanonical_tags_and_varints_have_same_uint64_boundary():
    assert counted((b"\xc0\x00" + b"\x80" * 9 + b"\x00") * 10000) == 10000


def test_large_strings_use_bounded_scalar_fallback():
    data = onnx.AttributeProto(
        name="strings", strings=[b"x" * 70000] * 2
    ).SerializeToString()
    assert counted(data) == 3


def test_short_alternating_fields_do_not_scan_overlapping_windows(monkeypatch):
    from mlbundle._onnx import _wire_runs

    scanned = []
    original = _wire_runs._varint_chunk

    def scan(data, start, end, tag):
        scanned.append(end - start)
        return original(data, start, end, tag)

    monkeypatch.setattr(_wire_runs, "_varint_chunk", scan)
    data = (b"\x08\x01\x38\x01") * 1000
    assert counted(data, onnx.TensorProto.DESCRIPTOR) == 2000
    assert not scanned
    # A run just beyond the probe must not speculate over the whole input.
    data = (b"\x08\x01" * 20 + b"\x38\x01" * 20) * 1000
    scanned.clear()
    assert counted(data, onnx.TensorProto.DESCRIPTOR) == 40000
    assert sum(scanned) <= 4 * len(data)
