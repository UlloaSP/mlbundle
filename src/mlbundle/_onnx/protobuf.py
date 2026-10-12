"""Wire bounds before protobuf constructs objects; no tensor decoding or file I/O."""

from typing import Any

from mlbundle.errors import InvalidModelError, LimitExceededError
from mlbundle.limits import ArchiveLimits

from ._wire_runs import packed_varints, repeated

# Each field/header or packed scalar consumes at least one distinct wire byte.
MAX_FIELDS = ArchiveLimits().max_member_bytes
MAX_DEPTH = 32


def _varint(data: bytes, position: int, end: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if position >= end:
            raise InvalidModelError("Truncated ONNX protobuf.")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if shift == 63 and byte > 1:
                break
            return value, position
    raise InvalidModelError("Invalid ONNX protobuf integer.")


def validate_wire(data: bytes, descriptor: Any) -> None:
    if not data:
        raise InvalidModelError("Empty ONNX model.")
    if len(data) > ArchiveLimits().max_member_bytes:
        raise LimitExceededError("ONNX model exceeds the byte limit.")
    budget = [0]
    _message(data, 0, len(data), descriptor, depth=0, budget=budget)


def _message(data, position, end, descriptor, *, depth, budget):
    if depth > MAX_DEPTH:
        raise LimitExceededError("ONNX protobuf nesting exceeds the limit.")
    seen = set()
    while position < end:
        budget[0] += 1
        if budget[0] > MAX_FIELDS:
            raise LimitExceededError("ONNX protobuf field count exceeds the limit.")
        start = position
        tag, position = _varint(data, position, end)
        number, wire = tag >> 3, tag & 7
        field = descriptor.fields_by_number.get(number)
        if field is None or (not field.is_repeated and number in seen):
            raise InvalidModelError("Unknown or duplicate ONNX protobuf field.")
        seen.add(number)
        value_start = position
        position = _field(data, position, end, field, wire, depth=depth, budget=budget)
        if field.is_repeated and (wire != 2 or field.type in (9, 12)):
            position, count = repeated(
                data, start, end, value_start=value_start, stop=position, wire=wire
            )
            budget[0] += count - 1
            if budget[0] > MAX_FIELDS:
                raise LimitExceededError("ONNX protobuf field count exceeds the limit.")


def _field(data, position, end, field, wire, *, depth, budget):
    expected = (
        2
        if field.type in (9, 11, 12)
        else {1: 1, 2: 5, 6: 1, 7: 5, 15: 5, 16: 1}.get(field.type, 0)
    )
    packed = field.is_repeated and expected != 2 and wire == 2
    if wire != expected and not packed:
        raise InvalidModelError("Invalid ONNX protobuf wire type.")
    if wire == 0:
        return _varint(data, position, end)[1]
    if wire in (1, 5):
        stop = position + (8 if wire == 1 else 4)
    else:
        size, position = _varint(data, position, end)
        stop = position + size
    if stop > end:
        raise InvalidModelError("Truncated ONNX protobuf field.")
    if field.type == 11:
        _message(
            data, position, stop, field.message_type, depth=depth + 1, budget=budget
        )
    elif packed:
        _packed(data, position, stop, expected, budget)
    return stop


def _packed(data, position, end, wire, budget):
    if wire in (1, 5):
        size = 8 if wire == 1 else 4
        if (end - position) % size:
            raise InvalidModelError("Invalid packed ONNX protobuf field.")
        budget[0] += (end - position) // size
    else:
        while position < end:
            stop, count = packed_varints(data, position, end)
            if not count:
                stop = _varint(data, position, end)[1]
                count = 1
            position = stop
            budget[0] += count
            if budget[0] > MAX_FIELDS:
                raise LimitExceededError(
                    "ONNX protobuf scalar count exceeds the limit."
                )
    if budget[0] > MAX_FIELDS:
        raise LimitExceededError("ONNX protobuf scalar count exceeds the limit.")
