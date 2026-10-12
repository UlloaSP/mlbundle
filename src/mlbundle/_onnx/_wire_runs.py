"""Bounded C regex scans of repeated wire scalars; no protobuf objects allocated."""

import re
from functools import lru_cache

import numpy as np

# Match exactly the same uint64 encodings accepted by the scalar walker,
# including overlong encodings; byte ten may contain only bit zero.
VARINT = rb"(?:[\x00-\x7f]|[\x80-\xff]{1,8}[\x00-\x7f]|[\x80-\xff]{9}[\x00-\x01])"
CHUNK_BYTES = 64 * 1024
PACKED = re.compile(rb"(?:" + VARINT + rb")+")


@lru_cache(maxsize=128)
def _pattern(tag: bytes, wire: int, value: bytes):
    scalar = VARINT if wire == 0 else rb".{%d}" % (8 if wire == 1 else 4)
    unit = re.escape(tag) + scalar if wire != 2 else re.escape(value)
    return re.compile(rb"(?:" + unit + rb")+", re.DOTALL)


def _varint_chunk(data: bytes, start: int, end: int, tag: bytes):
    values = np.frombuffer(data, dtype=np.uint8, count=end - start, offset=start)
    terminals = np.flatnonzero(values < 128)
    count = len(terminals) // 2
    if not count:
        return start, 0
    pairs = terminals[: count * 2].reshape(-1, 2)
    starts = np.concatenate(([0], pairs[:-1, 1] + 1))
    valid = pairs[:, 0] == starts + len(tag) - 1
    for offset, byte in enumerate(tag):
        # Clipping is only for malformed, too-short tags; their offset check fails.
        valid &= values[np.minimum(starts + offset, len(values) - 1)] == byte
    widths = pairs[:, 1] - pairs[:, 0]
    valid &= (widths <= 10) & ((widths != 10) | (values[pairs[:, 1]] <= 1))
    invalid = np.flatnonzero(~valid)
    count = int(invalid[0]) if len(invalid) else count
    return start + int(pairs[count - 1, 1]) + 1 if count else start, count


def _fixed_chunk(data: bytes, start: int, end: int, tag: bytes, wire: int):
    size = len(tag) + (8 if wire == 1 else 4)
    count = (end - start) // size
    values = np.frombuffer(
        data, dtype=np.uint8, count=count * size, offset=start
    ).reshape(-1, size)
    valid = np.all(values[:, : len(tag)] == np.frombuffer(tag, dtype=np.uint8), axis=1)
    invalid = np.flatnonzero(~valid)
    count = int(invalid[0]) if len(invalid) else count
    return start + count * size, count


def repeated(
    data: bytes, start: int, end: int, *, value_start: int, stop: int, wire: int
):
    """Include the already validated first field, then scan equivalent wire units."""
    if wire == 2 and stop - start > 4096:
        return stop, 1
    tag = data[start:value_start]
    pattern = _pattern(tag, wire, data[start:stop] if wire == 2 else b"")
    probe = pattern.match(data, start, min(end, start + 64))
    vectorize = wire != 2 and probe is not None and probe.end() - start >= 32
    chunk_bytes = 128 if vectorize else CHUNK_BYTES
    count, position = 0, start
    while position < end and data.startswith(tag, position, end):
        chunk_end = min(end, position + chunk_bytes)
        if vectorize:
            next_position, units = (
                _varint_chunk(data, position, chunk_end, tag)
                if wire == 0
                else _fixed_chunk(data, position, chunk_end, tag, wire)
            )
            # Grow only after proportional progress; short alternating fields
            # never speculate over a full 64 KiB window.
            if next_position - position >= (chunk_end - position) // 2 and units:
                count += units
                position = next_position
                chunk_bytes = min(CHUNK_BYTES, 2 * chunk_bytes)
                continue
        match = pattern.match(data, position, chunk_end)
        if match is None:
            break
        next_position = match.end()
        if wire == 0:
            # Every field has one terminating tag byte and one scalar byte.
            values = np.frombuffer(
                data, dtype=np.uint8, count=next_position - position, offset=position
            )
            count += int(np.count_nonzero(values < 128)) // 2
        else:
            size = stop - start if wire == 2 else len(tag) + (8 if wire == 1 else 4)
            count += (next_position - position) // size
        position = next_position
    return position, count


def packed_varints(data: bytes, position: int, end: int):
    match = PACKED.match(data, position, min(end, position + CHUNK_BYTES))
    if match is None:
        return position, 0
    stop = match.end()
    values = np.frombuffer(data, dtype=np.uint8, count=stop - position, offset=position)
    return stop, int(np.count_nonzero(values < 128))
