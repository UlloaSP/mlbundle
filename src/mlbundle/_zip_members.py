"""Validate local records and decompress with explicit output and CRC bounds."""

import struct
import zlib

from ._container_rules import CHUNK_BYTES
from ._zip_directory import LOCAL, U32_MAX, Entry, invalid, unpack, zip64_values
from .contract import MemberName
from .errors import InvalidContainerError


def _descriptor(data: bytes, start: int, end: int, entry: Entry) -> None:
    if end - start not in (12, 16, 20, 24):
        raise invalid()
    raw = data[start:end]
    candidates = (raw, raw[4:]) if raw.startswith(b"PK\x07\x08") else (raw,)
    for candidate in candidates:
        if len(candidate) not in (12, 20):
            continue
        record = struct.unpack("<III" if len(candidate) == 12 else "<IQQ", candidate)
        if record == (entry.crc, entry.compressed, entry.size):
            return
    raise invalid()


def local_payload(data: bytes, entry: Entry, boundary: int) -> memoryview:
    (
        signature,
        version,
        flags,
        method,
        time,
        date,
        crc,
        packed,
        size,
        name_length,
        extra_length,
    ) = unpack(LOCAL, data, entry.offset)
    if (signature, version, flags, method, time, date) != (
        b"PK\x03\x04",
        entry.version,
        entry.flags,
        entry.method,
        entry.time,
        entry.date,
    ):
        raise invalid()
    start = entry.offset + LOCAL.size
    end = start + name_length + extra_length
    payload_end = end + entry.compressed
    if payload_end > boundary or data[
        start : start + name_length
    ] != entry.name.value.encode("ascii"):
        raise invalid()
    if extra_length and version != 45:
        raise invalid()
    size, packed = zip64_values(
        data[start + name_length : end], [(size, U32_MAX), (packed, U32_MAX)]
    )
    if flags & 8:
        if (
            crc not in (0, entry.crc)
            or packed not in (0, entry.compressed)
            or size not in (0, entry.size)
        ):
            raise invalid()
        _descriptor(data, payload_end, boundary, entry)
    elif (crc, packed, size, payload_end) != (
        entry.crc,
        entry.compressed,
        entry.size,
        boundary,
    ):
        raise invalid()
    return memoryview(data)[end:payload_end]


def payloads(
    data: bytes, entries: list[Entry], directory_offset: int
) -> dict[MemberName, memoryview]:
    ordered = sorted(entries, key=lambda entry: entry.offset)
    if not ordered or ordered[0].offset != 0:
        raise invalid()
    boundaries = [entry.offset for entry in ordered[1:]] + [directory_offset]
    return {
        entry.name: local_payload(data, entry, boundary)
        for entry, boundary in zip(ordered, boundaries, strict=True)
    }


def decompress(payload: memoryview, entry: Entry) -> bytes:
    output = bytearray()
    crc = 0
    decoder = zlib.decompressobj(-15) if entry.method == 8 else None
    for start in range(0, len(payload), CHUNK_BYTES):
        pending = payload[start : start + CHUNK_BYTES]
        while pending:
            allowance = min(CHUNK_BYTES, entry.size - len(output) + 1)
            chunk = (
                decoder.decompress(pending, allowance) if decoder else bytes(pending)
            )
            if decoder and decoder.unused_data:
                raise invalid()
            if len(output) + len(chunk) > entry.size:
                raise InvalidContainerError(
                    "Actual member size exceeds ZIP declaration."
                )
            output.extend(chunk)
            crc = zlib.crc32(chunk, crc)
            pending = decoder.unconsumed_tail if decoder else b""
    if decoder and (not decoder.eof or decoder.unused_data):
        raise invalid()
    if len(output) != entry.size or crc != entry.crc:
        raise InvalidContainerError("Member size or CRC disagrees with ZIP metadata.")
    return bytes(output)
