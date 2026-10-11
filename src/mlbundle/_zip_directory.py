"""Bounded parsing of ZIP directory records, before any decompression."""

import stat
import struct
from dataclasses import dataclass

from ._container_rules import check_sizes, member_name
from .contract import MemberName
from .errors import InvalidContainerError, LimitExceededError
from .limits import ArchiveLimits

END = struct.Struct("<4s4H2IH")
END64 = struct.Struct("<4sQ2H2I4Q")
LOCATOR = struct.Struct("<4sIQI")
CENTRAL = struct.Struct("<4s6H3I5H2I")
LOCAL = struct.Struct("<4s5H3I2H")
U32_MAX = 0xFFFFFFFF
U16_MAX = 0xFFFF


def invalid() -> InvalidContainerError:
    return InvalidContainerError("Invalid or unsupported ZIP metadata.")


def unpack(record: struct.Struct, data: bytes, offset: int) -> tuple:
    if offset < 0 or offset + record.size > len(data):
        raise invalid()
    return record.unpack_from(data, offset)


def zip64_values(extra: bytes, fields: list[tuple[int, int]]) -> list[int]:
    """Accept only one ZIP64 field, consuming every supplied value exactly."""
    required = any(value == sentinel for value, sentinel in fields)
    if not extra:
        if required:
            raise invalid()
        return [value for value, _ in fields]
    if not required or len(extra) < 4:
        raise invalid()
    tag, length = struct.unpack_from("<HH", extra)
    if tag != 1 or length != len(extra) - 4:
        raise invalid()
    offset = 4
    result = []
    for value, sentinel in fields:
        resolved = value
        if value == sentinel:
            width = 4 if sentinel == U16_MAX else 8
            if offset + width > len(extra):
                raise invalid()
            resolved = int.from_bytes(extra[offset : offset + width], "little")
            offset += width
        result.append(resolved)
    if offset != len(extra):
        raise invalid()
    return result


def _directory_bounds(data: bytes) -> tuple[int, int, int]:
    end_offset = len(data) - END.size
    signature, disk, central_disk, disk_count, count, size, offset, comment = unpack(
        END, data, end_offset
    )
    if signature != b"PK\x05\x06" or disk or central_disk or comment:
        raise invalid()
    boundary = end_offset
    locator_offset = end_offset - LOCATOR.size
    if data[locator_offset : locator_offset + 4] == b"PK\x06\x07":
        _, disk64, position, disks = unpack(LOCATOR, data, locator_offset)
        if disk64 or disks != 1 or position + END64.size != locator_offset:
            raise invalid()
        record = unpack(END64, data, position)
        if (
            record[0] != b"PK\x06\x06"
            or record[1] != 44
            or record[3] != 45
            or record[4]
            or record[5]
        ):
            raise invalid()
        actual = record[6:10]
        original = (disk_count, count, size, offset)
        sentinels = (U16_MAX, U16_MAX, U32_MAX, U32_MAX)
        if any(
            old not in (new, sentinel)
            for old, new, sentinel in zip(original, actual, sentinels, strict=True)
        ):
            raise invalid()
        disk_count, count, size, offset = actual
        boundary = position
    elif U16_MAX in (disk_count, count) or U32_MAX in (size, offset):
        raise invalid()
    if disk_count != count or offset + size != boundary:
        raise invalid()
    return offset, size, count


@dataclass(frozen=True)
class Entry:
    name: MemberName
    flags: int
    method: int
    crc: int
    compressed: int
    size: int
    offset: int
    version: int
    time: int
    date: int


def _entry(data: bytes, offset: int, boundary: int) -> tuple[Entry, int]:
    fields = unpack(CENTRAL, data, offset)
    (
        signature,
        _made,
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
        comment_length,
        disk,
        _internal,
        external,
        local,
    ) = fields
    end = offset + CENTRAL.size + name_length + extra_length
    if signature != b"PK\x01\x02" or comment_length or end > boundary:
        raise invalid()
    name_start = offset + CENTRAL.size
    name = member_name(data[name_start : name_start + name_length])
    extra = data[name_start + name_length : end]
    if extra and version != 45:
        raise invalid()
    size, packed, local, disk = zip64_values(
        extra, [(size, U32_MAX), (packed, U32_MAX), (local, U32_MAX), (disk, U16_MAX)]
    )
    file_type = stat.S_IFMT(external >> 16)
    if disk or external & 0x10 or file_type not in (0, stat.S_IFREG):
        raise invalid()
    if method not in (0, 8) or flags & ~0x080E or version not in (10, 20, 45):
        raise invalid()
    if (method == 0 and flags & 6) or (method == 8 and version < 20):
        raise invalid()
    return Entry(
        name, flags, method, crc, packed, size, local, version, time, date
    ), end


def directory(data: bytes, limits: ArchiveLimits) -> tuple[list[Entry], int]:
    offset, size, count = _directory_bounds(data)
    if count > limits.max_members:
        raise LimitExceededError("Member count limit exceeded.")
    boundary = offset + size
    position = offset
    entries: list[Entry] = []
    names: set[MemberName] = set()
    for _ in range(count):
        entry, position = _entry(data, position, boundary)
        if entry.name in names:
            raise InvalidContainerError("Duplicate member name.")
        names.add(entry.name)
        entries.append(entry)
    if position != boundary:
        raise invalid()
    check_sizes(((e.name, e.compressed, e.size) for e in entries), limits)
    return entries, offset
