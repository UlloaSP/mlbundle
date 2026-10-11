"""Deterministic container writing; the writer owns integrity records."""

import errno
import os
import stat
import tempfile
import zlib
from collections.abc import Mapping
from io import BytesIO
from os import PathLike
from pathlib import Path

from ._container_rules import (
    CHUNK_BYTES,
    MEMBER_ORDER,
    check_members,
    check_sizes,
    effective_limits,
    member_name,
)
from ._zip_directory import CENTRAL, END, LOCAL
from .canonical import canonical_dumps
from .container import read_container
from .contract import MemberName
from .errors import InvalidContainerError, LimitExceededError
from .integrity import hash_member
from .limits import ArchiveLimits
from .manifest import Manifest


class _BoundedBuffer(BytesIO):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap

    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > self.cap:
            raise LimitExceededError("Archive byte limit exceeded.")
        return super().write(data)


def _finish_manifest(
    manifest: Manifest | Mapping[str, object], members: dict[MemberName, bytes]
) -> Manifest:
    value = (
        manifest.model_dump(mode="json", by_alias=True)
        if isinstance(manifest, Manifest)
        else dict(manifest)
    )
    value["integrity"] = {
        name.value: hash_member(data).model_dump() for name, data in members.items()
    }
    return Manifest.from_bytes(canonical_dumps(value))


def _atomic_write(destination: Path, data: bytes, overwrite: bool) -> None:
    if not overwrite and destination.exists():
        raise FileExistsError(destination)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".mlbundle-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            temporary.replace(destination)
        else:
            # Publish the complete file without a check/replace race.
            try:
                os.link(temporary, destination)
            except OSError as error:
                unsupported = {
                    errno.ENOSYS,
                    errno.ENOTSUP,
                    errno.EOPNOTSUPP,
                    errno.EPERM,
                    errno.EXDEV,
                }
                if error.errno not in unsupported and getattr(
                    error, "winerror", None
                ) not in (1, 50, 1314):
                    raise
                _exclusive_write(destination, data)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _exclusive_write(destination: Path, data: bytes) -> None:
    # Filesystems without hard links still get exclusive ownership; remove our
    # destination on every failure, never a competing writer's existing file.
    stream = destination.open("xb")
    try:
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _compress(data: bytes, cap: int) -> bytes:
    # RLE avoids the platform-specific match searches of zlib versus zlib-ng.
    encoder = zlib.compressobj(6, zlib.DEFLATED, -15, 8, zlib.Z_RLE)
    with _BoundedBuffer(cap) as packed:
        for offset in range(0, len(data), CHUNK_BYTES):
            packed.write(
                encoder.compress(memoryview(data)[offset : offset + CHUNK_BYTES])
            )
        packed.write(encoder.flush())
        return packed.getvalue()


def _archive_bytes(content: dict[MemberName, bytes], limits: ArchiveLimits) -> bytes:
    central = bytearray()
    sizes = []
    overhead = END.size + sum(
        LOCAL.size + CENTRAL.size + 2 * len(name.value) for name in content
    )
    remaining = min(limits.max_archive_bytes - overhead, limits.max_compressed_bytes)
    with _BoundedBuffer(limits.max_archive_bytes) as buffer:
        for name in MEMBER_ORDER:
            if name not in content:
                continue
            data = content[name]
            if remaining <= 0:
                raise LimitExceededError("Container byte limit exceeded.")
            packed = _compress(data, remaining)
            remaining -= len(packed)
            sizes.append((name, len(packed), len(data)))
            check_sizes(sizes, limits)
            raw_name = name.value.encode("ascii")
            crc = zlib.crc32(data)
            offset = buffer.tell()
            buffer.write(
                LOCAL.pack(
                    b"PK\x03\x04",
                    20,
                    0,
                    8,
                    0,
                    33,
                    crc,
                    len(packed),
                    len(data),
                    len(raw_name),
                    0,
                )
                + raw_name
            )
            buffer.write(packed)
            central.extend(
                CENTRAL.pack(
                    b"PK\x01\x02",
                    (3 << 8) | 20,
                    20,
                    0,
                    8,
                    0,
                    33,
                    crc,
                    len(packed),
                    len(data),
                    len(raw_name),
                    0,
                    0,
                    0,
                    0,
                    (stat.S_IFREG | 0o600) << 16,
                    offset,
                )
                + raw_name
            )
        directory_offset = buffer.tell()
        buffer.write(bytes(central))
        buffer.write(
            END.pack(
                b"PK\x05\x06",
                0,
                0,
                len(sizes),
                len(sizes),
                len(central),
                directory_offset,
                0,
            )
        )
        return buffer.getvalue()


def write_container(
    manifest: Manifest | Mapping[str, object],
    members: Mapping[MemberName, bytes],
    *,
    destination: str | PathLike[str] | None = None,
    overwrite: bool = False,
    limits: ArchiveLimits | None = None,
) -> bytes:
    """Finish integrity, encode a deterministic ZIP and optionally publish it.

    Supply only non-manifest member bytes and a manifest or its JSON-ready parts.
    Existing integrity records are replaced from those exact bytes. References
    must already identify the supplied members. Existing paths require overwrite.
    """
    selected = effective_limits(limits)
    content = {member_name(name): data for name, data in members.items()}
    if MemberName.MANIFEST in content:
        raise InvalidContainerError("The writer creates manifest.json.")
    check_members([MemberName.MANIFEST, *content])
    if len(content) + 1 > selected.max_members:
        raise LimitExceededError("Member count limit exceeded.")
    for name, data in content.items():
        if not isinstance(data, bytes):
            raise TypeError("Member content must be bytes.")
        if len(data) > selected.member_bytes(name):
            raise LimitExceededError(f"Member size limit exceeded: {name.value}.")
    if sum(map(len, content.values())) > selected.max_total_bytes:
        raise LimitExceededError("Total member size limit exceeded.")
    finished = _finish_manifest(manifest, content)
    content[MemberName.MANIFEST] = canonical_dumps(
        finished.model_dump(mode="json", by_alias=True),
        max_bytes=selected.member_bytes(MemberName.MANIFEST),
    )
    if sum(map(len, content.values())) > selected.max_total_bytes:
        raise LimitExceededError("Total member size limit exceeded.")
    check_members(content, finished)
    result = _archive_bytes(content, selected)
    # One owner for all ZIP/integrity/JSON checks, including actual compressed sizes.
    read_container(result, limits=selected)
    if destination is not None:
        _atomic_write(Path(destination), result, overwrite)
    return result
