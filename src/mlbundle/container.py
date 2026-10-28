"""The validated byte boundary shared by inspection and model APIs."""

import struct
import zlib
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from ._container_rules import CHUNK_BYTES, BundleSource, check_members, effective_limits
from ._zip_directory import directory
from ._zip_members import decompress, payloads
from .canonical import DecodedJSON, decode_json
from .contract import MemberName
from .errors import InvalidContainerError, LimitExceededError
from .integrity import verify_integrity
from .limits import ArchiveLimits
from .lock import EnvironmentLock, check_bundle_lock
from .manifest import Manifest


def _read_source(source: BundleSource, cap: int) -> bytes:
    if isinstance(source, bytes):
        if len(source) > cap:
            raise LimitExceededError("Archive byte limit exceeded.")
        return source
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        if path.stat().st_size > cap:
            raise LimitExceededError("Archive byte limit exceeded.")
        with path.open("rb") as stream:
            return _read_source(stream, cap)
    content = bytearray()
    while chunk := source.read(min(CHUNK_BYTES, cap - len(content) + 1)):
        if not isinstance(chunk, bytes):
            raise TypeError("Bundle source must be a binary stream.")
        if len(content) + len(chunk) > cap:
            raise LimitExceededError("Archive byte limit exceeded.")
        content.extend(chunk)
    return bytes(content)


@dataclass
class ValidatedContainer:
    manifest: Manifest
    members: dict[MemberName, bytes]
    json: dict[MemberName, DecodedJSON]
    lock: EnvironmentLock


def read_validated_container(
    source: BundleSource, *, limits: ArchiveLimits | None = None
) -> ValidatedContainer:
    """Validate ZIP records, bounded decompression, manifest, integrity and lock.

    Streams are consumed from their current position and are never closed here.
    Model bytes remain opaque: this function does not verify model claims.
    """
    selected = effective_limits(limits)
    data = _read_source(source, selected.max_archive_bytes)
    try:
        entries, offset = directory(data, selected)
        check_members(entry.name for entry in entries)
        packed = payloads(data, entries, offset)
        by_name = {entry.name: entry for entry in entries}
        manifest_bytes = decompress(
            packed[MemberName.MANIFEST], by_name[MemberName.MANIFEST]
        )
        manifest = Manifest.from_bytes(manifest_bytes)
        check_members(by_name, manifest)
        members = {MemberName.MANIFEST: manifest_bytes}
        decoded = {}
        for entry in entries:
            if entry.name != MemberName.MANIFEST:
                value = decompress(packed[entry.name], entry)
                verify_integrity(entry.name, value, manifest.integrity[entry.name])
                if entry.name in (
                    MemberName.SCHEMA,
                    MemberName.STRUCTURE,
                    MemberName.SAMPLES,
                ):
                    decoded[entry.name] = decode_json(
                        value, max_bytes=selected.member_bytes(entry.name)
                    )
                members[entry.name] = value
    except (struct.error, zlib.error, OverflowError):
        raise InvalidContainerError("Corrupt ZIP container.") from None
    lock = EnvironmentLock.from_bytes(members[MemberName.LOCK])
    check_bundle_lock(lock, manifest)
    return ValidatedContainer(manifest, members, decoded, lock)


def read_container(
    source: BundleSource, *, limits: ArchiveLimits | None = None
) -> tuple[Manifest, dict[MemberName, bytes]]:
    """Return the validated manifest and original bytes, without loading a model."""
    result = read_validated_container(source, limits=limits)
    return result.manifest, result.members
