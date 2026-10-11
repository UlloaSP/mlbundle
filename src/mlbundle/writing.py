"""Deterministic container writing; the writer owns integrity records."""

import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from io import BytesIO
from os import PathLike
from pathlib import Path

from ._container_rules import (
    MEMBER_ORDER,
    check_members,
    check_sizes,
    effective_limits,
    member_name,
)
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
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _archive_bytes(content: dict[MemberName, bytes], limits: ArchiveLimits) -> bytes:
    with _BoundedBuffer(limits.max_archive_bytes) as buffer:
        with zipfile.ZipFile(
            buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in MEMBER_ORDER:
                if name in content:
                    entry = zipfile.ZipInfo(name.value, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFREG | 0o600) << 16
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(entry, content[name], compresslevel=6)
                    check_sizes(
                        (
                            (MemberName(e.filename), e.compress_size, e.file_size)
                            for e in archive.infolist()
                        ),
                        limits,
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
