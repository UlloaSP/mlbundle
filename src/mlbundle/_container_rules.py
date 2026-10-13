"""Shared container names, member sets and resource checks."""

from collections.abc import Iterable
from os import PathLike
from typing import BinaryIO

from .contract import MODEL_MEMBERS, MemberName
from .errors import InvalidContainerError, LimitExceededError
from .limits import ArchiveLimits
from .manifest import Manifest

BundleSource = str | PathLike[str] | bytes | BinaryIO
CHUNK_BYTES = 64 * 1024
MEMBER_ORDER = (
    MemberName.MANIFEST,
    MemberName.SKOPS,
    MemberName.UBJSON,
    MemberName.CBM,
    MemberName.ONNX,
    MemberName.SCHEMA,
    MemberName.STRUCTURE,
    MemberName.SAMPLES,
)


def effective_limits(limits: ArchiveLimits | None) -> ArchiveLimits:
    # Revalidate even model_copy/model_construct values; ceilings are mandatory.
    return ArchiveLimits.model_validate((limits or ArchiveLimits()).model_dump())


def member_name(raw: bytes | str) -> MemberName:
    try:
        return MemberName(raw if isinstance(raw, str) else raw.decode("ascii"))
    except (UnicodeError, ValueError):
        raise InvalidContainerError("Unknown or unsafe member name.") from None


def check_members(
    names: Iterable[MemberName], manifest: Manifest | None = None
) -> None:
    present = set(names)
    if MemberName.MANIFEST not in present:
        raise InvalidContainerError("Missing manifest.json.")
    if MemberName.SCHEMA not in present:
        raise InvalidContainerError(
            "Bundle predates the required schema.json and must be re-exported."
        )
    models = {pair[1] for pair in MODEL_MEMBERS.values()}
    selected = present & models
    if len(selected) != 1 and selected not in (
        {MemberName.SKOPS, MemberName.UBJSON},
        {MemberName.SKOPS, MemberName.CBM},
    ):
        raise InvalidContainerError("Invalid model member combination.")
    if manifest is not None and present != {MemberName.MANIFEST, *manifest.integrity}:
        raise InvalidContainerError("Members disagree with manifest references.")


def check_sizes(
    sizes: Iterable[tuple[MemberName, int, int]], limits: ArchiveLimits
) -> None:
    compressed = uncompressed = count = 0
    for name, packed, unpacked in sizes:
        count += 1
        compressed += packed
        uncompressed += unpacked
        if packed > limits.max_compressed_bytes or unpacked > limits.member_bytes(name):
            raise LimitExceededError(f"Member size limit exceeded: {name.value}.")
        if unpacked > limits.max_compression_ratio * packed:
            raise LimitExceededError(f"Compression ratio limit exceeded: {name.value}.")
    if count > limits.max_members:
        raise LimitExceededError("Member count limit exceeded.")
    if (
        compressed > limits.max_compressed_bytes
        or uncompressed > limits.max_total_bytes
    ):
        raise LimitExceededError("Total member size limit exceeded.")
