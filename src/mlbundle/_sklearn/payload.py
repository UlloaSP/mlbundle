"""Bound the nested skops ZIP before skops reads schema or numeric buffers."""

from io import BytesIO
from math import prod
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from mlbundle.canonical import canonical_loads
from mlbundle.errors import InvalidModelError
from mlbundle.limits import MANIFEST_BYTES, ArchiveLimits


def validate_payload(data: bytes) -> None:
    limits = ArchiveLimits()
    with ZipFile(BytesIO(data)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (
            len(names) != len(set(names))
            or "schema.json" not in names
            or any("/" in name or "\\" in name or ".." in name for name in names)
            or sum(entry.file_size for entry in entries) > limits.max_member_bytes
        ):
            raise InvalidModelError("Invalid nested skops archive.")
        for entry in entries:
            if (
                entry.flag_bits & 1
                or entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                or entry.file_size > limits.max_member_bytes
                or entry.file_size > limits.max_compression_ratio * entry.compress_size
            ):
                raise InvalidModelError("Unsafe nested skops member bounds.")
            if entry.filename == "schema.json":
                if entry.file_size > MANIFEST_BYTES:
                    raise InvalidModelError("Skops schema exceeds the byte limit.")
                canonical_loads(archive.read(entry))
            elif entry.filename.endswith(".npy"):
                _array_header(archive, entry.filename, entry.file_size)


def _array_header(archive: ZipFile, name: str, size: int) -> None:
    with archive.open(name) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise InvalidModelError("Unsupported skops numeric array header.")
        if (
            dtype.hasobject
            or any(dimension < 0 for dimension in shape)
            or (prod(shape) * dtype.itemsize != size - stream.tell())
        ):
            raise InvalidModelError("Invalid skops numeric array dimensions.")
