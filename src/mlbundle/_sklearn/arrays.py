"""Check declared numeric and sparse storage against real bytes before allocation."""

from io import BytesIO
from math import prod
from zipfile import ZipFile

import numpy as np

from mlbundle.errors import InvalidModelError
from mlbundle.limits import ArchiveLimits

from .archive import bounded_zip

MAX_RANK = 32
MAX_FILES = 100_000
BYTE_CAP = ArchiveLimits().max_member_bytes


def array_header(
    archive: ZipFile, name: str, size: int
) -> tuple[tuple[int, ...], np.dtype]:
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
            or dtype.itemsize == 0
            or len(shape) > MAX_RANK
            or any(not 0 <= dimension <= BYTE_CAP for dimension in shape)
            or prod(shape) * dtype.itemsize != size - stream.tell()
        ):
            raise InvalidModelError("Invalid skops numeric array dimensions.")
        expected = size - stream.tell()
        actual = 0
        while chunk := stream.read(min(64 * 1024, expected - actual + 1)):
            actual += len(chunk)
            if actual > expected:
                raise InvalidModelError("Invalid skops numeric array stored size.")
        if actual != expected:
            raise InvalidModelError("Invalid skops numeric array stored size.")
        return shape, dtype


def sparse_headers(data: bytes, max_bytes: int = BYTE_CAP) -> int:
    with bounded_zip(data, 9) as archive:
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        if (
            len(names) != len(entries)
            or not {"format.npy", "shape.npy", "data.npy"} <= names
            or names
            - {
                "format.npy",
                "shape.npy",
                "data.npy",
                "indices.npy",
                "indptr.npy",
                "row.npy",
                "col.npy",
                "offsets.npy",
                "_is_array.npy",
            }
            or sum(entry.file_size for entry in entries) > max_bytes
        ):
            raise InvalidModelError("Invalid skops sparse archive.")
        headers = {}
        for entry in entries:
            if (
                entry.flag_bits & 1
                or entry.compress_type not in (0, 8)
                or entry.file_size > BYTE_CAP
                or entry.file_size > 200 * entry.compress_size
            ):
                raise InvalidModelError("Unsafe skops sparse member bounds.")
            headers[entry.filename] = array_header(
                archive, entry.filename, entry.file_size
            )
        if headers["shape.npy"][0] != (2,) or headers["format.npy"][0] != ():
            raise InvalidModelError("Invalid skops sparse shape or format.")
        shape = np.load(BytesIO(archive.read("shape.npy")), allow_pickle=False)
        if (
            shape.dtype.kind not in "iu"
            or any(not 0 <= dim <= BYTE_CAP for dim in shape)
            or prod(int(dim) for dim in shape) * headers["data.npy"][1].itemsize
            > BYTE_CAP
        ):
            raise InvalidModelError("Unsafe skops sparse declared shape.")
        # All buffers have already passed exact stored-byte checks. Loading them
        # here cannot allocate in proportion to an unsupported header declaration.
        values = {
            name[:-4]: np.load(BytesIO(archive.read(name)), allow_pickle=False)
            for name in names
        }
        _sparse_indices(values, shape)
        return sum(entry.file_size for entry in entries)


def _sparse_indices(values, shape) -> None:
    kind = values["format"].item()
    rows, columns = (int(dim) for dim in shape)
    if kind in (b"csr", b"csc", b"bsr"):
        indices, pointers = values["indices"], values["indptr"]
        blocks = len(values["data"])
        major, minor = (columns, rows) if kind == b"csc" else (rows, columns)
        if kind == b"bsr":
            block_shape = values["data"].shape
            if len(block_shape) != 3 or min(block_shape[1:]) < 1:
                raise InvalidModelError("Invalid skops sparse block shape.")
            major //= block_shape[1]
            minor //= block_shape[2]
        if (
            pointers.dtype.kind not in "iu"
            or indices.dtype.kind not in "iu"
            or pointers.shape != (major + 1,)
            or indices.shape != (blocks,)
            or pointers[0] != 0
            or pointers[-1] != blocks
            or np.any(pointers[1:] < pointers[:-1])
            or np.any(pointers < 0)
            or np.any(indices < 0)
            or np.any(indices >= minor)
        ):
            raise InvalidModelError("Invalid skops sparse index storage.")
    elif kind == b"coo":
        for name, width in (("row", rows), ("col", columns)):
            index = values[name]
            if (
                index.dtype.kind not in "iu"
                or index.shape != values["data"].shape
                or np.any(index < 0)
                or np.any(index >= width)
            ):
                raise InvalidModelError("Invalid skops sparse coordinate storage.")
    elif kind == b"dia":
        offsets = values["offsets"]
        if (
            offsets.dtype.kind not in "iu"
            or offsets.ndim != 1
            or values["data"].ndim != 2
            or len(offsets) != len(values["data"])
            or len(set(offsets.tolist())) != len(offsets)
        ):
            raise InvalidModelError("Invalid skops sparse diagonal storage.")
    else:
        raise InvalidModelError("Unsupported skops sparse format.")
