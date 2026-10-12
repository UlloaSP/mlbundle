"""Bound nested ZIP directory work before ZipFile creates entry objects."""

import stat
from io import BytesIO
from zipfile import ZipFile

from mlbundle._zip_directory import CENTRAL, _directory_bounds, unpack
from mlbundle.errors import InvalidContainerError, InvalidModelError


def bounded_zip(data: bytes, max_files: int) -> ZipFile:
    try:
        offset, size, count = _directory_bounds(data)
    except InvalidContainerError:
        raise InvalidModelError("Invalid nested skops archive.") from None
    if count > max_files:
        raise InvalidModelError("Skops nested member count exceeds the limit.")
    position = offset
    boundary = offset + size
    for _ in range(count):
        record = unpack(CENTRAL, data, position)
        if record[0] != b"PK\x01\x02" or stat.S_IFMT(record[15] >> 16) not in (
            0,
            stat.S_IFREG,
        ):
            raise InvalidModelError("Invalid skops nested ZIP directory.")
        position += CENTRAL.size + sum(record[10:13])
        if position > boundary:
            raise InvalidModelError("Invalid skops nested ZIP directory.")
    if position != boundary:
        raise InvalidModelError("Invalid skops nested ZIP member count.")
    return ZipFile(BytesIO(data))
