"""Checked views of the closed CatBoost 1.2.10 CBM FlatBuffer schema."""

import struct
from typing import Any

import numpy as np

from mlbundle.errors import InvalidModelError


class Buffer:
    def __init__(self, data: memoryview) -> None:
        self.data = data
        self.work = 0

    def span(self, position: int, size: int) -> memoryview:
        if position < 0 or size < 0 or size > len(self.data) - position:
            raise InvalidModelError("Truncated CatBoost CBM buffer.")
        return self.data[position : position + size]

    def number(self, position: int, fmt: str) -> Any:
        return struct.unpack("<" + fmt, self.span(position, struct.calcsize(fmt)))[0]

    def indirect(self, position: int) -> int:
        offset = self.number(position, "I")
        if offset < 4:
            raise InvalidModelError("Invalid CatBoost CBM offset.")
        target = position + offset
        self.span(target, 4)
        return target

    def table(self, position: int, fields: int) -> "Table":
        self.work += 1
        if self.work > 200_000:
            raise InvalidModelError("CatBoost CBM table work exceeds the limit.")
        return Table(self, position, fields)

    def string(self, position: int, cap: int) -> str:
        start = self.indirect(position)
        size = self.number(start, "I")
        if size > cap or self.number(start + 4 + size, "B") != 0:
            raise InvalidModelError("Invalid CatBoost CBM string length.")
        return bytes(self.span(start + 4, size)).decode("utf-8")


class Table:
    def __init__(self, buffer: Buffer, position: int, fields: int) -> None:
        self.buffer, self.position = buffer, position
        vtable = position - buffer.number(position, "i")
        size = buffer.number(vtable, "H")
        self.size = buffer.number(vtable + 2, "H")
        if size < 4 or size % 2 or size > 4 + 2 * fields or self.size < 4:
            raise InvalidModelError("Unknown or invalid CatBoost CBM table fields.")
        buffer.span(position, self.size)
        buffer.span(vtable, size)
        self.offsets = [buffer.number(vtable + i, "H") for i in range(4, size, 2)]
        if any(offset and not 4 <= offset < self.size for offset in self.offsets):
            raise InvalidModelError("Invalid CatBoost CBM field offset.")

    def field(self, index: int, size: int = 4) -> int | None:
        offset = self.offsets[index] if index < len(self.offsets) else 0
        if not offset:
            return None
        if size > self.size - offset:
            raise InvalidModelError("Invalid CatBoost CBM field size.")
        return self.position + offset

    def scalar(self, index: int, fmt: str, default: Any = 0) -> Any:
        position = self.field(index, struct.calcsize(fmt))
        return default if position is None else self.buffer.number(position, fmt)

    def string(self, index: int, cap: int) -> str:
        position = self.field(index)
        return "" if position is None else self.buffer.string(position, cap)

    def child(self, index: int, fields: int) -> "Table":
        position = self.field(index)
        if position is None:
            raise InvalidModelError("Missing CatBoost CBM table.")
        return self.buffer.table(self.buffer.indirect(position), fields)

    def vector(self, index: int, dtype: str, cap: int) -> np.ndarray:
        position = self.field(index)
        kind = np.dtype(dtype)
        if position is None:
            return np.empty(0, dtype=kind)
        start = self.buffer.indirect(position)
        count = self.buffer.number(start, "I")
        if count > cap:
            raise InvalidModelError("CatBoost CBM vector count exceeds the limit.")
        return np.frombuffer(self.buffer.span(start + 4, count * kind.itemsize), kind)

    def tables(self, index: int, fields: int, cap: int) -> list["Table"]:
        offsets = self.vector(index, "<u4", cap)
        if not len(offsets):
            return []
        position = self.field(index)
        if position is None:
            raise InvalidModelError("Missing CatBoost CBM vector.")
        start = self.buffer.indirect(position) + 4
        return [
            self.buffer.table(self.buffer.indirect(start + 4 * i), fields)
            for i in range(len(offsets))
        ]
