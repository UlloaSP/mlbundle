"""Bounded UBJSON preflight; numeric buffers stay as views, never expanded lists."""

import struct
from dataclasses import dataclass
from typing import Any

from mlbundle.errors import InvalidModelError

_FORMATS = {"i": "b", "U": "B", "I": "h", "l": "i", "L": "q", "d": "f", "D": "d"}


@dataclass(frozen=True)
class Buffer:
    data: memoryview
    marker: str
    count: int


class Decoder:
    def __init__(self, data: bytes) -> None:
        self.data = memoryview(data)
        self.position = 0
        self.values = 0

    def take(self, count: int) -> memoryview:
        if count < 0 or count > len(self.data) - self.position:
            raise InvalidModelError("Truncated UBJSON model.")
        result = self.data[self.position : self.position + count]
        self.position += count
        return result

    def marker(self) -> str:
        return chr(self.take(1)[0])

    def integer(self) -> int:
        marker = self.marker()
        if marker not in "iUIlL":
            raise InvalidModelError("Invalid UBJSON length.")
        return self.number(marker)

    def number(self, marker: str) -> Any:
        fmt = ">" + _FORMATS[marker]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return bytes(self.take(self.integer())).decode("utf-8")

    def value(self, depth: int = 0, marker: str | None = None) -> Any:
        self.values += 1
        if self.values > 1_000_000:
            raise InvalidModelError("UBJSON value count exceeds the limit.")
        if depth > 32:
            raise InvalidModelError("UBJSON nesting exceeds the limit.")
        marker = self.marker() if marker is None else marker
        if marker in _FORMATS:
            return self.number(marker)
        if marker == "S":
            return self.string()
        if marker in ("T", "F", "Z"):
            return {"T": True, "F": False, "Z": None}[marker]
        if marker in ("{", "["):
            return self.container(marker, depth + 1)
        raise InvalidModelError("Unsupported UBJSON token.")

    def header(self) -> tuple[str | None, int | None]:
        typed = None
        count = None
        if self.position < len(self.data) and chr(self.data[self.position]) == "$":
            self.take(1)
            typed = self.marker()
        if self.position < len(self.data) and chr(self.data[self.position]) == "#":
            self.take(1)
            count = self.integer()
            if count < 0 or count > len(self.data):
                raise InvalidModelError("Unsafe UBJSON collection length.")
        if typed is not None and count is None:
            raise InvalidModelError("Unbounded typed UBJSON collection.")
        return typed, count

    def container(self, marker: str, depth: int) -> Any:
        end = "}" if marker == "{" else "]"
        typed, count = self.header()
        if marker == "[" and typed in _FORMATS and count is not None:
            size = struct.calcsize(">" + _FORMATS[typed])
            return Buffer(self.take(count * size), typed, count)
        result: Any = {} if marker == "{" else []
        index = 0
        while count is None or index < count:
            if count is None and chr(self.data[self.position]) == end:
                self.take(1)
                break
            if index >= 100_000:
                raise InvalidModelError("UBJSON object collection exceeds the limit.")
            key = self.string() if marker == "{" else None
            value = self.value(depth, typed)
            if marker == "{":
                if key in result:
                    raise InvalidModelError("Duplicate UBJSON object key.")
                result[key] = value
            else:
                result.append(value)
            index += 1
        return result


def decode(data: bytes) -> dict[str, Any]:
    decoder = Decoder(data)
    value = decoder.value()
    if not isinstance(value, dict) or decoder.position != len(data):
        raise InvalidModelError("Invalid UBJSON model document.")
    return value
