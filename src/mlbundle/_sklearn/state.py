"""Closed sklearn state keys and a content digest independent of skops object IDs."""

import hashlib
import json
from importlib.resources import files
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import numpy as np

from mlbundle.canonical import canonical_dumps
from mlbundle.errors import InvalidModelError, UnsupportedModelError

FIELDS = json.loads(files(__package__).joinpath("state_fields.json").read_text())


def _buffer_digest(data: bytes) -> str:
    if data.startswith(b"PK"):
        with ZipFile(BytesIO(data)) as archive:
            record = {
                name: _buffer_digest(archive.read(name)) for name in archive.namelist()
            }
        return hashlib.sha256(canonical_dumps(record)).hexdigest()
    array = np.load(BytesIO(data), allow_pickle=False)
    if array.dtype.names:
        # Native Tree records have uninitialized padding; only named fields
        # participate in inference and survive the native round trip.
        record = {name: _numeric_digest(array[name]) for name in array.dtype.names}
        return hashlib.sha256(canonical_dumps(record)).hexdigest()
    return _numeric_digest(array)


def _numeric_digest(array: np.ndarray) -> str:
    if array.dtype.kind == "f" and np.isnan(array).any():
        array = array.copy()
        array[np.isnan(array)] = np.nan
    digest = hashlib.sha256(canonical_dumps([array.dtype.str, list(array.shape)]))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _hash(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + canonical_dumps(value)).hexdigest()


def validate_state(nodes: list[dict[str, Any]], resolve) -> None:
    for node in nodes:
        if node.get("__loader__") != "ObjectNode":
            continue
        name = node.get("__module__", "") + "." + node.get("__class__", "")
        if name not in FIELDS:
            raise UnsupportedModelError((name,))
        state = resolve(node.get("content"))
        if state.get("__loader__") == "JsonNode" and state.get("content") == "null":
            continue
        if state.get("__loader__") != "DictNode":
            raise InvalidModelError("Invalid sklearn object state.")
        attrs = state.get("content")
        if not isinstance(attrs, dict) or set(attrs) - set(FIELDS[name]):
            raise InvalidModelError("Unknown sklearn estimator state.")
        # DictNode otherwise silently truncates attributes through zip().
        keys = resolve(state.get("key_types"))
        if keys.get("__loader__") != "ListNode" or len(keys["content"]) != len(attrs):
            raise InvalidModelError("Invalid sklearn state key types.")
        for raw_key in keys["content"]:
            key = resolve(raw_key)
            if key.get("__loader__") != "TypeNode" or (
                key.get("__module__"),
                key.get("__class__"),
            ) != ("builtins", "str"):
                raise InvalidModelError("Sklearn state keys must be strings.")


def state_digest(schema: Any, archive, resolve) -> str:
    """Merkle hash all state and buffers, without expanding shared references."""
    memo: dict[int, str] = {}
    buffers: dict[str, str] = {}

    def digest(value: Any) -> str:
        if isinstance(value, dict):
            node = resolve(value) if "__loader__" in value else value
            if id(node) in memo:
                return memo[id(node)]
            content = {}
            for key, child in node.items():
                if key == "__id__":
                    continue
                if key == "file":
                    if child not in buffers:
                        buffers[child] = _buffer_digest(archive.read(child))
                    content[key] = _hash(b"file:", buffers[child])
                else:
                    content[key] = digest(child)
            result = _hash(b"dict:", content)
            memo[id(node)] = result
            return result
        if isinstance(value, list):
            return _hash(b"list:", [digest(child) for child in value])
        return _hash(b"scalar:", value)

    return digest(schema)
