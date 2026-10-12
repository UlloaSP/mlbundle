"""Closed sklearn state keys and a content digest independent of skops object IDs."""

import hashlib
import json
from importlib.resources import files
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import numpy as np

from mlbundle.canonical import canonical_dumps
from mlbundle.errors import InvalidModelError, LimitExceededError, UnsupportedModelError
from mlbundle.limits import MANIFEST_BYTES

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
    if domain == b"scalar:" or (isinstance(value, dict) and "$float" in value):
        encoded = canonical_dumps(value)
    else:
        # These records contain only checked string keys and fixed-width hex digests.
        size = (
            sum(len(key) + 70 for key in value)
            if isinstance(value, dict)
            else len(value) * 67
            if isinstance(value, list)
            else 66
        ) + 1
        if size > MANIFEST_BYTES:
            raise LimitExceededError("JSON exceeds the byte limit.")
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MANIFEST_BYTES:
            raise LimitExceededError("JSON exceeds the byte limit.")
    return hashlib.sha256(domain + encoded).hexdigest()


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


def _memoized_hash(cache, key, domain: bytes, value) -> str:
    if key not in cache:
        cache[key] = _hash(domain, value)
    return cache[key]


def _record_key(value):
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return tuple(value) if isinstance(value, list) else value


def state_digest(schema: Any, archive, resolve) -> str:
    """Merkle hash all state and buffers, without expanding shared references."""
    memo: dict[int, str] = {}
    buffers: dict[str, str] = {}
    scalars: dict[tuple[type, Any], str] = {}
    records: dict[tuple, str] = {}

    def record(domain: bytes, value) -> str:
        key = (domain, _record_key(value))
        return _memoized_hash(records, key, domain, value)

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
                    content[key] = record(b"file:", buffers[child])
                else:
                    content[key] = digest(child)
            result = record(b"dict:", content)
            memo[id(node)] = result
            return result
        if isinstance(value, list):
            return record(b"list:", [digest(child) for child in value])
        key = (type(value), value.hex() if type(value) is float else value)
        return _memoized_hash(scalars, key, b"scalar:", value)

    return digest(schema)
