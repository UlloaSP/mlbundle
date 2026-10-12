"""Bound the nested skops ZIP before skops reads schema or numeric buffers."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from pydantic import JsonValue

from mlbundle.canonical import canonical_loads
from mlbundle.errors import InvalidModelError
from mlbundle.limits import (
    MANIFEST_BYTES,
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_OUTPUTS,
    ArchiveLimits,
)

from .archive import bounded_zip
from .arrays import MAX_FILES, array_header, sparse_headers
from .dtypes import validate_dtypes
from .nodes import validate_nodes
from .references import References
from .state import state_digest, validate_state

BUFFER_BYTES = ArchiveLimits().max_member_bytes


def validate_payload(data: bytes) -> str:
    limits = ArchiveLimits()
    with bounded_zip(data, MAX_FILES) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (
            len(names) > MAX_FILES
            or len(names) != len(set(names))
            or "schema.json" not in names
            or any("/" in name or "\\" in name or ".." in name for name in names)
            or sum(entry.file_size for entry in entries) > limits.max_member_bytes
        ):
            raise InvalidModelError("Invalid nested skops archive.")
        schema_entry = archive.getinfo("schema.json")
        if schema_entry.file_size > MANIFEST_BYTES:
            raise InvalidModelError("Skops schema exceeds the byte limit.")
        for entry in entries:
            if (
                entry.flag_bits & 1
                or entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                or entry.file_size > limits.max_member_bytes
                or entry.file_size > limits.max_compression_ratio * entry.compress_size
            ):
                raise InvalidModelError("Unsafe nested skops member bounds.")
        schema = canonical_loads(archive.read(schema_entry))
        references = References(schema)
        validate_nodes(references.nodes, schema)
        validate_state(references.nodes, references.resolve)
        arrays = {name for name in names if name.endswith(".npy")} | _array_references(
            schema
        )
        sparse = {
            node["file"]
            for node in references.nodes
            if node.get("__loader__") == "SparseMatrixNode"
        }
        remaining = limits.max_member_bytes - sum(
            archive.getinfo(name).file_size for name in arrays if name in names
        )
        sparse_sizes = {}
        for name in sparse:
            sparse_sizes[name] = sparse_headers(archive.read(name), remaining)
            remaining -= sparse_sizes[name]
        _buffer_budget(references, archive, sparse_sizes)
        headers = {}
        for name in arrays:
            if name not in names:
                raise InvalidModelError("Missing skops numeric array member.")
            headers[name] = _array_header(
                archive, name, archive.getinfo(name).file_size
            )
        validate_dtypes(references.nodes, references.resolve, archive)
        _tree_constructors(references, archive, headers)
        files = _array_references(schema) | sparse | {"schema.json"}
        if set(names) != files:
            raise InvalidModelError("Unreferenced skops member.")
        return state_digest(schema, archive, references.resolve)


def _array_references(schema: JsonValue) -> set[str]:
    arrays: set[str] = set()
    pending = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if node.get("__loader__") == "NdArrayNode" and "file" in node:
                name = node["file"]
                if not isinstance(name, str):
                    raise InvalidModelError("Invalid skops numeric array reference.")
                arrays.add(name)
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return arrays


def _buffer_budget(
    references: References, archive: ZipFile, sparse: dict[str, int]
) -> None:
    seen: set[int] = set()
    total = 0
    for node in references.nodes:
        if (
            node.get("__loader__") not in ("NdArrayNode", "SparseMatrixNode")
            or "file" not in node
        ):
            continue
        identity = node.get("__id__", id(node))
        if identity in seen:
            continue
        seen.add(identity)
        total += sparse.get(node["file"], archive.getinfo(node["file"]).file_size)
        if total > BUFFER_BYTES:
            raise InvalidModelError("Skops materialized buffer bytes exceed the limit.")


_array_header = array_header


def _tree_constructors(references: References, archive: ZipFile, headers) -> None:
    seen: set[int] = set()
    for node in references.nodes:
        identity = node.get("__id__", id(node))
        if node.get("__loader__") == "TreeNode" and identity not in seen:
            seen.add(identity)
            reduced = node.get("__reduce__")
            if not isinstance(reduced, dict):
                raise InvalidModelError("Invalid skops tree constructor.")
            args = references.resolve(reduced.get("args"))
            state = references.resolve(node.get("content"))
            _tree_constructor(
                {**node, "__reduce__": {"args": args}, "content": state},
                references.memo,
                archive,
                headers,
            )


def _resolve(
    value: JsonValue, memo: dict[int, dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise InvalidModelError("Invalid skops tree constructor schema.")
    identity = value.get("__id__")
    if type(identity) is int and identity in memo:
        return memo[identity]
    if value.get("__loader__") == "CachedNode":
        raise InvalidModelError("Invalid skops tree constructor reference.")
    return value


def _integer(value: JsonValue, memo: dict[int, dict[str, JsonValue]]) -> int:
    node = _resolve(value, memo)
    content = node.get("content")
    if (
        node.get("__loader__") != "JsonNode"
        or node.get("is_json") is not True
        or not isinstance(content, str)
    ):
        raise InvalidModelError("Invalid skops tree constructor integer.")
    result = canonical_loads(content.encode())
    if type(result) is not int:
        raise InvalidModelError("Invalid skops tree constructor integer.")
    return result


def _numeric_header(
    value: JsonValue, memo: dict[int, dict[str, JsonValue]], headers
) -> tuple[str, tuple[int, ...], np.dtype]:
    node = _resolve(value, memo)
    name = node.get("file")
    if (
        node.get("__loader__") != "NdArrayNode"
        or node.get("type") != "numpy"
        or not isinstance(name, str)
    ):
        raise InvalidModelError("Invalid skops tree numeric storage.")
    shape, dtype = headers[name]
    return name, shape, dtype


def _tree_constructor(
    node: dict[str, JsonValue],
    memo: dict[int, dict[str, JsonValue]],
    archive: ZipFile,
    headers,
) -> None:
    reduced = node.get("__reduce__")
    arguments = reduced.get("args") if isinstance(reduced, dict) else None
    if not isinstance(arguments, dict):
        raise InvalidModelError("Invalid skops tree constructor arguments.")
    args = arguments.get("content")
    if (
        arguments.get("__loader__") != "TupleNode"
        or not isinstance(args, list)
        or len(args) != 3
    ):
        raise InvalidModelError("Invalid skops tree constructor arguments.")
    width, outputs = _integer(args[0], memo), _integer(args[2], memo)
    if not 1 <= width <= MAX_FEATURES or not 1 <= outputs <= MAX_OUTPUTS:
        raise InvalidModelError("Unsafe skops tree constructor dimensions.")
    name, shape, dtype = _numeric_header(args[1], memo, headers)
    if shape != (outputs,) or dtype != np.dtype(np.intp):
        raise InvalidModelError("Invalid skops tree class-count dimensions.")
    classes = np.load(BytesIO(archive.read(name)), allow_pickle=False)
    if np.any(classes < 1) or np.any(classes > MAX_CLASSES):
        raise InvalidModelError("Unsafe skops tree class counts.")
    state = node.get("content")
    if not isinstance(state, dict):
        raise InvalidModelError("Invalid skops tree state.")
    attributes = state.get("content")
    if not isinstance(attributes, dict):
        raise InvalidModelError("Invalid skops tree state.")
    if set(attributes) != {"node_count", "max_depth", "nodes", "values"}:
        raise InvalidModelError("Unknown skops tree state.")
    count = _integer(attributes.get("node_count"), memo)
    depth = _integer(attributes.get("max_depth"), memo)
    _, nodes_shape, _ = _numeric_header(attributes.get("nodes"), memo, headers)
    _, values_shape, _ = _numeric_header(attributes.get("values"), memo, headers)
    if (
        count < 1
        or not 0 <= depth < count
        or nodes_shape != (count,)
        or values_shape != (count, outputs, int(max(classes)))
    ):
        raise InvalidModelError("Invalid skops tree storage dimensions.")
