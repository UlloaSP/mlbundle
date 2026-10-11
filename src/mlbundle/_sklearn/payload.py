"""Bound the nested skops ZIP before skops reads schema or numeric buffers."""

from io import BytesIO
from math import prod
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from pydantic import JsonValue

from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.errors import InvalidModelError
from mlbundle.limits import (
    MANIFEST_BYTES,
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_OUTPUTS,
    ArchiveLimits,
)


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
        arrays = {name for name in names if name.endswith(".npy")} | _array_references(
            schema
        )
        for name in arrays:
            if name not in names:
                raise InvalidModelError("Missing skops numeric array member.")
            _array_header(archive, name, archive.getinfo(name).file_size)
        _tree_constructors(schema, archive)


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


def _array_header(
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
            or any(dimension < 0 for dimension in shape)
            or (prod(shape) * dtype.itemsize != size - stream.tell())
        ):
            raise InvalidModelError("Invalid skops numeric array dimensions.")
        return shape, dtype


def _tree_constructors(schema: JsonValue, archive: ZipFile) -> None:
    nodes: list[dict[str, JsonValue]] = []
    memo: dict[int, dict[str, JsonValue]] = {}
    variants: dict[int, list[dict[str, JsonValue]]] = {}
    pending = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            nodes.append(value)
            if (
                value.get("__module__") == "sklearn.tree._tree"
                and value.get("__class__") == "Tree"
                and value.get("__loader__") not in ("TreeNode", "CachedNode")
            ):
                raise InvalidModelError("Invalid skops native tree loader.")
            identity = value.get("__id__")
            if (
                "__loader__" in value
                and "__id__" in value
                and (type(identity) is not int or identity <= 0)
            ):
                raise InvalidModelError("Invalid skops object identity.")
            if type(identity) is int and value.get("__loader__") != "CachedNode":
                if identity in memo and (
                    value.get("__loader__") != memo[identity].get("__loader__")
                    or (
                        value.get("__loader__") in ("JsonNode", "NdArrayNode")
                        and canonical_dumps(value) != canonical_dumps(memo[identity])
                    )
                ):
                    raise InvalidModelError("Conflicting skops object references.")
                memo[identity] = value
                variants.setdefault(identity, []).append(value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    for node in nodes:
        if node.get("__loader__") == "TreeNode":
            _tree_variants(node, memo, variants, archive)


def _tree_variants(
    node: dict[str, JsonValue],
    memo: dict[int, dict[str, JsonValue]],
    variants: dict[int, list[dict[str, JsonValue]]],
    archive: ZipFile,
) -> None:
    reduced = node.get("__reduce__")
    if not isinstance(reduced, dict):
        raise InvalidModelError("Invalid skops tree constructor.")
    for args in _variants(reduced.get("args"), variants):
        for state in _variants(node.get("content"), variants):
            _tree_constructor(
                {**node, "__reduce__": {"args": args}, "content": state}, memo, archive
            )


def _variants(
    value: JsonValue, variants: dict[int, list[dict[str, JsonValue]]]
) -> list[dict[str, JsonValue]]:
    if not isinstance(value, dict):
        raise InvalidModelError("Invalid skops tree constructor schema.")
    identity = value.get("__id__")
    if type(identity) is int and identity in variants:
        return variants[identity]
    if value.get("__loader__") == "CachedNode":
        raise InvalidModelError("Invalid skops tree constructor reference.")
    return [value]


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
    value: JsonValue, memo: dict[int, dict[str, JsonValue]], archive: ZipFile
) -> tuple[str, tuple[int, ...], np.dtype]:
    node = _resolve(value, memo)
    name = node.get("file")
    if (
        node.get("__loader__") != "NdArrayNode"
        or node.get("type") != "numpy"
        or not isinstance(name, str)
    ):
        raise InvalidModelError("Invalid skops tree numeric storage.")
    shape, dtype = _array_header(archive, name, archive.getinfo(name).file_size)
    return name, shape, dtype


def _tree_constructor(
    node: dict[str, JsonValue], memo: dict[int, dict[str, JsonValue]], archive: ZipFile
) -> None:
    reduced = node.get("__reduce__")
    if not isinstance(reduced, dict):
        raise InvalidModelError("Invalid skops tree constructor.")
    arguments = reduced.get("args")
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
    name, shape, dtype = _numeric_header(args[1], memo, archive)
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
    count = _integer(attributes.get("node_count"), memo)
    depth = _integer(attributes.get("max_depth"), memo)
    _, nodes_shape, _ = _numeric_header(attributes.get("nodes"), memo, archive)
    _, values_shape, _ = _numeric_header(attributes.get("values"), memo, archive)
    if (
        count < 1
        or not 0 <= depth < count
        or nodes_shape != (count,)
        or values_shape != (count, outputs, int(max(classes)))
    ):
        raise InvalidModelError("Invalid skops tree storage dimensions.")
