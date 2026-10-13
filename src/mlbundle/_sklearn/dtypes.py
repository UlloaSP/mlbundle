"""Validate skops' empty-array dtype representation before construction."""

from mlbundle.canonical import canonical_loads
from mlbundle.errors import InvalidModelError

from .arrays import array_header
from .trusted import dtype_identifier


def validate_dtypes(nodes, resolve, archive) -> None:
    for node in nodes:
        if node.get("__loader__") != "DTypeNode":
            continue
        content = resolve(node.get("content"))
        if content.get("__loader__") != "NdArrayNode" or (
            content.get("__module__"),
            content.get("__class__"),
        ) != ("numpy", "ndarray"):
            raise InvalidModelError("Invalid skops dtype storage.")
        if content.get("type") == "numpy":
            name = content["file"]
            shape, dtype = array_header(archive, name, archive.getinfo(name).file_size)
            dtype_identifier(dtype)
            if shape != (0,):
                raise InvalidModelError("Skops dtype storage must be empty and scalar.")
        elif content.get("type") == "json":
            # Only the canonical empty object array emitted by dtype_get_state.
            shape = resolve(content.get("shape"))
            if shape.get("__loader__") != "TupleNode" or len(shape["content"]) != 1:
                raise InvalidModelError("Invalid skops object dtype shape.")
            size = resolve(shape["content"][0])
            if (
                size.get("__loader__") != "JsonNode"
                or size.get("is_json") is not True
                or canonical_loads(size["content"].encode()) != 0
                or size["content"] != "0"
                or content.get("content") != []
            ):
                raise InvalidModelError("Skops object dtype storage must be empty.")
        else:
            raise InvalidModelError("Invalid skops dtype storage kind.")
