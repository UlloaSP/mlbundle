"""Closed skops node layouts: no unused state, loaders or hidden child objects."""

from typing import Any

from mlbundle.errors import InvalidModelError, UnsupportedModelError

from .trusted import FUNCTIONS, OPTIMIZER_TYPES, SCALAR_NAMES

FIELDS = {
    "ObjectNode": {"content"},
    "TreeNode": {"content", "__reduce__"},
    "LossNode": {"content", "__reduce__"},
    "DictNode": {"content", "key_types"},
    "ListNode": {"content"},
    "TupleNode": {"content"},
    "SetNode": {"content"},
    "JsonNode": {"content", "is_json"},
    "TypeNode": set(),
    "FunctionNode": set(),
    "CachedNode": set(),
    "NdArrayNode": {"type", "file", "content", "shape"},
    "SparseMatrixNode": {"type", "file"},
    "RandomStateNode": {"content"},
    "RandomGeneratorNode": {"content"},
    "MaskedArrayNode": {"content"},
    "DTypeNode": {"content"},
    "SliceNode": {"content"},
    "ConstructorFromReduceNode": {"content"},
    "BytesNode": {"file"},
    "BytearrayNode": {"file"},
}
COMMON = {"__class__", "__module__", "__loader__", "__id__"}


def validate_nodes(nodes: list[dict[str, Any]], root: Any) -> None:
    for node in nodes:
        loader = node["__loader__"]
        name = node.get("__module__", "") + "." + node.get("__class__", "")
        _special_loader(node, name, loader)
        if name in OPTIMIZER_TYPES and loader not in ("ObjectNode", "CachedNode"):
            raise InvalidModelError("Invalid skops optimizer loader.")
        if (
            node.get("__module__") == "sklearn.tree._tree"
            and node.get("__class__") == "Tree"
            and loader not in ("TreeNode", "CachedNode")
        ):
            raise InvalidModelError("Invalid skops native tree loader.")
        if loader not in FIELDS:
            raise UnsupportedModelError((name,))
        allowed = (
            COMMON
            | FIELDS[loader]
            | ({"protocol", "_skops_version"} if node is root else set())
        )
        if set(node) - allowed:
            raise InvalidModelError("Unknown skops node state.")
        if loader in ("TreeNode", "LossNode") and set(node["__reduce__"]) != {"args"}:
            raise InvalidModelError("Unknown skops constructor state.")


def _special_loader(node, name, loader) -> None:
    if name in FUNCTIONS and loader not in ("FunctionNode", "CachedNode"):
        raise InvalidModelError("Invalid skops function loader.")
    if loader == "FunctionNode" and name not in FUNCTIONS:
        raise UnsupportedModelError((name,))
    if name == "numpy.dtype" and loader not in ("DTypeNode", "CachedNode"):
        raise InvalidModelError("Invalid skops dtype loader.")
    if loader == "DTypeNode" and name != "numpy.dtype":
        raise InvalidModelError("Invalid skops dtype identity.")
    if (
        loader == "TypeNode"
        and node.get("__module__") == "numpy"
        and name not in SCALAR_NAMES
    ):
        raise UnsupportedModelError((name,))
