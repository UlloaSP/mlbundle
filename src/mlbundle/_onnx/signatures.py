"""Supported tabular graph signatures and feature input layouts."""

from typing import Any

from mlbundle.errors import InvalidModelError
from mlbundle.limits import MAX_FEATURES

DTYPES = {1: "float32", 11: "float64", 6: "int32", 7: "int64"}


def signature(value: Any) -> dict[str, Any]:
    typ = value.type
    if typ.HasField("tensor_type"):
        tensor = typ.tensor_type
        if not tensor.HasField("shape"):
            raise InvalidModelError("ONNX input/output ranks must be declared.")
        dtype = {**DTYPES, 8: "string"}.get(tensor.elem_type)
        if dtype is None:
            raise InvalidModelError("Unsupported ONNX input/output dtype.")
        shape = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in tensor.shape.dim
        ]
        return {"name": value.name, "dtype": dtype, "shape": shape}
    if typ.HasField("sequence_type"):
        element = typ.sequence_type.elem_type
        if element.HasField("map_type"):
            mapping = element.map_type
            dtype = DTYPES.get(mapping.value_type.tensor_type.elem_type)
            if mapping.key_type in (7, 8) and dtype in ("float32", "float64"):
                key = "int64" if mapping.key_type == 7 else "string"
                return {
                    "name": value.name,
                    "dtype": f"sequence<map<{key},{dtype}>>",
                    "shape": None,
                }
    raise InvalidModelError("Unsupported ONNX input/output layout.")


def _batch(shape: list[Any]) -> bool:
    return bool(shape) and (not isinstance(shape[0], int) or shape[0] == 1)


def _inputs(graph: Any) -> tuple[dict[str, Any], ...]:
    inputs = tuple(signature(value) for value in graph.input)
    for value in inputs:
        shape = value["shape"]
        if (
            value["dtype"] not in DTYPES.values()
            or shape is None
            or len(shape) != 2
            or not _batch(shape)
            or not isinstance(shape[1], int)
            or not 1 <= shape[1] <= MAX_FEATURES
        ):
            raise InvalidModelError(
                "ONNX inputs require numeric matrices with fixed feature width and dynamic or one-row batches."
            )
        if len(inputs) > 1 and shape[1] != 1:
            raise InvalidModelError(
                "Multiple ONNX inputs require one column per input."
            )
    return inputs
