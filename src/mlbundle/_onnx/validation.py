"""Self-contained inference graph validation before checker or runtime loading."""

from math import prod
from typing import Any

import onnx

from mlbundle.errors import InvalidModelError, LimitExceededError, MLBundleError
from mlbundle.limits import MAX_FEATURES, ArchiveLimits

from .protobuf import validate_wire

DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml"})
MAX_NODES = 100_000
MAX_RANK = 32
BYTE_CAP = ArchiveLimits().max_member_bytes
# Storage field, bytes per element, primitive values per element.
TENSOR_TYPES = {
    1: ("float_data", 4, 1),
    2: ("int32_data", 1, 1),
    3: ("int32_data", 1, 1),
    4: ("int32_data", 2, 1),
    5: ("int32_data", 2, 1),
    6: ("int32_data", 4, 1),
    7: ("int64_data", 8, 1),
    8: ("string_data", 0, 1),
    9: ("int32_data", 1, 1),
    10: ("int32_data", 2, 1),
    11: ("double_data", 8, 1),
    12: ("uint64_data", 4, 1),
    13: ("uint64_data", 8, 1),
    14: ("float_data", 8, 2),
    15: ("double_data", 16, 2),
    16: ("int32_data", 2, 1),
}


def _dimensions(dims: Any, *, allow_zero: bool) -> int:
    if len(dims) > MAX_RANK or any(
        dimension < (0 if allow_zero else 1) or dimension > BYTE_CAP
        for dimension in dims
    ):
        raise LimitExceededError("ONNX tensor dimensions exceed the limit.")
    count = prod(dims)
    if count > BYTE_CAP:
        raise LimitExceededError("ONNX tensor element count exceeds the limit.")
    return count


def _tensor(tensor: Any) -> int:
    if tensor.data_location != onnx.TensorProto.DEFAULT or tensor.external_data:
        raise InvalidModelError("ONNX external tensor data is unsupported.")
    if tensor.HasField("segment") or tensor.data_type not in TENSOR_TYPES:
        raise InvalidModelError("Unsupported ONNX tensor storage.")
    count = _dimensions(tensor.dims, allow_zero=True)
    field, size, factor = TENSOR_TYPES[tensor.data_type]
    if count * max(size, 1) > BYTE_CAP:
        raise LimitExceededError("ONNX tensor storage exceeds the limit.")
    populated = [
        name
        for name in {value[0] for value in TENSOR_TYPES.values()}
        if len(getattr(tensor, name))
    ]
    if tensor.HasField("raw_data"):
        valid = size > 0 and not populated and len(tensor.raw_data) == count * size
    else:
        valid = (
            all(name == field for name in populated)
            and len(getattr(tensor, field)) == count * factor
        )
    if not valid:
        raise InvalidModelError("ONNX tensor dimensions and stored values disagree.")
    return max(count * max(size, 1), len(tensor.raw_data))


def _shape(shape: Any) -> None:
    if len(shape.dim) > MAX_RANK:
        raise LimitExceededError("ONNX value rank exceeds the limit.")
    static = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            static.append(dim.dim_value)
        elif len(dim.dim_param) > 1024:
            raise LimitExceededError("ONNX dim_param exceeds the 1024-character limit.")
    _dimensions(static, allow_zero=True)


def _operator(value: Any) -> None:
    if value.domain not in DOMAINS:
        raise InvalidModelError("Unsupported ONNX operator domain.")
    if value.DESCRIPTOR.name == "NodeProto" and (
        value.overload or value.device_configurations
    ):
        raise InvalidModelError("Unsupported ONNX node configuration.")


def _sparse(value: Any) -> None:
    count = _dimensions(value.dims, allow_zero=True)
    size = max(TENSOR_TYPES.get(value.values.data_type, (None, BYTE_CAP, 1))[1], 1)
    if count * size > BYTE_CAP:
        raise LimitExceededError("ONNX sparse tensor dense storage exceeds the limit.")


def _children(value: Any) -> list[Any]:
    children = []
    for field, child in value.ListFields():
        if field.type == field.TYPE_MESSAGE:
            children.extend(child if field.is_repeated else [child])
    return children


def _visit(model: Any) -> None:
    pending = [model]
    nodes = storage = 0
    while pending:
        value = pending.pop()
        name = value.DESCRIPTOR.name
        if name == "TensorProto":
            storage += _tensor(value)
            if storage > BYTE_CAP:
                raise LimitExceededError(
                    "ONNX aggregate tensor storage exceeds the limit."
                )
        elif name == "SparseTensorProto":
            _sparse(value)
        elif name == "TensorShapeProto":
            _shape(value)
        elif name in ("NodeProto", "FunctionProto", "OperatorSetIdProto"):
            _operator(value)
            if name == "NodeProto":
                nodes += 1
                if nodes > MAX_NODES:
                    raise LimitExceededError("ONNX node count exceeds the limit.")
        pending.extend(_children(value))


def parse_model(data: bytes) -> Any:
    try:
        validate_wire(data, onnx.ModelProto.DESCRIPTOR)
        model = onnx.ModelProto()
        model.ParseFromString(data)
        _visit(model)
        if model.training_info or model.functions or model.configuration:
            raise InvalidModelError(
                "ONNX training, local functions and device configuration are unsupported."
            )
        if any(item.key == "ort_config" for item in model.metadata_props):
            raise InvalidModelError(
                "ONNX runtime configuration metadata is unsupported."
            )
        if not 1 <= len(model.graph.input) <= MAX_FEATURES:
            raise InvalidModelError("ONNX requires bounded tabular inputs.")
        initializers = {item.name for item in model.graph.initializer} | {
            item.values.name for item in model.graph.sparse_initializer
        }
        if initializers & {item.name for item in model.graph.input}:
            raise InvalidModelError("ONNX overridable initializers are unsupported.")
        # full_check=False avoids shape inference executing constant propagation.
        onnx.checker.check_model(model, full_check=False)
        return model
    except MLBundleError:
        raise
    except Exception:
        raise InvalidModelError("Invalid ONNX protobuf or graph.") from None
