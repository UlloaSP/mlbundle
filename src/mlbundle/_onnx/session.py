"""Fixed runtime settings and agreement with its inferred input/output metadata."""

from typing import Any, cast

import onnxruntime as ort

from mlbundle.errors import InvalidModelError

from .description import Description

ORT_TYPES = {
    "float32": "tensor(float)",
    "float64": "tensor(double)",
    "int32": "tensor(int32)",
    "int64": "tensor(int64)",
    "string": "tensor(string)",
}


def _agree(expected: dict[str, Any], actual: Any) -> None:
    dtype = expected["dtype"]
    if dtype.startswith("sequence<map<"):
        dtype = dtype.removeprefix("sequence<map<").removesuffix(">>")
        key, value = dtype.split(",")
        native_type = f"seq(map({key},{ORT_TYPES[value]}))"
    else:
        native_type = ORT_TYPES[dtype]
    if expected["name"] != actual.name or native_type != actual.type:
        raise InvalidModelError("ONNX graph and runtime input/output types disagree.")
    shape = expected["shape"]
    if shape is not None and (
        actual.shape is None
        or len(shape) != len(actual.shape)
        or any(
            isinstance(a, int) and a != b
            for a, b in zip(shape, actual.shape, strict=True)
        )
    ):
        raise InvalidModelError(
            "ONNX graph and runtime input/output dimensions disagree."
        )


def create_session(data: bytes, description: Description) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_profiling = False
    options.optimized_model_filepath = ""
    options.log_severity_level = 4
    options.add_session_config_entry("session.load_model_format", "ONNX")
    options.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(
        data,
        sess_options=options,
        providers=["CPUExecutionProvider"],
        read_config_from_model=False,
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    expected_outputs = cast(
        list[dict[str, Any]], description.claims.hyperparameters["outputs"]
    )
    if len(inputs) != len(description.inputs) or len(outputs) != len(expected_outputs):
        raise InvalidModelError("ONNX graph and runtime input/output counts disagree.")
    for expected, actual in zip(description.inputs, inputs, strict=True):
        _agree(expected, actual)
    for expected, actual in zip(expected_outputs, outputs, strict=True):
        _agree(expected, actual)
        if actual.name == description.output and expected["shape"] is not None:
            width = (
                1
                if description.claims.n_classes is None
                else description.claims.n_classes[0]
            )
            if (
                len(actual.shape) == 2
                and isinstance(actual.shape[1], int)
                and actual.shape[1] != width
            ):
                raise InvalidModelError(
                    "ONNX runtime prediction width disagrees with native claims."
                )
    return session
