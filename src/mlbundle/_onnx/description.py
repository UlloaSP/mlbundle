"""Native graph signatures, output contracts and original label claims."""

from dataclasses import dataclass
from typing import Any, cast

from pydantic import JsonValue

from mlbundle.canonical import canonical_dumps
from mlbundle.claims import ClassLabels, ModelClaims
from mlbundle.contract import ModelKind
from mlbundle.environment import OnnxInfo, Opset
from mlbundle.errors import InvalidModelError
from mlbundle.limits import MAX_CLASSES

from .signatures import _batch, _inputs, signature

LABEL_OPS = frozenset(
    {"ZipMap", "LinearClassifier", "TreeEnsembleClassifier", "SVMClassifier"}
)


@dataclass(frozen=True)
class Description:
    claims: ModelClaims
    inputs: tuple[dict[str, Any], ...]
    output: str
    maps: bool


def _ancestors(graph: Any, output: str) -> list[Any]:
    producers = {name: node for node in graph.node for name in node.output if name}
    pending, seen, result = [output], set(), []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        node = producers.get(name)
        if node is not None:
            result.append(node)
            pending.extend(node.input)
    return result


def _labels(nodes: list[Any]) -> ClassLabels | None:
    found = []
    for node in nodes:
        if node.domain != "ai.onnx.ml" or node.op_type not in LABEL_OPS:
            continue
        for attr in node.attribute:
            if attr.name == "classlabels_strings":
                found.append([value.decode("utf-8") for value in attr.strings])
            elif attr.name in ("classlabels_ints", "classlabels_int64s"):
                found.append(list(attr.ints))
    if not found:
        return None
    if any(
        canonical_dumps(labels) != canonical_dumps(found[0]) for labels in found[1:]
    ):
        raise InvalidModelError("ONNX native label records disagree.")
    return cast(ClassLabels, [found[0]])


def _native_labels(graph: Any, output: str) -> ClassLabels | None:
    producers = {name: node for node in graph.node for name in node.output if name}
    original = output
    for _ in range(len(graph.node)):
        node = producers.get(output)
        if node is None:
            return None
        if node.domain == "ai.onnx.ml" and node.op_type in LABEL_OPS:
            if node.op_type == "ZipMap" or (
                len(node.output) > 1 and output == node.output[1]
            ):
                return _labels([node])
            break
        if (
            node.domain in ("", "ai.onnx")
            and node.op_type in ("Identity", "Cast")
            and len(node.input) == 1
        ):
            output = node.input[0]
        else:
            break
    if any(
        node.domain == "ai.onnx.ml" and node.op_type in LABEL_OPS
        for node in _ancestors(graph, original)
    ):
        raise InvalidModelError(
            "Unsupported transformation of ONNX labeled probabilities."
        )
    return None


def _softmax(graph: Any, output: str) -> bool:
    producers = {name: node for node in graph.node for name in node.output if name}
    for _ in range(len(graph.node)):
        node = producers.get(output)
        if node is None or node.domain not in ("", "ai.onnx"):
            return False
        if node.op_type == "Softmax":
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
            return axis in (1, -1)
        if node.op_type not in ("Identity", "Cast") or len(node.input) != 1:
            return False
        output = node.input[0]
    return False


def _probability(value: dict[str, Any]) -> bool:
    if value["dtype"].startswith("sequence<map<"):
        return True
    shape = value["shape"]
    return (
        value["dtype"] in ("float32", "float64")
        and shape is not None
        and len(shape) == 2
        and _batch(shape)
        and (not isinstance(shape[1], int) or 1 <= shape[1] <= MAX_CLASSES)
    )


def _label_output(value: dict[str, Any]) -> bool:
    shape = value["shape"]
    return (
        value["dtype"] in ("int64", "string")
        and shape is not None
        and _batch(shape)
        and (len(shape) == 1 or (len(shape) == 2 and shape[1] == 1))
    )


def _regression_width(graph: Any, output: str) -> None:
    producers = {name: node for node in graph.node for name in node.output if name}
    for _ in range(len(graph.node)):
        node = producers.get(output)
        if node is None:
            return
        if node.domain == "ai.onnx.ml" and node.op_type in (
            "LinearRegressor",
            "TreeEnsembleRegressor",
        ):
            attr_name = "targets" if node.op_type == "LinearRegressor" else "n_targets"
            width = next(
                (attr.i for attr in node.attribute if attr.name == attr_name), 1
            )
            if width != 1:
                raise InvalidModelError(
                    "ONNX native regression operator must expose one target."
                )
            return
        if (
            node.domain not in ("", "ai.onnx")
            or node.op_type not in ("Identity", "Cast")
            or len(node.input) != 1
        ):
            return
        output = node.input[0]


def _outputs(
    graph: Any, outputs: list[dict[str, Any]]
) -> tuple[ModelKind, dict[str, Any], ClassLabels | None, int | None]:
    if len(outputs) == 2:
        candidates = [
            value
            for index, value in enumerate(outputs)
            if _probability(value) and _label_output(outputs[1 - index])
        ]
    elif len(outputs) == 1:
        value = outputs[0]
        evidence = _native_labels(graph, value["name"]) is not None or _softmax(
            graph, value["name"]
        )
        candidates = [value] if _probability(value) and evidence else []
    else:
        candidates = []
    if len(candidates) == 1:
        value = candidates[0]
        labels = _native_labels(graph, value["name"])
        count = value["shape"][1] if value["shape"] is not None else None
        if not isinstance(count, int):
            count = len(labels[0]) if labels else None
        if count is None or (labels is not None and len(labels[0]) != count):
            raise InvalidModelError(
                "ONNX probability width and native labels disagree."
            )
        return ModelKind.CLASSIFIER, value, labels, count
    if len(outputs) == 1:
        value = outputs[0]
        shape = value["shape"]
        if (
            value["dtype"] in ("float32", "float64")
            and shape is not None
            and _batch(shape)
            and (len(shape) == 1 or (len(shape) == 2 and shape[1] == 1))
        ):
            _regression_width(graph, value["name"])
            return ModelKind.REGRESSOR, value, None, None
    raise InvalidModelError(
        "ONNX kind or probability/regression output layout is indeterminate."
    )


def describe(model: Any) -> Description:
    graph = model.graph
    if not 1 <= len(graph.output) <= 2:
        raise InvalidModelError(
            "ONNX requires one regression or probability output, optionally with labels."
        )
    inputs = _inputs(graph)
    outputs = [signature(value) for value in graph.output]
    kind, output, labels, count = _outputs(graph, outputs)
    width = sum(value["shape"][1] for value in inputs)
    metadata = {item.key: item.value for item in model.metadata_props}
    if len(metadata) != len(model.metadata_props):
        raise InvalidModelError("Duplicate ONNX metadata keys.")
    claims = ModelClaims(
        estimator_class="ONNX",
        kind=kind,
        hyperparameters=cast(
            dict[str, JsonValue],
            {"inputs": list(inputs), "outputs": outputs, "metadata_props": metadata},
        ),
        n_features=width,
        n_classes=None if count is None else [count],
        n_outputs=1,
        feature_names=[value["name"] for value in inputs] if len(inputs) > 1 else None,
        feature_dtypes=[
            value["dtype"] for value in inputs for _ in range(value["shape"][1])
        ],
        class_labels=labels,
        onnx=OnnxInfo(
            opsets=[
                Opset(domain=item.domain, version=item.version)
                for item in sorted(model.opset_import, key=lambda item: item.domain)
            ],
            ir_version=model.ir_version,
            producer_name=model.producer_name,
            producer_version=model.producer_version,
        ),
    )
    return Description(claims, inputs, output["name"], output["shape"] is None)
