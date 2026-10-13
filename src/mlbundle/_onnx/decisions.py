"""Graph-owned label positions, including final ZipMap renaming."""

from typing import Any

from mlbundle.claims import ClassLabels
from mlbundle.errors import MLBundleError

from .labels import LABEL_OPS, labels_for

_LABEL_PRESERVING = frozenset({"Identity", "Cast", "Reshape", "Squeeze", "Unsqueeze"})


def _label_node(producers: dict, output: str, steps: int) -> Any:
    for _ in range(steps):
        node = producers.get(output)
        if node is None:
            return None
        if (
            node.domain in ("", "ai.onnx")
            and node.op_type in _LABEL_PRESERVING
            and node.input
        ):
            output = node.input[0]
        else:
            return node, output
    return None


def _same_columns(producers: dict, output: str, source: str, steps: int) -> bool:
    for _ in range(steps + 1):
        if output == source:
            return True
        node = producers.get(output)
        if node is None or len(node.input) != 1:
            return False
        if not (
            (node.domain in ("", "ai.onnx") and node.op_type in ("Identity", "Cast"))
            or (node.domain == "ai.onnx.ml" and node.op_type == "ZipMap")
        ):
            return False
        output = node.input[0]
    return False


def decision_labels(
    graph: Any, output: str, probability: str, count: int
) -> ClassLabels | None:
    producers = {name: node for node in graph.node for name in node.output if name}
    steps = len(graph.node)
    found = _label_node(producers, output, steps)
    if found is None:
        return None
    node, source_output = found
    if node.domain == "ai.onnx.ml" and node.op_type in LABEL_OPS - {"ZipMap"}:
        if (
            len(node.output) != 2
            or source_output != node.output[0]
            or not _same_columns(producers, probability, node.output[1], steps)
        ):
            return None
        try:
            return labels_for([node])
        except (MLBundleError, UnicodeError):
            # Unrecoverable identities still undergo ordinary known-class checks.
            return None
    if node.domain in ("", "ai.onnx") and node.op_type == "ArgMax" and node.input:
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        if axis in (1, -1) and _same_columns(
            producers, probability, node.input[0], steps
        ):
            return [list(range(count))]
    return None
