"""One extraction owner for native ONNX class-label attributes."""

from typing import Any, cast

from mlbundle.canonical import canonical_dumps
from mlbundle.claims import ClassLabels
from mlbundle.errors import InvalidModelError

LABEL_OPS = frozenset(
    {"ZipMap", "LinearClassifier", "TreeEnsembleClassifier", "SVMClassifier"}
)


def labels_for(nodes: list[Any]) -> ClassLabels | None:
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
