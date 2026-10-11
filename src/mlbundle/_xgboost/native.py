"""Closed native dimensions, objectives and recoverable config projection."""

from typing import Any

import numpy as np

from mlbundle.canonical import canonical_loads
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError, UnsupportedModelError
from mlbundle.limits import MAX_CLASSES, MAX_FEATURES, ArchiveLimits

from .ubjson import Buffer, decode

ATTRIBUTE = "mlbundle"
MAX_TREES = 100_000
CLASSIFIERS = frozenset({"binary:logistic", "multi:softprob"})
REGRESSORS = frozenset(
    {
        "reg:squarederror",
        "reg:absoluteerror",
        "reg:pseudohubererror",
        "reg:squaredlogerror",
        "reg:logistic",
        "count:poisson",
        "reg:gamma",
        "reg:tweedie",
    }
)


def dimensions(learner: dict[str, Any]) -> tuple[ModelKind, int, int | None]:
    params = learner["learner_model_param"]
    width, classes, outputs = (
        int(params[key]) for key in ("num_feature", "num_class", "num_target")
    )
    if (
        not 1 <= width <= MAX_FEATURES
        or not 0 <= classes <= MAX_CLASSES
        or outputs != 1
    ):
        raise UnsupportedModelError(
            reason="XGBoost requires bounded features/classes and a single target."
        )
    objective = learner["objective"]["name"]
    if objective in CLASSIFIERS:
        if (objective == "binary:logistic" and classes != 0) or (
            objective == "multi:softprob" and classes < 2
        ):
            raise InvalidModelError("Invalid XGBoost classification dimensions.")
        return ModelKind.CLASSIFIER, width, max(2, classes)
    if objective in REGRESSORS and classes == 0:
        return ModelKind.REGRESSOR, width, None
    raise UnsupportedModelError(
        reason="Unsupported XGBoost objective; only probabilities and regression are supported."
    )


def tree_model(learner: dict[str, Any]) -> dict[str, Any]:
    booster = learner["gradient_booster"]
    if booster["name"] != "gbtree":
        raise UnsupportedModelError(
            reason="Only XGBoost gbtree (including native dropout) is supported."
        )
    return booster


def configuration(booster: Any) -> dict[str, Any]:
    value = canonical_loads(booster.save_config().encode())
    learner = value.get("learner") if isinstance(value, dict) else None
    if not isinstance(learner, dict):
        raise InvalidModelError("Invalid XGBoost native configuration.")
    return learner


def parameters(learner: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    gb = tree_model(learner)
    return {
        "learner_model_param": learner["learner_model_param"],
        "objective": learner["objective"],
        "gradient_booster": {
            "name": gb["name"],
            "gbtree_model_param": gb["gbtree_model_param"],
        },
        "prediction": settings,
    }


def validate_payload(data: bytes) -> None:
    if len(data) > ArchiveLimits().max_member_bytes:
        raise InvalidModelError("XGBoost payload exceeds the byte limit.")
    if not data.startswith(b"{L\x00\x00\x00\x00\x00\x00\x00\x07learner{"):
        raise InvalidModelError("Expected a native XGBoost UBJSON model.")
    learner = decode(data)["learner"]
    dimensions(learner)
    model = tree_model(learner)["model"]
    count = int(model["gbtree_model_param"]["num_trees"])
    trees = model["trees"]
    if (
        not 1 <= count <= MAX_TREES
        or not isinstance(trees, list)
        or len(trees) != count
    ):
        raise InvalidModelError("Invalid or oversized XGBoost tree count.")
    groups = _array(model["tree_info"])
    group_count = max(1, int(learner["learner_model_param"]["num_class"]))
    if (
        groups.shape != (count,)
        or groups.dtype.kind not in "iu"
        or np.any(groups < 0)
        or np.any(groups >= group_count)
    ):
        raise InvalidModelError("Invalid XGBoost tree output-group indices.")
    pointers = _array(model["iteration_indptr"])
    if (
        pointers.ndim != 1
        or not 2 <= len(pointers) <= count + 1
        or pointers.dtype.kind not in "iu"
        or pointers[0] != 0
        or pointers[-1] != count
        or np.any(pointers[1:] <= pointers[:-1])
    ):
        raise InvalidModelError("Invalid XGBoost iteration pointers.")
    for tree in trees:
        nodes = int(tree["tree_param"]["num_nodes"])
        if (
            nodes < 1
            or int(tree["tree_param"]["size_leaf_vector"]) != 1
            or int(tree["tree_param"]["num_feature"])
            != int(learner["learner_model_param"]["num_feature"])
            or int(tree["tree_param"]["num_deleted"]) != 0
        ):
            raise InvalidModelError("Unsupported XGBoost tree dimensions.")
        for key in (
            "left_children",
            "right_children",
            "split_indices",
            "split_conditions",
            "parents",
            "default_left",
            "loss_changes",
            "sum_hessian",
            "base_weights",
        ):
            values = tree[key]
            length = len(values) if isinstance(values, list) else values.count
            if length != nodes:
                raise InvalidModelError("Invalid XGBoost node array dimensions.")
        _tree_indices(tree, nodes, int(learner["learner_model_param"]["num_feature"]))


def _array(value: Any) -> np.ndarray:
    if isinstance(value, Buffer):
        formats = {
            "i": "i1",
            "U": "u1",
            "I": ">i2",
            "l": ">i4",
            "L": ">i8",
            "d": ">f4",
            "D": ">f8",
        }
        return np.frombuffer(value.data, dtype=formats[value.marker])
    return np.asarray(value)


def _tree_indices(tree: dict[str, Any], nodes: int, width: int) -> None:
    left, right, features = (
        _array(tree[key])
        for key in ("left_children", "right_children", "split_indices")
    )
    if any(array.dtype.kind not in "iu" for array in (left, right, features)):
        raise InvalidModelError("Invalid XGBoost tree index types.")
    seen = bytearray(nodes)
    pending = [(0, 0)]
    while pending:
        node, depth = pending.pop()
        if node < 0 or node >= nodes or seen[node] or depth > 14:
            raise InvalidModelError("Invalid XGBoost tree reachability or depth.")
        seen[node] = 1
        a, b = int(left[node]), int(right[node])
        if a == b == -1:
            continue
        if a < 0 or b < 0 or not 0 <= features[node] < width:
            raise InvalidModelError("Invalid XGBoost tree child or feature indices.")
        pending.extend(((a, depth + 1), (b, depth + 1)))
    if not all(seen):
        raise InvalidModelError("Unreachable XGBoost tree nodes.")
