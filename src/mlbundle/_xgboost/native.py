"""Closed native dimensions, objectives and recoverable config projection."""

from typing import Any

import numpy as np

from mlbundle.canonical import canonical_loads
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError, UnsupportedModelError
from mlbundle.limits import MAX_CLASSES, MAX_FEATURES, ArchiveLimits

from .arrays import array as _array, tree_indices as _tree_indices
from .categories import validate_categories, validate_encodings
from .state import attributes, native_keys
from .ubjson import decode

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
    if (
        objective == "multi:softprob"
        and int(learner["objective"]["softmax_multiclass_param"]["num_class"])
        != classes
    ):
        raise InvalidModelError("XGBoost objective and learner class counts disagree.")
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
    document = decode(data)
    native_keys(document)
    learner = document["learner"]
    dimensions(learner)
    model = tree_model(learner)["model"]
    validate_encodings(
        model["cats"], int(learner["learner_model_param"]["num_feature"])
    )
    count = int(model["gbtree_model_param"]["num_trees"])
    trees = model["trees"]
    if (
        not 1 <= count <= MAX_TREES
        or not isinstance(trees, list)
        or len(trees) != count
    ):
        raise InvalidModelError("Invalid or oversized XGBoost tree count.")
    _iterations(model, count, learner)
    categorical_bytes = 0
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
            "split_type",
        ):
            values = tree[key]
            length = len(values) if isinstance(values, list) else values.count
            if length != nodes:
                raise InvalidModelError("Invalid XGBoost node array dimensions.")
        _tree_indices(tree, nodes, int(learner["learner_model_param"]["num_feature"]))
        categorical_bytes += validate_categories(tree, nodes)
        if categorical_bytes > ArchiveLimits().max_member_bytes:
            raise InvalidModelError(
                "XGBoost categorical bitset bytes exceed the limit."
            )


def _iterations(model: dict[str, Any], count: int, learner: dict[str, Any]) -> None:
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
    parallel = int(model["gbtree_model_param"]["num_parallel_tree"])
    if not 1 <= parallel <= MAX_TREES or np.any(
        np.diff(pointers) != parallel * group_count
    ):
        raise InvalidModelError(
            "Invalid XGBoost parallel-tree or iteration dimensions."
        )
    attributes(learner["attributes"], len(pointers) - 1)
    weights = _array(model.get("weight_drop", []))
    if (
        weights.shape not in ((0,), (count,))
        or weights.dtype.kind not in "iuf"
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise InvalidModelError("Invalid XGBoost dropout weights.")
