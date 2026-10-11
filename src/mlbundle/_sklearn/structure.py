"""Unmodified native structure, with tree index validation before native traversal."""

from typing import Any, cast

import numpy as np
from pydantic import JsonValue
from sklearn.pipeline import Pipeline
from sklearn.tree._tree import Tree

from mlbundle.canonical import canonical_value
from mlbundle.errors import InvalidModelError, UnsupportedModelError
from mlbundle.limits import MAX_CLASSES, MAX_FEATURES, MAX_OUTPUTS, STRUCTURE_BYTES

_ARRAYS = (
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "value",
    "impurity",
    "n_node_samples",
    "weighted_n_node_samples",
)


def validate_tree(tree: Tree) -> None:
    count = tree.node_count
    if (
        count < 1
        or not 1 <= tree.n_features <= MAX_FEATURES
        or not 1 <= tree.n_outputs <= MAX_OUTPUTS
        or tree.n_classes.shape != (tree.n_outputs,)
        or np.any(tree.n_classes < 1)
        or np.any(tree.n_classes > MAX_CLASSES)
        or tree.value.shape != (count, tree.n_outputs, int(max(tree.n_classes)))
        or any(len(getattr(tree, name)) != count for name in _ARRAYS)
    ):
        raise InvalidModelError("Invalid sklearn tree dimensions.")
    # Visit each node exactly once: reject cycles, aliases, unreachable nodes and
    # out-of-range feature/child indices before feature_importances_ or inference.
    seen: set[int] = set()
    pending = [(0, 0)]
    depth = 0
    while pending:
        node, level = pending.pop()
        if not 0 <= node < count or node in seen:
            raise InvalidModelError("Invalid sklearn tree child indices.")
        seen.add(node)
        depth = max(depth, level)
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        if left == right == -1:
            continue
        if left < 0 or right < 0 or not 0 <= tree.feature[node] < tree.n_features:
            raise InvalidModelError("Invalid sklearn tree feature or child indices.")
        pending.extend(((left, level + 1), (right, level + 1)))
    if len(seen) != count or depth != tree.max_depth:
        raise InvalidModelError("Invalid sklearn tree reachability or depth.")


def validate_tree_storage(model: object) -> None:
    """Check all tree objects, including trees inside pipelines/meta-estimators."""
    seen: set[int] = set()
    pending = [model]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Tree):
            validate_tree(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
            pending.extend(value.flat)
        elif type(value).__module__.startswith("sklearn.") and hasattr(
            value, "__dict__"
        ):
            pending.extend(vars(value).values())


def _tree(tree: Tree) -> dict[str, object]:
    return {
        "node_count": tree.node_count,
        "n_features": tree.n_features,
        "n_outputs": tree.n_outputs,
        "n_classes": tree.n_classes,
        **{name: getattr(tree, name) for name in _ARRAYS},
    }


def extract_structure(model: Any) -> dict[str, JsonValue] | None:
    # Pipeline internals address transformed inputs; v1 has no transformation
    # mapping, so it makes no visualization claim for a pipeline.
    if isinstance(model, Pipeline):
        return None
    validate_tree_storage(model)
    result: dict[str, object] = {
        "feature_importances": getattr(model, "feature_importances_", None)
    }
    if hasattr(model, "tree_"):
        result.update(family="decision_tree", tree=_tree(model.tree_))
    elif hasattr(model, "estimators_"):
        estimators = np.asarray(model.estimators_, dtype=object)
        if all(hasattr(estimator, "tree_") for estimator in estimators.flat):
            result.update(
                family="tree_ensemble",
                estimator_shape=estimators.shape,
                trees=[_tree(estimator.tree_) for estimator in estimators.flat],
            )
        elif any(hasattr(estimator, "tree_") for estimator in estimators.flat):
            raise UnsupportedModelError(
                reason="Mixed tree/non-tree ensembles have no supported structure."
            )
    if (
        "family" not in result
        and hasattr(model, "coef_")
        and hasattr(model, "intercept_")
    ):
        result.update(family="linear", coef=model.coef_, intercept=model.intercept_)
    if "family" not in result:
        if result["feature_importances"] is None:
            return None
        result["family"] = "feature_importances"
    return cast(
        dict[str, JsonValue], canonical_value(result, max_bytes=STRUCTURE_BYTES)
    )
