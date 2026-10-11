"""Unmodified native structure, with tree index validation before native traversal."""

from typing import Any, cast

import numpy as np
from pydantic import JsonValue
from sklearn.base import BaseEstimator, is_classifier
from sklearn.multioutput import RegressorChain
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
            if isinstance(value, BaseEstimator):
                _validate_owner(value)
            pending.extend(vars(value).values())


def _width(model: Any) -> int | None:
    if not hasattr(model, "n_features_in_"):
        return None
    width = model.n_features_in_
    if (
        not isinstance(width, (int, np.integer))
        or isinstance(width, (bool, np.bool_))
        or not 1 <= width <= MAX_FEATURES
    ):
        raise InvalidModelError("Invalid sklearn estimator input width.")
    return int(width)


def _validate_owner(model: Any) -> None:
    width = _width(model)
    if hasattr(model, "tree_") and (
        not isinstance(model.tree_, Tree) or model.tree_.n_features != width
    ):
        raise InvalidModelError(
            "Sklearn tree and owning estimator input widths disagree."
        )
    if hasattr(model, "tree_") and (
        not isinstance(model.n_outputs_, (int, np.integer))
        or isinstance(model.n_outputs_, (bool, np.bool_))
        or model.n_outputs_ != model.tree_.n_outputs
    ):
        raise InvalidModelError(
            "Sklearn tree and owning estimator output counts disagree."
        )
    if is_classifier(model) and (
        hasattr(model, "tree_") or hasattr(model, "estimators_")
    ):
        _validate_classes(model)
    if width is None or not hasattr(model, "estimators_"):
        return
    estimators = list(np.asarray(model.estimators_, dtype=object).flat)
    feature_sets = getattr(model, "estimators_features_", None)
    if feature_sets is not None and len(feature_sets) != len(estimators):
        raise InvalidModelError("Invalid ensemble feature mappings.")
    for index, estimator in enumerate(estimators):
        expected = width + index if isinstance(model, RegressorChain) else width
        if feature_sets is not None:
            features = np.asarray(feature_sets[index])
            if (
                features.ndim != 1
                or features.dtype.kind not in "iu"
                or len(features) == 0
                or np.any(features < 0)
                or np.any(features >= width)
            ):
                raise InvalidModelError("Invalid ensemble feature mappings.")
            expected = len(features)
        actual = _width(estimator)
        if actual is not None and actual != expected:
            raise InvalidModelError(
                "Ensemble and constituent estimator input widths disagree."
            )


def _validate_classes(model: Any) -> None:
    classes = model.classes_ if isinstance(model.classes_, list) else [model.classes_]
    if not classes or any(np.asarray(labels).ndim != 1 for labels in classes):
        raise InvalidModelError("Invalid classifier label dimensions.")
    counts = np.asarray([len(labels) for labels in classes])
    if hasattr(model, "n_outputs_") and model.n_outputs_ != len(classes):
        raise InvalidModelError("Classifier labels and native output counts disagree.")
    if hasattr(model, "n_classes_"):
        native = np.asarray(model.n_classes_)
        if (
            native.dtype.kind not in "iu"
            or native.ndim != (0 if len(classes) == 1 else 1)
            or not np.array_equal(np.atleast_1d(native), counts)
        ):
            raise InvalidModelError(
                "Classifier labels and native class counts disagree."
            )
    if hasattr(model, "tree_") and not np.array_equal(model.tree_.n_classes, counts):
        raise InvalidModelError("Classifier labels and tree class counts disagree.")


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
