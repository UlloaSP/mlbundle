"""Closed booster attributes and complete native prediction-state projections."""

import math
from typing import Any

import numpy as np

from mlbundle.canonical import canonical_loads
from mlbundle.errors import InvalidModelError

from .ubjson import Buffer

ATTRIBUTES = frozenset({"mlbundle", "best_iteration", "best_score", "scikit_learn"})
TREE_FIELDS = frozenset(
    {
        "base_weights",
        "categories",
        "categories_nodes",
        "categories_segments",
        "categories_sizes",
        "default_left",
        "id",
        "left_children",
        "loss_changes",
        "parents",
        "right_children",
        "split_conditions",
        "split_indices",
        "split_type",
        "sum_hessian",
        "tree_param",
    }
)
OBJECTIVE_FIELDS = {
    "binary:logistic": ("reg_loss_param", "scale_pos_weight"),
    "multi:softprob": ("softmax_multiclass_param", "num_class"),
    "reg:squarederror": ("reg_loss_param", "scale_pos_weight"),
    "reg:gamma": ("reg_loss_param", "scale_pos_weight"),
    "reg:logistic": ("reg_loss_param", "scale_pos_weight"),
    "reg:absoluteerror": None,
    "reg:squaredlogerror": None,
    "reg:pseudohubererror": ("pseudo_huber_param", "huber_slope"),
    "reg:tweedie": ("tweedie_regression_param", "tweedie_variance_power"),
    "count:poisson": ("poisson_regression_param", "max_delta_step"),
}


def keys(value: Any, allowed: set[str] | frozenset[str]) -> None:
    if not isinstance(value, dict) or set(value) != allowed:
        raise InvalidModelError("Unknown or missing XGBoost native state.")


def attributes(value: Any, rounds: int | None = None) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) - ATTRIBUTES
        or any(type(item) is not str for item in value.values())
    ):
        raise InvalidModelError("Unknown XGBoost booster attribute.")
    if "best_iteration" in value:
        text = value["best_iteration"]
        if (
            not text.isascii()
            or not text.isdecimal()
            or len(text) > 6
            or (rounds is not None and not 0 <= int(text) < rounds)
        ):
            raise InvalidModelError("Invalid XGBoost best iteration.")
    if "best_score" in value and (
        len(value["best_score"]) > 256 or not math.isfinite(float(value["best_score"]))
    ):
        raise InvalidModelError("Invalid XGBoost best score.")
    if "scikit_learn" in value:
        record = canonical_loads(value["scikit_learn"].encode())
        if (
            not isinstance(record, dict)
            or set(record) != {"_estimator_type"}
            or record["_estimator_type"] not in ("classifier", "regressor")
        ):
            raise InvalidModelError("Invalid XGBoost sklearn identity attribute.")
    return value


def native_keys(document: dict[str, Any]) -> None:
    keys(document, {"learner", "version"})
    learner = document["learner"]
    keys(
        learner,
        {
            "attributes",
            "feature_names",
            "feature_types",
            "gradient_booster",
            "learner_model_param",
            "objective",
        },
    )
    keys(
        learner["learner_model_param"],
        {"base_score", "boost_from_average", "num_class", "num_feature", "num_target"},
    )
    objective = learner["objective"]
    if isinstance(objective, dict) and objective.get("name") in OBJECTIVE_FIELDS:
        fields = OBJECTIVE_FIELDS[objective["name"]]
        keys(objective, {"name"} | ({fields[0]} if fields else set()))
        if fields:
            keys(objective[fields[0]], {fields[1]})
    booster = learner["gradient_booster"]
    keys(booster, {"name", "model"})
    native = booster["model"]
    keys(
        native,
        {"cats", "gbtree_model_param", "iteration_indptr", "tree_info", "trees"}
        | ({"weight_drop"} if "weight_drop" in native else set()),
    )
    keys(native["gbtree_model_param"], {"num_parallel_tree", "num_trees"})
    keys(native["cats"], {"enc", "feature_segments", "sorted_idx"})
    for encoding in native["cats"]["enc"]:
        if set(encoding) == {"offsets", "values"}:
            keys(encoding, {"offsets", "values"})
        else:
            keys(encoding, {"type", "values"})
    for tree in native["trees"]:
        keys(tree, TREE_FIELDS)
        keys(
            tree["tree_param"],
            {"num_deleted", "num_feature", "num_nodes", "size_leaf_vector"},
        )


def projection(value: Any) -> Any:
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
    if isinstance(value, dict):
        return {key: projection(child) for key, child in value.items()}
    if isinstance(value, list):
        return [projection(child) for child in value]
    return value
