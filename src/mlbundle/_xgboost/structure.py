"""All native tree dump fields and wrapper feature importances, without truncation."""

from typing import Any, cast

from pydantic import JsonValue

from mlbundle.canonical import canonical_loads, canonical_value
from mlbundle.errors import LimitExceededError
from mlbundle.limits import STRUCTURE_BYTES

from .native import MAX_TREES, configuration, tree_model


def extract_structure(model: Any) -> dict[str, JsonValue]:
    booster = model.get_booster()
    native = tree_model(configuration(booster))
    count = int(native["gbtree_model_param"]["num_trees"])
    if not 1 <= count <= MAX_TREES:
        raise LimitExceededError("XGBoost tree count exceeds the limit.")
    dumps = booster.get_dump(dump_format="json", with_stats=True)
    if (
        len(dumps) != count
        or sum(len(tree.encode()) for tree in dumps) > STRUCTURE_BYTES
    ):
        raise LimitExceededError("XGBoost structure exceeds the limit.")
    return cast(
        dict[str, JsonValue],
        canonical_value(
            {
                "family": "xgboost",
                "trees": [
                    canonical_loads(tree.encode(), max_bytes=STRUCTURE_BYTES)
                    for tree in dumps
                ],
                "feature_importances": model.feature_importances_,
            },
            max_bytes=STRUCTURE_BYTES,
        ),
    )
