"""All native tree dump fields and wrapper feature importances, without truncation."""

import json
from typing import Any, cast

from pydantic import JsonValue

from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.errors import LimitExceededError
from mlbundle.limits import STRUCTURE_BYTES

from .native import MAX_TREES, configuration, tree_model, validate_payload
from .state import projection


def extract_structure_bytes(
    model: Any, *, document: dict[str, Any] | None = None
) -> bytes:
    booster = model.get_booster()
    native = tree_model(configuration(booster))
    count = int(native["gbtree_model_param"]["num_trees"])
    if not 1 <= count <= MAX_TREES:
        raise LimitExceededError("XGBoost tree count exceeds the limit.")
    if document is None:
        document, _ = validate_payload(bytes(booster.save_raw(raw_format="ubj")))
    native_model = document["learner"]["gradient_booster"]["model"]
    dumps = booster.get_dump(dump_format="json", with_stats=True)
    if (
        len(dumps) != count
        or sum(len(tree.encode()) for tree in dumps) + len(dumps) + 1 > STRUCTURE_BYTES
    ):
        raise LimitExceededError("XGBoost structure exceeds the limit.")
    return canonical_dumps(
        {
            "family": "xgboost",
            "native_model": projection(native_model),
            "trees": canonical_loads(
                ("[" + ",".join(dumps) + "]").encode(), max_bytes=STRUCTURE_BYTES
            ),
            "feature_importances": model.feature_importances_,
        },
        max_bytes=STRUCTURE_BYTES,
    )


def extract_structure(model: Any) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], json.loads(extract_structure_bytes(model)))
