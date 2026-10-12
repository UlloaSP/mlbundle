"""Recomputed native snapshots and commitments independent of CBM buffer layout."""

from hashlib import sha256
from typing import Any

import numpy as np

from mlbundle.canonical import canonical_dumps
from mlbundle.errors import InvalidModelError
from mlbundle.limits import MANIFEST_BYTES

from .native import Snapshot, decode


def snapshot(model: Any) -> Snapshot:
    # CatBoost has no public in-memory save. This audited native CBM serializer
    # lets extraction recompute loaded state without reader-side disk writes.
    return decode(model._serialize_model())


def commitment(state: Snapshot) -> str:
    def digest(value: Any) -> str:
        if isinstance(value, np.ndarray):
            header = canonical_dumps([value.dtype.str, list(value.shape)])
            result = sha256(b"array:")
            result.update(header)
            result.update(memoryview(value))
            return result.hexdigest()
        if isinstance(value, dict):
            record = {key: digest(child) for key, child in value.items()}
            return sha256(b"dict:" + canonical_dumps(record)).hexdigest()
        if isinstance(value, list):
            return sha256(
                b"list:" + canonical_dumps([digest(child) for child in value])
            ).hexdigest()
        return sha256(
            b"scalar:" + canonical_dumps(value, max_bytes=MANIFEST_BYTES)
        ).hexdigest()

    return digest({"native_model": state.native, "metadata": state.metadata})


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_value(child) for child in value]
    return value


def feature_importances(model: Any, width: int) -> list[float]:
    values = np.asarray(
        model.copy().get_feature_importance(
            type="PredictionValuesChange", thread_count=1
        ),
        dtype=float,
    )
    if values.shape != (width,) or not np.isfinite(values).all() or np.any(values < 0):
        raise InvalidModelError("Invalid CatBoost feature importances.")
    return values.tolist()
