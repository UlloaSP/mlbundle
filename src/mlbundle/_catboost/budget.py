"""Bound structure expansion before native parsing or allocating Python lists."""

from typing import Any

from mlbundle.errors import LimitExceededError
from mlbundle.limits import STRUCTURE_BYTES


def structure_budget(native: dict[str, Any]) -> None:
    # Worst finite binary64 JSON token plus separator is 25 bytes. The bound
    # intentionally rejects some compact-value models before list allocation.
    size = 4096
    for value in native.values():
        if hasattr(value, "dtype"):
            size += value.size * (25 if value.dtype.kind == "f" else 12)
    for feature in native["float_features"]:
        size += 256 + 6 * len(feature["name"].encode("utf-8"))
        size += feature["borders"].size * 25
    if size > STRUCTURE_BYTES:
        raise LimitExceededError(
            "CatBoost structure allocation budget exceeds the limit."
        )
