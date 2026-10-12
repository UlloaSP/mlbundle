"""Native numeric views and tree index checks before XGBoost parsing."""

from typing import Any

import numpy as np

from mlbundle.errors import InvalidModelError

from .ubjson import Buffer


def array(value: Any) -> np.ndarray:
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


def tree_indices(tree: dict[str, Any], nodes: int, width: int) -> None:
    left, right, features = (
        array(tree[key]) for key in ("left_children", "right_children", "split_indices")
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
