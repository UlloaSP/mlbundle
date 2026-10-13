"""Linear preflight for the expanded size/depth of native shared JSON containers."""

from typing import Any

from .errors import LimitExceededError

_SCALARS = (str, int, float, bool, type(None))
_NUMBERS = {int, float, bool, type(None)}


def _sequence_cost(item):
    # C map/set classify flat numeric arrays without Python calls per element.
    if len(item) >= 64 and set(map(type, item)) <= _NUMBERS:
        return 1 + 2 * len(item), ()
    return 1, item


def _child_cost(child: Any, measure) -> tuple[int, int]:
    if type(child) in _SCALARS:
        return (len(child) + 2 if type(child) is str else 1), 0
    return measure(child)


def _array_cost(value: Any) -> tuple[int, int, Any]:
    return value.ndim, value.size, value.flat if value.dtype.kind == "O" else ()


def bound_work(value: Any, max_bytes: int, max_depth: int) -> None:
    memo: dict[int, tuple[int, int]] = {}
    active: set[int] = set()

    def measure(item):
        array = (
            type(item).__module__.split(".")[0] == "numpy"
            and type(item).__name__ == "ndarray"
        )
        if not isinstance(item, (dict, list, tuple)) and not array:
            return (len(item) + 2 if isinstance(item, str) else 1), 0
        identity = id(item)
        if identity in active:
            raise LimitExceededError("Cyclic JSON containers exceed the nesting limit.")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        depth, size = 1, 1
        if array:
            depth, size, children = _array_cost(item)
        elif isinstance(item, dict):
            size += sum(len(key) + 3 for key in item if isinstance(key, str))
            children = item.values()
        else:
            size, children = _sequence_cost(item)
        if size > max_bytes or depth > max_depth:
            raise LimitExceededError("Expanded JSON size or depth exceeds the limit.")
        for child in children:
            child_size, child_depth = _child_cost(child, measure)
            size += child_size + 1
            depth = max(depth, child_depth + 1)
            if size > max_bytes or depth > max_depth:
                raise LimitExceededError(
                    "Expanded JSON size or depth exceeds the limit."
                )
        active.remove(identity)
        memo[identity] = size, depth
        return size, depth

    measure(value)
