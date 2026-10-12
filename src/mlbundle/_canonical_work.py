"""Linear preflight for the expanded size/depth of native shared JSON containers."""

from typing import Any

from .errors import LimitExceededError


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
            children = item
        if size > max_bytes or depth > max_depth:
            raise LimitExceededError("Expanded JSON size or depth exceeds the limit.")
        for child in children:
            child_size, child_depth = measure(child)
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
