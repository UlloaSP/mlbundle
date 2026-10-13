"""Bound skops' recursive audit work before it builds or audits shared nodes."""

from typing import Any

from mlbundle.canonical import canonical_dumps
from mlbundle.errors import InvalidModelError

MAX_NODES = 100_000
MAX_REFERENCES = 100_000
MAX_VISITS = 100_000
MAX_DEPTH = 32


def _without_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_ids(child) for key, child in value.items() if key != "__id__"
        }
    if isinstance(value, list):
        return [_without_ids(child) for child in value]
    return value


class References:
    def __init__(self, schema: Any) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.memo: dict[int, dict[str, Any]] = {}
        pending = [schema]
        self.reference_count = 0
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if "__loader__" in value:
                    self.nodes.append(value)
                    if len(self.nodes) > MAX_NODES:
                        raise InvalidModelError("Skops node count exceeds the limit.")
                    self._record(value)
                pending.extend(reversed(list(value.values())))
            elif isinstance(value, list):
                pending.extend(reversed(value))
        self._work(schema)

    def _record(self, value: dict[str, Any]) -> None:
        identity = value.get("__id__")
        if identity is not None and (type(identity) is not int or identity <= 0):
            raise InvalidModelError("Invalid skops object identity.")
        if value["__loader__"] == "CachedNode":
            self.reference_count += 1
            if self.reference_count > MAX_REFERENCES:
                raise InvalidModelError("Skops reference count exceeds the limit.")
        elif identity is not None:
            previous = self.memo.get(identity)
            if previous is not None and (
                previous["__loader__"] != value["__loader__"]
                or (
                    value["__loader__"] in ("JsonNode", "NdArrayNode")
                    and canonical_dumps(_without_ids(previous))
                    != canonical_dumps(_without_ids(value))
                )
            ):
                raise InvalidModelError("Conflicting skops object references.")
            self.memo.setdefault(identity, value)

    def resolve(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise InvalidModelError("Invalid skops node reference.")
        if value.get("__loader__") == "CachedNode":
            identity = value.get("__id__")
            if type(identity) is not int:
                raise InvalidModelError("Invalid skops object identity.")
            target = self.memo.get(identity)
            if target is None:
                raise InvalidModelError("Unknown skops object reference.")
            if (target.get("__module__"), target.get("__class__")) != (
                value.get("__module__"),
                value.get("__class__"),
            ):
                raise InvalidModelError("Conflicting skops reference type.")
            return target
        return value

    def _work(self, schema: Any) -> None:
        memo: dict[int, tuple[int, int]] = {}
        active: set[int] = set()

        def cost(value):
            if not isinstance(value, (dict, list)):
                return 0, 0
            value = self.resolve(value) if isinstance(value, dict) else value
            identity = id(value)
            if identity in active:
                raise InvalidModelError("Cyclic skops object references.")
            if identity in memo:
                return memo[identity]
            active.add(identity)
            visits, depth = 1, 1
            children = value.values() if isinstance(value, dict) else value
            for child in children:
                child_visits, child_depth = cost(child)
                visits += child_visits
                depth = max(depth, child_depth + 1)
                if visits > MAX_VISITS or depth > MAX_DEPTH:
                    raise InvalidModelError(
                        "Skops expanded node visits or depth exceed the limit."
                    )
            active.remove(identity)
            memo[identity] = visits, depth
            return visits, depth

        cost(schema)
