"""Operation-local immutable native bytes; mutable model identity is never cached."""

import json
from typing import Any

from pydantic import JsonValue

from .native import validate_payload
from .structure import extract_structure_bytes


class Snapshots:
    def __init__(self) -> None:
        self._payloads: list[tuple[bytes, dict[str, Any], memoryview]] = []
        self._structures: list[tuple[tuple, bytes]] = []

    def payload(self, data: bytes) -> tuple[dict[str, Any], memoryview]:
        for stored, document, model_bytes in self._payloads:
            if data == stored:
                return document, model_bytes
        document, model_bytes = validate_payload(data)
        self._payloads = [*self._payloads[-1:], (data, document, model_bytes)]
        return document, model_bytes

    def structure(self, model: Any) -> dict[str, JsonValue]:
        document, native_bytes = self.payload(
            bytes(model.get_booster().save_raw(raw_format="ubj"))
        )
        learner = document["learner"]
        key = (
            native_bytes,
            tuple(learner["feature_names"]),
            tuple(learner["feature_types"]),
            model.importance_type or "gain",
        )
        for stored, encoded in self._structures:
            if key == stored:
                return json.loads(encoded)
        encoded = extract_structure_bytes(model, document=document)
        self._structures = [*self._structures[-1:], (key, encoded)]
        return json.loads(encoded)
