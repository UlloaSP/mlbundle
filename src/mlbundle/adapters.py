"""Adapter and predictor contracts, independent of optional library types."""

from typing import Protocol

from pydantic import JsonValue

from .claims import ModelClaims
from .contract import Library, MemberName
from .declarations import ExportDeclarations

RegressionOutput = list[float] | list[list[float]]
ProbabilityOutput = list[list[float]] | list[list[list[float]]]


class Predictor(Protocol):
    # Optional record_input(row, names) hook preserves native selector semantics;
    # without that hook, records use one numeric row in manifest feature order.
    def predict(self, data: object) -> RegressionOutput: ...
    def predict_proba(self, data: object) -> ProbabilityOutput: ...
    # Row-major native decisions as indices, or None when no native decision exists.
    def predict_class_indices(self, data: object) -> list[list[int]] | None: ...


class Adapter(Protocol):
    @property
    def library(self) -> Library: ...
    @property
    def model_member(self) -> MemberName: ...
    def recognizes(self, model: object) -> bool: ...
    def prepare_export(
        self, model: object, declarations: ExportDeclarations
    ) -> object: ...
    def serialize(self, model: object) -> bytes: ...
    def deserialize(self, data: bytes) -> object: ...
    def extract_claims(self, model: object) -> ModelClaims: ...
    def extract_structure(self, model: object) -> dict[str, JsonValue] | None: ...
    # Shared load supplies verified/resolved claims, including declared ONNX labels.
    def predictor(
        self, model: object, claims: ModelClaims | None = None
    ) -> Predictor: ...
