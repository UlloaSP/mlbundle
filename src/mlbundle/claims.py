"""One representation for extracted claims and deterministic comparison."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, JsonValue, field_validator, model_validator

from ._base import ContractModel, Name, ShortString
from .canonical import canonical_dumps
from .contract import ModelKind
from .environment import OnnxInfo
from .errors import ClaimMismatchError
from .limits import (
    HYPERPARAMETER_BYTES,
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_OUTPUTS,
    STRUCTURE_BYTES,
)

Label = Annotated[str, Field(max_length=1024)] | int | float | bool
LabelSet = Annotated[list[Label], Field(min_length=1, max_length=MAX_CLASSES)]
ClassLabels = Annotated[list[LabelSet], Field(min_length=1, max_length=MAX_OUTPUTS)]
ClassCounts = Annotated[
    list[Annotated[int, Field(gt=0, le=MAX_CLASSES)]],
    Field(min_length=1, max_length=MAX_OUTPUTS),
]
FeatureNames = Annotated[list[Name], Field(min_length=1, max_length=MAX_FEATURES)]
FeatureDtypes = Annotated[
    list[ShortString], Field(min_length=1, max_length=MAX_FEATURES)
]


class EstimatorInfo(ContractModel):
    estimator_class: ShortString
    kind: ModelKind
    hyperparameters: dict[str, JsonValue] = Field(max_length=4096)
    n_features: int = Field(gt=0, le=MAX_FEATURES)
    n_classes: ClassCounts | None
    n_outputs: int = Field(gt=0, le=MAX_OUTPUTS)

    @field_validator("hyperparameters")
    @classmethod
    def bounded_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        canonical_dumps(value, max_bytes=HYPERPARAMETER_BYTES, max_depth=16)
        return value

    @model_validator(mode="after")
    def counts_match_kind(self) -> EstimatorInfo:
        if self.kind == ModelKind.CLASSIFIER:
            if self.n_classes is None or len(self.n_classes) != self.n_outputs:
                raise ValueError("Classifier counts must describe each output.")
        elif self.n_classes is not None:
            raise ValueError("Regressors cannot declare class counts.")
        return self


def validate_labels(info: EstimatorInfo, labels: ClassLabels | None) -> None:
    if info.kind == ModelKind.REGRESSOR:
        if labels is not None:
            raise ValueError("Regressors cannot declare class labels.")
        return
    if labels is None or info.n_classes != [len(output) for output in labels]:
        raise ValueError("Class labels must match the output class counts.")
    for output in labels:
        encoded = [canonical_dumps(label) for label in output]
        if len(encoded) != len(set(encoded)):
            raise ValueError("Class labels must be unique within an output.")


class ModelClaims(EstimatorInfo):
    feature_names: FeatureNames | None
    feature_dtypes: FeatureDtypes | None = None
    class_labels: ClassLabels | None
    onnx: OnnxInfo | None = None

    @model_validator(mode="after")
    def consistent_claims(self) -> ModelClaims:
        validate_labels(self, self.class_labels)
        if self.feature_names is not None and (
            len(self.feature_names) != self.n_features
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("Model feature names must match the feature count.")
        if (
            self.feature_dtypes is not None
            and len(self.feature_dtypes) != self.n_features
        ):
            raise ValueError("Model dtypes must match the feature count.")
        return self


def verify_claims(expected: ModelClaims, actual: ModelClaims) -> None:
    expected_fields = expected.model_dump(mode="json")
    actual_fields = actual.model_dump(mode="json")
    for name, value in expected_fields.items():
        if canonical_dumps(value) != canonical_dumps(actual_fields[name]):
            raise ClaimMismatchError(name)


def verify_structure(expected: JsonValue, actual: JsonValue) -> None:
    if canonical_dumps(expected, max_bytes=STRUCTURE_BYTES) != canonical_dumps(
        actual, max_bytes=STRUCTURE_BYTES
    ):
        raise ClaimMismatchError("structure")
