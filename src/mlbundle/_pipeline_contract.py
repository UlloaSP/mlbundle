"""Core-only composite records; declarations never select Python constructors."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from ._base import ContractModel
from .canonical import canonical_dumps
from .claims import FeatureNames, ModelClaims, validate_labels
from .contract import MODEL_MEMBERS, Library, MemberName, SerializationFormat
from .limits import HYPERPARAMETER_BYTES, MAX_FEATURES

StepName = Annotated[str, Field(min_length=1, max_length=256)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OPTIONS = {"steps", "memory", "verbose", "transform_input"}


def valid_step_names(names: list[str]) -> bool:
    return len(set(names)) == len(names) and all(
        "__" not in name and name not in OPTIONS for name in names
    )


class TransformerInfo(ContractModel):
    member: Literal["model.skops"]
    estimator_class: Literal["Pipeline"]
    hyperparameters: dict[str, JsonValue] = Field(max_length=4096)
    state_sha256: Digest

    @field_validator("hyperparameters")
    @classmethod
    def bounded_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        canonical_dumps(value, max_bytes=HYPERPARAMETER_BYTES, max_depth=16)
        return value


class NativeComponentInfo(ContractModel):
    library: Literal[Library.XGBOOST, Library.CATBOOST]
    serialization: SerializationFormat
    member: MemberName
    claims: ModelClaims

    @model_validator(mode="after")
    def native_pairing(self) -> NativeComponentInfo:
        if (self.serialization, self.member) != MODEL_MEMBERS[self.library]:
            raise ValueError("Invalid native component pairing.")
        classes = {
            Library.XGBOOST: {"XGBClassifier", "XGBRegressor"},
            Library.CATBOOST: {"CatBoostClassifier", "CatBoostRegressor"},
        }
        if self.claims.estimator_class not in classes[self.library]:
            raise ValueError("Invalid native component class.")
        if self.claims.onnx is not None:
            raise ValueError("Composite components cannot declare ONNX fields.")
        validate_labels(self.claims, self.claims.class_labels)
        return self


class TransformedFeatures(ContractModel):
    n_features: int = Field(gt=0, le=MAX_FEATURES)
    names: FeatureNames
    origin: Literal["transformer", "model", "input", "generated"]

    @model_validator(mode="after")
    def aligned_names(self) -> TransformedFeatures:
        if len(self.names) != self.n_features or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError("Invalid transformed names.")
        if self.origin == "generated" and self.names != [
            f"transformed_feature_{i}" for i in range(1, self.n_features + 1)
        ]:
            raise ValueError("Invalid generated transformed names.")
        return self


class PipelineInfo(ContractModel):
    step_names: list[StepName] = Field(min_length=1, max_length=128)
    transformers: TransformerInfo | None
    estimator: NativeComponentInfo
    transformed_features: TransformedFeatures

    @model_validator(mode="after")
    def consistent_recipe(self) -> PipelineInfo:
        if not valid_step_names(self.step_names):
            raise ValueError("Invalid Pipeline names.")
        if (len(self.step_names) > 1) != (self.transformers is not None):
            raise ValueError("Transformer recipe and steps disagree.")
        if self.transformed_features.n_features != self.estimator.claims.n_features:
            raise ValueError("Composite boundary widths disagree.")
        return self


def expected_model_members(
    library: Library, pipeline: PipelineInfo | None
) -> set[MemberName]:
    if pipeline is None:
        return {MODEL_MEMBERS[library][1]}
    members = {pipeline.estimator.member}
    if pipeline.transformers is not None:
        members.add(MemberName.SKOPS)
    return members
