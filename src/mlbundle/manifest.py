"""Version 1 manifest validation and canonical JSON boundaries."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, JsonValue, ValidationError, field_validator, model_validator

from ._base import ContractModel, Name, ShortString
from .canonical import canonical_dumps, canonical_loads
from .claims import (
    ClassLabels,
    EstimatorInfo,
    FeatureDtypes,
    FeatureNames,
    ModelClaims,
    validate_labels,
)
from .contract import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MODEL_MEMBERS,
    FeatureOrigin,
    Library,
    MemberName,
    SerializationFormat,
)
from .environment import TrainingEnvironment
from .errors import (
    InvalidJSONError,
    InvalidManifestError,
    UnsupportedFormatVersionError,
)
from .integrity import IntegrityRecord
from .limits import METADATA_BYTES, ArchiveLimits

_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|\+00:00)"
)


class ModelInfo(EstimatorInfo):
    library: Library
    serialization: SerializationFormat

    @model_validator(mode="after")
    def native_serialization(self) -> ModelInfo:
        if self.serialization != MODEL_MEMBERS[self.library][0]:
            raise ValueError("Library and serialization must agree.")
        return self


class FeatureInfo(ContractModel):
    names: FeatureNames
    dtypes: list[ShortString | None] = Field(min_length=1, max_length=100_000)
    model_dtypes: FeatureDtypes | None = None
    origin: FeatureOrigin

    @model_validator(mode="after")
    def aligned_features(self) -> FeatureInfo:
        if len(self.names) != len(self.dtypes) or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError("Feature names and dtypes must align and names be unique.")
        if self.model_dtypes is not None and len(self.model_dtypes) != len(self.names):
            raise ValueError("Model dtypes must align with feature names.")
        if self.origin == FeatureOrigin.GENERATED and self.names != [
            f"feature_{index}" for index in range(1, len(self.names) + 1)
        ]:
            raise ValueError("Generated names must follow the format convention.")
        return self


class TargetInfo(ContractModel):
    class_labels: ClassLabels | None
    name: Name | None = None


class MemberReferences(ContractModel):
    schema_: Literal["schema.json"] | None = Field(default=None, alias="schema")
    structure: Literal["structure.json"] | None = None
    samples: Literal["samples.json"] | None = None


class Manifest(ContractModel):
    format_name: Literal["mlbundle"]
    format_version: Literal["1"]
    bundle_id: UUID
    created_at: datetime
    writer_version: ShortString
    model: ModelInfo
    features: FeatureInfo
    target: TargetInfo
    environment: TrainingEnvironment
    integrity: dict[MemberName, IntegrityRecord] = Field(min_length=1, max_length=4)
    references: MemberReferences
    user_metadata: dict[str, JsonValue] = Field(default_factory=dict, max_length=1024)

    @field_validator("format_version", mode="before")
    @classmethod
    def supported_version(cls, value: object) -> object:
        if isinstance(value, str) and value != FORMAT_VERSION:
            raise UnsupportedFormatVersionError
        return value

    @field_validator("created_at")
    @classmethod
    def utc_time(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Creation time must have a UTC offset.")
        return value

    @field_validator("created_at", mode="before")
    @classmethod
    def timestamp_spelling(cls, value: object) -> object:
        if isinstance(value, str):
            if _UTC_TIMESTAMP.fullmatch(value) is None:
                raise ValueError("Creation time must be an ISO UTC timestamp.")
            return datetime.fromisoformat(value)
        return value

    @field_validator("user_metadata")
    @classmethod
    def bounded_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        canonical_dumps(value, max_bytes=METADATA_BYTES, max_depth=8)
        return value

    @model_validator(mode="after")
    def consistent_manifest(self) -> Manifest:
        if len(self.features.names) != self.model.n_features:
            raise ValueError("Feature count and names must agree.")
        validate_labels(self.model, self.target.class_labels)
        expected = {MODEL_MEMBERS[self.model.library][1]}
        expected.update(
            MemberName(member)
            for member in self.references.model_dump().values()
            if member is not None
        )
        if set(self.integrity) != expected:
            raise ValueError("Integrity must cover exactly the referenced members.")
        limits = ArchiveLimits()
        if any(
            record.size > limits.member_bytes(name)
            for name, record in self.integrity.items()
        ):
            raise ValueError("A member exceeds its format size cap.")
        if (self.model.library == Library.ONNX) != (self.environment.onnx is not None):
            raise ValueError("ONNX graph metadata is required only for ONNX models.")
        if self.environment.packages.get("mlbundle") != self.writer_version:
            raise ValueError("Writer and environment mlbundle versions must agree.")
        return self

    def verified_claims(self) -> ModelClaims:
        fields = self.model.model_dump(exclude={"library", "serialization"})
        return ModelClaims(
            **fields,
            feature_names=(
                self.features.names
                if self.features.origin == FeatureOrigin.MODEL
                else None
            ),
            class_labels=self.target.class_labels,
            feature_dtypes=self.features.model_dtypes,
            onnx=self.environment.onnx,
        )

    def to_bytes(self) -> bytes:
        return canonical_dumps(self.model_dump(mode="json", by_alias=True))

    @classmethod
    def from_bytes(cls, data: bytes) -> Manifest:
        try:
            value = canonical_loads(data)
            if not isinstance(value, dict) or value.get("format_name") != FORMAT_NAME:
                raise InvalidManifestError
            version = value.get("format_version")
            if isinstance(version, str) and version != FORMAT_VERSION:
                raise UnsupportedFormatVersionError
            return cls.model_validate_json(canonical_dumps(value))
        except (ValidationError, InvalidJSONError):
            raise InvalidManifestError from None
