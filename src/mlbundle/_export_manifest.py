"""Manifest construction shared by standalone and composite export."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from uuid import uuid4

from pydantic import JsonValue

from .claims import ModelClaims
from .contract import (
    FORMAT_NAME,
    FORMAT_VERSION,
    FeatureOrigin,
    Library,
    MemberName,
    SerializationFormat,
)
from .environment import capture_environment
from .training import TrainingDeclaration


def manifest_parts(
    claims: ModelClaims,
    library: Library,
    serialization: SerializationFormat,
    *,
    structure: dict[str, JsonValue] | None,
    feature_names: Sequence[str] | None,
    training: TrainingDeclaration | None,
    target_name: str | None,
    metadata: Mapping[str, JsonValue] | None,
) -> dict[str, object]:
    names = claims.feature_names
    origin = FeatureOrigin.MODEL
    if names is None:
        origin = (
            FeatureOrigin.DECLARED
            if feature_names is not None
            else FeatureOrigin.GENERATED
        )
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{index}" for index in range(1, claims.n_features + 1)]
        )
        if training is not None:
            names, origin = training.names, FeatureOrigin.DATAFRAME
    model = claims.model_dump(
        mode="json", exclude={"feature_names", "feature_dtypes", "class_labels", "onnx"}
    )
    model.update(library=library.value, serialization=serialization.value)
    return {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "bundle_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "writer_version": version("mlbundle"),
        "model": model,
        "features": {
            "names": names,
            "dtypes": [None] * claims.n_features
            if training is None
            else training.dtypes,
            "model_dtypes": claims.feature_dtypes,
            "origin": origin.value,
        },
        "target": {"class_labels": claims.class_labels, "name": target_name},
        "environment": capture_environment(onnx=claims.onnx).model_dump(mode="json"),
        "references": {
            "structure": None if structure is None else MemberName.STRUCTURE.value,
            "schema": None if training is None else MemberName.SCHEMA.value,
            "samples": None
            if training is None or training.samples is None
            else MemberName.SAMPLES.value,
        },
        "user_metadata": {} if metadata is None else dict(metadata),
    }
