"""Manifest construction shared by standalone and composite export."""

from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from uuid import uuid4

from pydantic import JsonValue

from .claims import ModelClaims
from .contract import (
    FORMAT_NAME,
    FORMAT_VERSION,
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
    training: TrainingDeclaration,
    target_names: list[str] | None,
    metadata: Mapping[str, JsonValue] | None,
) -> dict[str, object]:
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
            "names": training.names,
            "dtypes": training.dtypes,
            "model_dtypes": claims.feature_dtypes,
            "origin": training.origin.value,
        },
        "target": {"class_labels": claims.class_labels, "names": target_names},
        "environment": capture_environment(onnx=claims.onnx).model_dump(mode="json"),
        "references": {
            "structure": None if structure is None else MemberName.STRUCTURE.value,
            "schema": MemberName.SCHEMA.value,
            "samples": None if training.samples is None else MemberName.SAMPLES.value,
        },
        "user_metadata": {} if metadata is None else dict(metadata),
    }
