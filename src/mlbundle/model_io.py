"""Shared export and validating load; adapters own all native-model behavior."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from os import PathLike
from typing import cast
from uuid import uuid4

from pydantic import JsonValue

from ._container_rules import MEMBER_ORDER, BundleSource, effective_limits
from .canonical import canonical_dumps, canonical_loads
from .claims import Label, ModelClaims, verify_claims, verify_structure
from .container import read_container
from .contract import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MODEL_MEMBERS,
    FeatureOrigin,
    Library,
    MemberName,
    SerializationFormat,
)
from .declarations import ExportDeclarations, export_declarations, resolve_labels
from .environment import capture_environment
from .errors import ClaimMismatchError, InvalidModelError, LimitExceededError
from .inspection import inspect
from .limits import ArchiveLimits
from .registry import find_adapter, get_adapter
from .results import LoadedBundle, MetadataView
from .writing import write_container


def _manifest_parts(
    claims: ModelClaims,
    library: Library,
    serialization: SerializationFormat,
    *,
    structure: dict[str, JsonValue] | None,
    feature_names: Sequence[str] | None,
    target_name: str | None,
    metadata: Mapping[str, JsonValue] | None,
) -> dict[str, object]:
    names = claims.feature_names
    origin = FeatureOrigin.MODEL
    if names is None:
        origin = (
            FeatureOrigin.DATAFRAME
            if feature_names is not None
            else FeatureOrigin.GENERATED
        )
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{index}" for index in range(1, claims.n_features + 1)]
        )
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
            "dtypes": [None] * claims.n_features,
            "model_dtypes": claims.feature_dtypes,
            "origin": origin.value,
        },
        "target": {"class_labels": claims.class_labels, "name": target_name},
        "environment": capture_environment(onnx=claims.onnx).model_dump(mode="json"),
        "references": {
            "structure": None if structure is None else MemberName.STRUCTURE.value
        },
        "user_metadata": {} if metadata is None else dict(metadata),
    }


def export(
    model: object,
    destination: str | PathLike[str],
    *,
    feature_names: Sequence[str] | None = None,
    class_labels: Sequence[Sequence[Label]] | None = None,
    target_name: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
    overwrite: bool = False,
    limits: ArchiveLimits | None = None,
) -> MetadataView:
    """Publish a native bundle only after its claims survive serialization/reloading.

    Model-exposed feature names take precedence over supplied names. Otherwise
    supplied names are declarations; omitted names become feature_1 … feature_N.
    Class labels are output-major declarations handled by the native adapter;
    omitted labels use native labels or generated encoded indices.
    No training rows, schema or sample data are included by this API.
    """
    selected = effective_limits(limits)
    declarations = export_declarations(class_labels)
    adapter = find_adapter(model)
    model = adapter.prepare_export(model, declarations)
    data = adapter.serialize(model)
    if len(data) > selected.member_bytes(adapter.model_member):
        raise LimitExceededError("Model member size limit exceeded.")
    claims = resolve_labels(adapter.extract_claims(model), declarations)
    structure = adapter.extract_structure(model)
    restored = adapter.deserialize(data)
    verify_claims(
        claims, resolve_labels(adapter.extract_claims(restored), declarations)
    )
    verify_structure(structure, adapter.extract_structure(restored))
    members = {adapter.model_member: data}
    if structure is not None:
        members[MemberName.STRUCTURE] = canonical_dumps(
            structure, max_bytes=selected.member_bytes(MemberName.STRUCTURE)
        )
    parts = _manifest_parts(
        claims,
        adapter.library,
        MODEL_MEMBERS[adapter.library][0],
        structure=structure,
        feature_names=feature_names,
        target_name=target_name,
        metadata=metadata,
    )
    result = write_container(
        parts, members, destination=destination, overwrite=overwrite, limits=selected
    )
    return inspect(result, limits=selected)


def _optional_json(
    members: dict[MemberName, bytes], name: MemberName, limits: ArchiveLimits
) -> JsonValue:
    if name not in members:
        return None
    return canonical_loads(members[name], max_bytes=limits.member_bytes(name))


def load(source: BundleSource, *, limits: ArchiveLimits | None = None) -> LoadedBundle:
    """Verify format, integrity and recomputed native claims before returning a model."""
    selected = effective_limits(limits)
    manifest, members = read_container(source, limits=selected)
    adapter = get_adapter(manifest.model.library)
    model = adapter.deserialize(members[adapter.model_member])
    actual = adapter.extract_claims(model)
    expected = manifest.verified_claims()
    # Check native dimensions before using any label declaration from the bundle.
    verify_claims(
        expected.model_copy(update={"class_labels": actual.class_labels}), actual
    )
    actual = resolve_labels(
        actual,
        ExportDeclarations(
            class_labels=manifest.target.class_labels
            if actual.class_labels is None
            else None
        ),
    )
    verify_claims(expected, actual)
    structure = _optional_json(members, MemberName.STRUCTURE, selected)
    extracted = adapter.extract_structure(model)
    if (MemberName.STRUCTURE in members) != (extracted is not None):
        raise ClaimMismatchError("structure")
    verify_structure(structure, extracted)
    schema = _optional_json(members, MemberName.SCHEMA, selected)
    samples = _optional_json(members, MemberName.SAMPLES, selected)
    if MemberName.SCHEMA in members and (
        not isinstance(schema, dict)
        or not isinstance(schema.get("fields"), list)
        or not isinstance(schema.get("reports"), list)
    ):
        raise InvalidModelError("Schema member must contain fields and reports arrays.")
    if MemberName.SAMPLES in members and (
        not isinstance(samples, list)
        or any(
            not isinstance(row, dict) or set(row) != set(manifest.features.names)
            for row in samples
        )
    ):
        raise InvalidModelError("Samples member must contain row objects.")
    return LoadedBundle(
        manifest=manifest,
        members=tuple(name for name in MEMBER_ORDER if name in members),
        native_model=model,
        predictor=adapter.predictor(model),
        schema=cast(dict[str, JsonValue] | None, schema),
        structure=cast(dict[str, JsonValue] | None, structure),
        samples=cast(list[dict[str, JsonValue]] | None, samples),
    )
