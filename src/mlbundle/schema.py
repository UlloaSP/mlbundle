"""Schema/report and sample validation shared by export, inspect and load."""

from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from ._base import ContractModel
from ._schema_fields import OneHotField, SchemaField, SeriesField
from .canonical import DecodedJSON, canonical_dumps
from .claims import ModelClaims
from .contract import FeatureOrigin, MemberName, ModelKind
from .errors import InvalidSamplesError, InvalidSchemaError
from .limits import MAX_CLASSES, MAX_FEATURES, MAX_OUTPUTS, ArchiveLimits
from .manifest import Manifest

OneHotSeparator = Annotated[str, Field(min_length=1, max_length=16, strict=True)]
OutputPosition = Annotated[int, Field(ge=0, lt=MAX_OUTPUTS)]
ReportLabel = Annotated[str, Field(min_length=1, max_length=32)]
DisplayLabel = Annotated[str, Field(max_length=1024)]


def validate_separator(value: object) -> str:
    try:
        return TypeAdapter(OneHotSeparator).validate_python(value)
    except ValidationError:
        raise InvalidSchemaError(
            "onehot_separator must be a nonempty string of at most 16 characters."
        ) from None


class ClassifierReport(ContractModel):
    kind: Literal["classifier"]
    label: ReportLabel
    mappedTo: OutputPosition
    labels: Annotated[list[DisplayLabel], Field(min_length=1, max_length=MAX_CLASSES)]
    showClassProbabilities: Literal[True]


class RegressorReport(ContractModel):
    kind: Literal["regressor"]
    label: ReportLabel
    mappedTo: OutputPosition


Report = Annotated[ClassifierReport | RegressorReport, Field(discriminator="kind")]


class SchemaDeclaration(ContractModel):
    schema_version: Literal["2"]
    source: Literal["training_data", "default"]
    onehot_separator: OneHotSeparator
    fields: Annotated[list[SchemaField], Field(min_length=1, max_length=MAX_FEATURES)]
    reports: Annotated[list[Report], Field(min_length=1, max_length=MAX_OUTPUTS)]


def reports_for(claims: ModelClaims) -> list[dict[str, JsonValue]]:
    classifier = claims.kind == ModelKind.CLASSIFIER
    reports: list[dict[str, JsonValue]] = []
    label = "Predicted class" if classifier else "Predicted value"
    for output in range(claims.n_outputs):
        report: dict[str, JsonValue] = {
            "kind": claims.kind.value,
            "label": label if claims.n_outputs == 1 else f"{label} {output}",
            "mappedTo": output,
        }
        if classifier:
            labels = [str(value) for value in (claims.class_labels or [])[output]]
            if len(set(labels)) != len(labels):
                raise InvalidSchemaError(
                    "Class labels must have unique string forms per output."
                )
            report.update(
                labels=cast(list[JsonValue], labels), showClassProbabilities=True
            )
        reports.append(report)
    return reports


def positional_fields(fields: list[dict], names: list[str]) -> None:
    """Rewrite inferred name mappings, retaining mlschema's labels and grouping."""
    positions = {name: index for index, name in enumerate(names)}

    def rewrite(field: dict) -> None:
        if "mappedTo" in field:
            field["mappedTo"] = positions[field["mappedTo"]]
        for key in ("field1", "field2"):
            if key in field:
                rewrite(field[key])
        for option in field.get("options", []):
            if isinstance(option, dict):
                rewrite(option)

    for field in fields:
        rewrite(field)


def _mapping_index(
    mapping: str | int, positions: dict[str, int], origin: FeatureOrigin
) -> int:
    if origin == FeatureOrigin.MODEL:
        if type(mapping) is not str:
            raise ValueError
        return positions[mapping]
    if type(mapping) is not int or not 0 <= mapping < len(positions):
        raise ValueError
    return mapping


def _field_indices(
    field: SchemaField,
    names: list[str],
    positions: dict[str, int],
    origin: FeatureOrigin,
    separator: str,
) -> list[int]:
    if isinstance(field, OneHotField):
        indices = []
        for option in field.options:
            index = _mapping_index(option.mappedTo, positions, origin)
            if names[index].split(separator, 1) != [field.label, option.value]:
                raise ValueError
            if option.label != option.value:
                raise ValueError
            indices.append(index)
        return indices
    index = _mapping_index(field.mappedTo, positions, origin)
    if field.label != names[index]:
        raise ValueError
    if isinstance(field, SeriesField) and any(
        _mapping_index(sub.mappedTo, positions, origin) != index
        for sub in (field.field1, field.field2)
    ):
        raise ValueError
    return [index]


def validate_schema(
    value: JsonValue,
    names: list[str],
    claims: ModelClaims,
    *,
    origin: FeatureOrigin,
    target_names: list[str] | None,
    max_bytes: int,
) -> dict[str, JsonValue]:
    """Check declarations against resolved claims, without inferring historical truth."""
    if set(target_names or []) & set(names):
        raise InvalidSchemaError("Schema features cannot contain declared targets.")
    if isinstance(value, dict) and value.get("schema_version") != "2":
        raise InvalidSchemaError(
            "Unsupported schema version; bundle must be re-exported."
        )
    try:
        declaration = SchemaDeclaration.model_validate(value)
        positions = {name: index for index, name in enumerate(names)}
        mappings = [
            _field_indices(
                field, names, positions, origin, declaration.onehot_separator
            )
            for field in declaration.fields
        ]
        flattened = [index for group in mappings for index in group]
        if len(flattened) != len(names) or set(flattened) != set(range(len(names))):
            raise ValueError
        if any(group != sorted(group) for group in mappings):
            raise ValueError
        first_positions = [group[0] for group in mappings]
        if first_positions != sorted(first_positions):
            raise ValueError
        schema = cast(dict[str, JsonValue], value)
        if canonical_dumps(schema["reports"], max_bytes=max_bytes) != canonical_dumps(
            reports_for(claims), max_bytes=max_bytes
        ):
            raise ValueError
    except (ValidationError, ValueError, KeyError):
        raise InvalidSchemaError("Invalid schema fields or reports.") from None
    return cast(dict[str, JsonValue], value)


def validate_samples(value: JsonValue, names: list[str]) -> list[dict[str, JsonValue]]:
    expected = set(names)
    if not isinstance(value, list) or any(
        not isinstance(row, dict) or set(row) != expected for row in value
    ):
        raise InvalidSamplesError("Samples must contain exactly the declared features.")
    return cast(list[dict[str, JsonValue]], value)


def read_declarations(
    manifest: Manifest, members: dict[MemberName, DecodedJSON], limits: ArchiveLimits
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]] | None]:
    samples = None
    if MemberName.SAMPLES in members and set(manifest.target.names or []) & set(
        manifest.features.names
    ):
        raise InvalidSamplesError("Samples cannot contain the declared target.")
    schema = validate_schema(
        members[MemberName.SCHEMA].value,
        manifest.features.names,
        manifest.verified_claims(),
        origin=manifest.features.origin,
        target_names=manifest.target.names,
        max_bytes=limits.member_bytes(MemberName.SCHEMA),
    )
    if MemberName.SAMPLES in members:
        samples = validate_samples(
            members[MemberName.SAMPLES].value,
            manifest.features.names,
        )
    return schema, samples
