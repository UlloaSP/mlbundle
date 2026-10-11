"""Schema/report and sample validation shared by export, inspect and load."""

from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, ValidationError

from ._base import ContractModel
from ._schema_fields import OneHotField, SchemaField, SeriesField
from .canonical import canonical_dumps, canonical_loads
from .claims import Label, ModelClaims
from .contract import MemberName, ModelKind
from .errors import InvalidSamplesError, InvalidSchemaError
from .limits import MAX_CLASSES, MAX_FEATURES, MAX_OUTPUTS, ArchiveLimits
from .manifest import Manifest

ONEHOT_SEPARATOR = "__"


class ClassifierReport(ContractModel):
    kind: Literal["classifier"]
    label: Literal["Predicted class"]
    mappedTo: Literal["classifier"]
    output: Annotated[int, Field(ge=0, lt=MAX_OUTPUTS)]
    labels: Annotated[list[Label], Field(min_length=1, max_length=MAX_CLASSES)]
    showClassProbabilities: Literal[True]


class RegressorReport(ContractModel):
    kind: Literal["regressor"]
    label: Literal["Predicted value"]
    mappedTo: Literal["regressor"]
    output: Annotated[int, Field(ge=0, lt=MAX_OUTPUTS)]


Report = Annotated[ClassifierReport | RegressorReport, Field(discriminator="kind")]


class SchemaDeclaration(ContractModel):
    schema_version: Literal["1"]
    fields: Annotated[list[SchemaField], Field(min_length=1, max_length=MAX_FEATURES)]
    reports: Annotated[list[Report], Field(min_length=1, max_length=MAX_OUTPUTS)]


def reports_for(claims: ModelClaims) -> list[dict[str, JsonValue]]:
    classifier = claims.kind == ModelKind.CLASSIFIER
    reports: list[dict[str, JsonValue]] = []
    for output in range(claims.n_outputs):
        report: dict[str, JsonValue] = {
            "kind": claims.kind.value,
            "label": "Predicted class" if classifier else "Predicted value",
            "mappedTo": claims.kind.value,
            "output": output,
        }
        if classifier:
            report.update(
                labels=list((claims.class_labels or [])[output]),
                showClassProbabilities=True,
            )
        reports.append(report)
    return reports


def _field_names(field: SchemaField) -> list[str]:
    if isinstance(field, OneHotField):
        names = []
        for option in field.options:
            if option.mappedTo.split(ONEHOT_SEPARATOR, 1) != [
                field.label,
                option.value,
            ]:
                raise ValueError
            if option.label != option.value:
                raise ValueError
            names.append(option.mappedTo)
        return names
    if field.label != field.mappedTo:
        raise ValueError
    if isinstance(field, SeriesField) and any(
        sub.mappedTo != field.mappedTo for sub in (field.field1, field.field2)
    ):
        raise ValueError
    return [field.mappedTo]


def validate_schema(
    value: JsonValue, names: list[str], claims: ModelClaims, *, max_bytes: int
) -> dict[str, JsonValue]:
    """Check declarations against resolved claims, without inferring historical truth."""
    try:
        declaration = SchemaDeclaration.model_validate(value)
        positions = {name: index for index, name in enumerate(names)}
        mappings = [_field_names(field) for field in declaration.fields]
        flattened = [name for group in mappings for name in group]
        if len(flattened) != len(names) or set(flattened) != set(names):
            raise ValueError
        if any(group != sorted(group, key=positions.__getitem__) for group in mappings):
            raise ValueError
        first_positions = [positions[group[0]] for group in mappings]
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
    manifest: Manifest, members: dict[MemberName, bytes], limits: ArchiveLimits
) -> tuple[dict[str, JsonValue] | None, list[dict[str, JsonValue]] | None]:
    schema = samples = None
    if (
        MemberName.SAMPLES in members
        and manifest.target.name in manifest.features.names
    ):
        raise InvalidSamplesError("Samples cannot contain the declared target.")
    if MemberName.SCHEMA in members:
        if manifest.target.name in manifest.features.names:
            raise InvalidSchemaError(
                "Schema features cannot contain the declared target."
            )
        schema = validate_schema(
            canonical_loads(
                members[MemberName.SCHEMA],
                max_bytes=limits.member_bytes(MemberName.SCHEMA),
            ),
            manifest.features.names,
            manifest.verified_claims(),
            max_bytes=limits.member_bytes(MemberName.SCHEMA),
        )
    if MemberName.SAMPLES in members:
        samples = validate_samples(
            canonical_loads(
                members[MemberName.SAMPLES],
                max_bytes=limits.member_bytes(MemberName.SAMPLES),
            ),
            manifest.features.names,
        )
    return schema, samples
