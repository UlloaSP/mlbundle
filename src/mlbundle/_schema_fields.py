"""Closed inference-only mlschema 0.2.1 wire vocabulary; no optional imports."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from ._base import ContractModel, Name
from .limits import MAX_FEATURES

FieldLabel = Annotated[str, Field(min_length=1, max_length=100)]
MappedTo = Name | Annotated[int, Field(ge=0, lt=MAX_FEATURES)]


class FieldBase(ContractModel):
    label: FieldLabel
    required: bool
    mappedTo: MappedTo


class SimpleField(FieldBase):
    kind: Literal["text", "boolean", "date"]


class NumberField(FieldBase):
    kind: Literal["number"]
    step: Annotated[float, Field(gt=0)]


class CategoryField(FieldBase):
    kind: Literal["category"]
    options: Annotated[list[str], Field(min_length=1, max_length=MAX_FEATURES)]

    @model_validator(mode="after")
    def unique_options(self) -> CategoryField:
        if len(set(self.options)) != len(self.options):
            raise ValueError("Category options must be unique.")
        return self


ScalarField = Annotated[
    SimpleField | NumberField | CategoryField, Field(discriminator="kind")
]


class SeriesField(FieldBase):
    kind: Literal["series"]
    field1: ScalarField
    field2: ScalarField


class OneHotOption(ContractModel):
    label: Name
    value: Name
    mappedTo: MappedTo


class OneHotField(ContractModel):
    kind: Literal["onehot-category"]
    label: FieldLabel
    required: bool
    options: Annotated[list[OneHotOption], Field(min_length=2, max_length=MAX_FEATURES)]


SchemaField = Annotated[
    SimpleField | NumberField | CategoryField | SeriesField | OneHotField,
    Field(discriminator="kind"),
]
