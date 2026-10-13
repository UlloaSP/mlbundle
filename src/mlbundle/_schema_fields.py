"""Closed inferred vocabulary with MLForm series columns; no optional imports."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from ._base import ContractModel, Name
from .limits import MAX_FEATURES

FieldLabel = Annotated[str, Field(min_length=1, max_length=100)]
MappedTo = Name | Annotated[int, Field(ge=0, lt=MAX_FEATURES)]


class ScalarBase(ContractModel):
    label: FieldLabel
    required: bool


class FieldBase(ScalarBase):
    mappedTo: MappedTo


class SimpleProperties(ScalarBase):
    kind: Literal["text", "boolean", "date"]


class NumberProperties(ScalarBase):
    kind: Literal["number"]
    step: Annotated[float, Field(gt=0)]


class CategoryProperties(ScalarBase):
    kind: Literal["category"]
    options: Annotated[list[str], Field(min_length=1, max_length=MAX_FEATURES)]

    @model_validator(mode="after")
    def unique_options(self) -> CategoryProperties:
        if len(set(self.options)) != len(self.options):
            raise ValueError("Category options must be unique.")
        return self


class SimpleField(SimpleProperties):
    mappedTo: MappedTo


class NumberField(NumberProperties):
    mappedTo: MappedTo


class CategoryField(CategoryProperties):
    mappedTo: MappedTo


class SimpleColumn(SimpleProperties):
    id: Literal["field1", "field2"]


class NumberColumn(NumberProperties):
    id: Literal["field1", "field2"]


class CategoryColumn(CategoryProperties):
    id: Literal["field1", "field2"]


SeriesColumn = Annotated[
    SimpleColumn | NumberColumn | CategoryColumn, Field(discriminator="kind")
]


class SeriesField(FieldBase):
    kind: Literal["series"]
    columns: Annotated[list[SeriesColumn], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def ordered_columns(self) -> SeriesField:
        if [column.id for column in self.columns] != ["field1", "field2"]:
            raise ValueError("Series columns must be field1 then field2.")
        return self


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
