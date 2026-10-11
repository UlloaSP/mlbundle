"""Shared strict model configuration and bounded field types."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

ShortString = Annotated[str, StringConstraints(min_length=1, max_length=256)]
Name = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class ContractModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )
