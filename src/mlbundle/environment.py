"""Environment declarations without importing optional model libraries."""

from __future__ import annotations

import platform
import sysconfig
from typing import Annotated

from pydantic import Field, model_validator

from ._base import ContractModel, ShortString


class Opset(ContractModel):
    domain: Annotated[str, Field(max_length=256)]
    version: int = Field(gt=0, le=2**31 - 1)


class OnnxInfo(ContractModel):
    opsets: list[Opset] = Field(min_length=1, max_length=128)
    ir_version: int = Field(gt=0, le=2**31 - 1)
    producer_name: Annotated[str, Field(max_length=65536)]
    producer_version: Annotated[str, Field(max_length=65536)]

    @model_validator(mode="after")
    def unique_domains(self) -> OnnxInfo:
        domains = [opset.domain for opset in self.opsets]
        if domains != sorted(set(domains)):
            raise ValueError("Opsets must have unique, sorted domains.")
        return self


class TrainingEnvironment(ContractModel):
    """The training machine; package versions live in the bundle's pylock.toml."""

    python_version: ShortString
    python_implementation: ShortString
    free_threaded: bool
    os: ShortString
    os_version: ShortString
    architecture: ShortString
    onnx: OnnxInfo | None = None


def capture_environment(*, onnx: OnnxInfo | None = None) -> TrainingEnvironment:
    return TrainingEnvironment(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        os=platform.system(),
        os_version=platform.release(),
        architecture=platform.machine(),
        onnx=onnx,
    )
