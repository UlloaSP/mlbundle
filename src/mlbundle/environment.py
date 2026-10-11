"""Environment declarations without importing optional model libraries."""

from __future__ import annotations

import platform
from importlib import metadata
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ._base import ContractModel, ShortString

PackageName = Literal[
    "mlbundle",
    "mlschema",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "onnx",
    "onnxruntime",
    "skops",
]
PACKAGE_NAMES: tuple[PackageName, ...] = (
    "mlbundle",
    "mlschema",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "onnx",
    "onnxruntime",
    "skops",
)


class Opset(ContractModel):
    domain: Annotated[str, Field(max_length=256)]
    version: int = Field(gt=0, le=2**31 - 1)


class OnnxInfo(ContractModel):
    opsets: list[Opset] = Field(min_length=1, max_length=128)
    ir_version: int = Field(gt=0, le=2**31 - 1)
    producer_name: Annotated[str, Field(max_length=256)]
    producer_version: Annotated[str, Field(max_length=256)]

    @model_validator(mode="after")
    def unique_domains(self) -> OnnxInfo:
        domains = [opset.domain for opset in self.opsets]
        if domains != sorted(set(domains)):
            raise ValueError("Opsets must have unique, sorted domains.")
        return self


class TrainingEnvironment(ContractModel):
    python_version: ShortString
    python_implementation: ShortString
    os: ShortString
    os_version: ShortString
    architecture: ShortString
    packages: dict[PackageName, ShortString] = Field(max_length=len(PACKAGE_NAMES))
    onnx: OnnxInfo | None = None


def capture_environment(*, onnx: OnnxInfo | None = None) -> TrainingEnvironment:
    packages: dict[PackageName, str] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return TrainingEnvironment(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        os=platform.system(),
        os_version=platform.release(),
        architecture=platform.machine(),
        packages=packages,
        onnx=onnx,
    )
