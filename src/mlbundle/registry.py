"""Only mlbundle-owned module paths can select an adapter."""

from importlib import import_module
from types import MappingProxyType
from typing import cast

from .adapters import Adapter
from .contract import MODEL_MEMBERS, Library
from .errors import (
    AdapterNotImplementedError,
    MissingOptionalDependencyError,
    UnsupportedModelError,
)

_ADAPTERS = MappingProxyType(
    {
        Library.SKLEARN: (
            "mlbundle._sklearn",
            "sklearn",
            {"sklearn", "skops", "numpy", "scipy"},
        ),
        Library.XGBOOST: (
            "mlbundle._xgboost",
            "xgboost",
            {"xgboost", "sklearn", "numpy", "scipy"},
        ),
        Library.ONNX: ("mlbundle._onnx", "onnx", {"onnx", "onnxruntime", "numpy"}),
    }
)


def get_adapter(library: Library) -> Adapter:
    if not isinstance(library, Library):
        raise UnsupportedModelError
    path, extra, dependencies = _ADAPTERS[library]
    try:
        module = import_module(path)
        adapter = cast(Adapter, module.Adapter())
    except ModuleNotFoundError as error:
        if error.name == path:
            raise AdapterNotImplementedError(library.value) from None
        if error.name and error.name.split(".")[0] in dependencies:
            raise MissingOptionalDependencyError(extra) from None
        raise
    if adapter.library != library or adapter.model_member != MODEL_MEMBERS[library][1]:
        raise UnsupportedModelError
    return adapter
