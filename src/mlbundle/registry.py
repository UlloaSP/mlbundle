"""Only mlbundle-owned module paths can select an adapter."""

from importlib import import_module
from os import PathLike
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


def find_adapter(model: object) -> Adapter:
    """Ask every available closed adapter; preserve dependency errors if none matches."""
    missing: dict[Library, MissingOptionalDependencyError] = {}
    for library in Library:
        try:
            adapter = get_adapter(library)
        except AdapterNotImplementedError:
            continue
        except MissingOptionalDependencyError as error:
            missing[library] = error
            continue
        if adapter.recognizes(model):
            return adapter
    # Class ancestry only identifies which missing extra to report; it never
    # selects an import path, adapter or estimator constructor.
    roots = {base.__module__.split(".")[0] for base in type(model).__mro__}
    for library, error in missing.items():
        if library.value in roots or (
            library == Library.ONNX and isinstance(model, (bytes, str, PathLike))
        ):
            raise error
    raise UnsupportedModelError
