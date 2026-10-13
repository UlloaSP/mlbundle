"""Pickle-free model bundles."""

from importlib.metadata import version as _version

from .claims import ModelClaims
from .contract import (
    FORMAT_NAME,
    FORMAT_VERSION,
    FeatureOrigin,
    Library,
    MemberName,
    ModelKind,
    SerializationFormat,
)
from .environment import OnnxInfo, Opset, TrainingEnvironment
from .errors import (
    AdapterNotImplementedError,
    ClaimMismatchError,
    IntegrityMismatchError,
    InvalidContainerError,
    InvalidJSONError,
    InvalidManifestError,
    InvalidModelError,
    InvalidSamplesError,
    InvalidSchemaError,
    LimitExceededError,
    MissingOptionalDependencyError,
    MLBundleError,
    PredictionInputError,
    UnsupportedFormatVersionError,
    UnsupportedModelError,
)
from .inspection import inspect
from .integrity import IntegrityRecord
from .limits import ArchiveLimits
from .manifest import FeatureInfo, Manifest, MemberReferences, ModelInfo, TargetInfo
from .model_io import export, load
from .results import LoadedBundle, MetadataView

__version__ = _version("mlbundle")

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "AdapterNotImplementedError",
    "ArchiveLimits",
    "ClaimMismatchError",
    "FeatureInfo",
    "FeatureOrigin",
    "IntegrityMismatchError",
    "IntegrityRecord",
    "InvalidContainerError",
    "InvalidJSONError",
    "InvalidManifestError",
    "InvalidModelError",
    "InvalidSamplesError",
    "InvalidSchemaError",
    "Library",
    "LimitExceededError",
    "LoadedBundle",
    "MLBundleError",
    "Manifest",
    "MemberName",
    "MemberReferences",
    "MetadataView",
    "MissingOptionalDependencyError",
    "ModelClaims",
    "ModelInfo",
    "ModelKind",
    "OnnxInfo",
    "Opset",
    "PredictionInputError",
    "SerializationFormat",
    "TargetInfo",
    "TrainingEnvironment",
    "UnsupportedFormatVersionError",
    "UnsupportedModelError",
    "__version__",
    "export",
    "inspect",
    "load",
]
