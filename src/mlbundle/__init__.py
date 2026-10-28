"""Pickle-free model bundles."""

from importlib.metadata import version as _version

from ._environment_check import check_environment
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
    EnvironmentMismatchError,
    IntegrityMismatchError,
    InvalidContainerError,
    InvalidEnvironmentLockError,
    InvalidJSONError,
    InvalidManifestError,
    InvalidModelError,
    InvalidSamplesError,
    InvalidSchemaError,
    LimitExceededError,
    LockGenerationError,
    MissingOptionalDependencyError,
    MLBundleError,
    PredictionInputError,
    UnsupportedFormatVersionError,
    UnsupportedModelError,
)
from .inspection import inspect
from .integrity import IntegrityRecord
from .limits import ArchiveLimits
from .lock import EnvironmentLock, LockedPackage, LockedWheel
from .manifest import FeatureInfo, Manifest, MemberReferences, ModelInfo, TargetInfo
from .model_io import export, load
from .records import OutputPrediction
from .results import LoadedBundle, MetadataView

__version__ = _version("mlbundle")

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "AdapterNotImplementedError",
    "ArchiveLimits",
    "ClaimMismatchError",
    "EnvironmentLock",
    "EnvironmentMismatchError",
    "FeatureInfo",
    "FeatureOrigin",
    "IntegrityMismatchError",
    "IntegrityRecord",
    "InvalidContainerError",
    "InvalidEnvironmentLockError",
    "InvalidJSONError",
    "InvalidManifestError",
    "InvalidModelError",
    "InvalidSamplesError",
    "InvalidSchemaError",
    "Library",
    "LimitExceededError",
    "LoadedBundle",
    "LockGenerationError",
    "LockedPackage",
    "LockedWheel",
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
    "OutputPrediction",
    "PredictionInputError",
    "SerializationFormat",
    "TargetInfo",
    "TrainingEnvironment",
    "UnsupportedFormatVersionError",
    "UnsupportedModelError",
    "__version__",
    "check_environment",
    "export",
    "inspect",
    "load",
]
