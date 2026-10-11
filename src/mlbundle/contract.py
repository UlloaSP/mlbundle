"""Closed format identifiers; these never select external Python classes."""

from enum import StrEnum
from types import MappingProxyType

FORMAT_NAME = "mlbundle"
FORMAT_VERSION = "1"


class Library(StrEnum):
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    ONNX = "onnx"


class ModelKind(StrEnum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


class SerializationFormat(StrEnum):
    SKOPS = "skops"
    UBJSON = "ubjson"
    ONNX = "onnx"


class FeatureOrigin(StrEnum):
    MODEL = "model"
    DATAFRAME = "dataframe"
    GENERATED = "generated"


class MemberName(StrEnum):
    MANIFEST = "manifest.json"
    SKOPS = "model.skops"
    UBJSON = "model.ubj"
    ONNX = "model.onnx"
    SCHEMA = "schema.json"
    STRUCTURE = "structure.json"
    SAMPLES = "samples.json"


MODEL_MEMBERS = MappingProxyType(
    {
        Library.SKLEARN: (SerializationFormat.SKOPS, MemberName.SKOPS),
        Library.XGBOOST: (SerializationFormat.UBJSON, MemberName.UBJSON),
        Library.ONNX: (SerializationFormat.ONNX, MemberName.ONNX),
    }
)
