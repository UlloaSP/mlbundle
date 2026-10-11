"""Original ONNX bytes with a validated CPU session; optional imports stay here."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import onnx
from pydantic import JsonValue

from mlbundle.claims import ModelClaims
from mlbundle.contract import Library, MemberName
from mlbundle.declarations import ExportDeclarations, resolve_labels
from mlbundle.errors import InvalidModelError, LimitExceededError, MLBundleError
from mlbundle.limits import ArchiveLimits

from .description import Description, describe
from .prediction import Predictor
from .session import create_session
from .validation import parse_model


@dataclass(frozen=True)
class OnnxModel:
    """Validated native inference session and the exact bytes that created it."""

    data: bytes
    session: Any


def _description(data: bytes) -> Description:
    try:
        return describe(parse_model(data))
    except MLBundleError:
        raise
    except Exception:
        raise InvalidModelError("Invalid ONNX model claims.") from None


class Adapter:
    library = Library.ONNX
    model_member = MemberName.ONNX

    def recognizes(self, model: object) -> bool:
        return isinstance(model, (bytes, onnx.ModelProto, OnnxModel)) or (
            isinstance(model, (str, PathLike)) and Path(model).suffix.lower() == ".onnx"
        )

    def prepare_export(self, model: object, declarations: ExportDeclarations) -> object:
        try:
            if isinstance(model, OnnxModel):
                data = model.data
            elif isinstance(model, bytes):
                data = model
            elif isinstance(model, onnx.ModelProto):
                if model.ByteSize() > ArchiveLimits().max_member_bytes:
                    raise LimitExceededError("ONNX model exceeds the byte limit.")
                data = model.SerializeToString()
            else:
                path = Path(model)  # type: ignore[arg-type]
                cap = ArchiveLimits().max_member_bytes
                if path.stat().st_size > cap:
                    raise LimitExceededError("ONNX model exceeds the byte limit.")
                with path.open("rb") as stream:
                    data = stream.read(cap + 1)
            description = _description(data)
            resolve_labels(description.claims, declarations)
            return self.deserialize(data)
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Cannot read ONNX model input.") from None

    def serialize(self, model: Any) -> bytes:
        return model.data

    def deserialize(self, data: bytes) -> OnnxModel:
        description = _description(data)
        try:
            return OnnxModel(data, create_session(data, description))
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("ONNX CPU session creation failed.") from None

    def extract_claims(self, model: Any) -> ModelClaims:
        return _description(model.data).claims

    def extract_structure(self, model: object) -> dict[str, JsonValue] | None:  # noqa: ARG002
        return None

    def predictor(self, model: Any) -> Predictor:
        return Predictor(model.session, _description(model.data))
