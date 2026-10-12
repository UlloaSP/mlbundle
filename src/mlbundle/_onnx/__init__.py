"""Original ONNX bytes with a validated CPU session; optional imports stay here."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import onnx
from pydantic import JsonValue, ValidationError

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
    except ValidationError as error:
        failure = error.errors(include_input=False, include_url=False)[0]
        # Disclose closed field names, never author-supplied keys or values.
        fields = {"producer_name", "producer_version", "feature_names", "class_labels"}
        field = next((part for part in failure["loc"] if part in fields), None)
        if field is not None and failure["type"] == "string_too_long":
            limit = failure.get("ctx", {})["max_length"]
            raise InvalidModelError(
                f"ONNX {field} exceeds the {limit}-character limit."
            ) from None
        raise InvalidModelError("Invalid ONNX model claims.") from None
    except Exception:
        raise InvalidModelError("Invalid ONNX model claims.") from None


class Adapter:
    library = Library.ONNX
    model_member = MemberName.ONNX

    def __init__(self) -> None:
        self._data: bytes | None = None
        self._validated: Description | None = None

    def _describe(self, data: bytes) -> Description:
        # An adapter belongs to one export/load, never a global cache or bundle.
        if data is not self._data or self._validated is None:
            description = _description(data)
            self._data, self._validated = data, description
        return self._validated

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
            description = self._describe(data)
            resolve_labels(description.claims, declarations)
            return self.deserialize(data)
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Cannot read ONNX model input.") from None

    def serialize(self, model: Any) -> bytes:
        return model.data

    def deserialize(self, data: bytes) -> OnnxModel:
        description = self._describe(data)
        try:
            return OnnxModel(data, create_session(data, description))
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("ONNX CPU session creation failed.") from None

    def extract_claims(self, model: Any) -> ModelClaims:
        return self._describe(model.data).claims

    def extract_structure(self, model: object) -> dict[str, JsonValue] | None:  # noqa: ARG002
        return None

    def predictor(self, model: Any) -> Predictor:
        return Predictor(model.session, self._describe(model.data))
