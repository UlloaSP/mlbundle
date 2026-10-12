"""One owned skops byte boundary for estimators and transformer components."""

from collections.abc import Callable

import skops.io as sio

from mlbundle.errors import InvalidModelError, MLBundleError, UnsupportedModelError

from .payload import validate_payload
from .trusted import TRUSTED_TYPES


def audit_types(data: bytes) -> None:
    unsupported = tuple(
        sorted(set(sio.get_untrusted_types(data=data)) - set(TRUSTED_TYPES))
    )
    if unsupported:
        raise UnsupportedModelError(unsupported)


def serialize(model: object, validate: Callable[[object], None]) -> bytes:
    try:
        data = sio.dumps(model)
        try:
            validate_payload(data)
        except UnsupportedModelError:
            audit_types(data)
            raise
        audit_types(data)
        validate(model)
        return data
    except MLBundleError:
        raise
    except Exception:
        raise UnsupportedModelError(
            reason="Model cannot be serialized safely with skops."
        ) from None


def deserialize(data: bytes, validate: Callable[[object], None]) -> object:
    try:
        expected_state = validate_payload(data)
        audit_types(data)
        model = sio.loads(data, trusted=list(TRUSTED_TYPES))
        validate(model)
        if expected_state != validate_payload(sio.dumps(model)):
            raise InvalidModelError("Skops state does not survive native construction.")
        return model
    except MLBundleError:
        raise
    except Exception:
        raise InvalidModelError("Invalid skops model payload.") from None
