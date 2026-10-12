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


class SkopsIO:
    """Reuse checked snapshots inside one operation; never across public calls."""

    def __init__(
        self,
        validate: Callable[[object], None],
        payload: Callable[[bytes], str] = validate_payload,
    ) -> None:
        self.validate = validate
        self.payload = payload
        self._states: list[tuple[object, str]] = []
        self._payload: bytes | None = None
        self._digest: str | None = None

    def state(self, model: object) -> str:
        for native, digest in self._states:
            if native is model:
                return digest
        return self.payload(sio.dumps(model))

    def serialize(self, model: object) -> bytes:
        try:
            self._states = []
            self._payload, self._digest = None, None
            data = sio.dumps(model)
            try:
                digest = self.payload(data)
            except UnsupportedModelError:
                audit_types(data)
                raise
            audit_types(data)
            self.validate(model)
            self._states = [(model, digest)]
            self._payload, self._digest = data, digest
            return data
        except MLBundleError:
            raise
        except Exception:
            raise UnsupportedModelError(
                reason="Model cannot be serialized safely with skops."
            ) from None

    def deserialize(self, data: bytes) -> object:
        try:
            expected = self._digest if data is self._payload else self.payload(data)
            self._payload, self._digest = None, None
            audit_types(data)
            model = sio.loads(data, trusted=list(TRUSTED_TYPES))
            self.validate(model)
            actual = self.payload(sio.dumps(model))
            if expected != actual:
                raise InvalidModelError(
                    "Skops state does not survive native construction."
                )
            self._states = [*self._states[-1:], (model, actual)]
            return model
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid skops model payload.") from None
