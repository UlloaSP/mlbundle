"""Bounded errors at bundle trust boundaries."""


def _identifier(value: str) -> str:
    if (
        len(value) <= 160
        and value.isascii()
        and all(character.isalnum() or character in "._-" for character in value)
    ):
        return value
    return "unrecognized"


class MLBundleError(Exception):
    """Base for failures reported by mlbundle."""


class InvalidContainerError(MLBundleError):
    """The archive violates the container contract."""


class LimitExceededError(MLBundleError):
    """An input exceeds a resource limit."""


class UnsupportedFormatVersionError(MLBundleError):
    def __init__(self) -> None:
        super().__init__("Unsupported mlbundle format version.")


class InvalidManifestError(MLBundleError):
    def __init__(self) -> None:
        super().__init__("Invalid mlbundle manifest.")


class InvalidJSONError(MLBundleError):
    def __init__(self) -> None:
        super().__init__("Invalid or unsupported JSON value.")


class IntegrityMismatchError(MLBundleError):
    def __init__(self, member: str) -> None:
        self.member = _identifier(member)
        super().__init__(f"Integrity mismatch for member {self.member}.")


class ClaimMismatchError(MLBundleError):
    def __init__(self, claim: str) -> None:
        self.claim = _identifier(claim)
        super().__init__(f"Model claim mismatch: {self.claim}.")


class UnsupportedModelError(MLBundleError):
    def __init__(
        self, untrusted_types: tuple[str, ...] = (), *, reason: str | None = None
    ) -> None:
        self.untrusted_types = untrusted_types
        names = ", ".join(_identifier(name) for name in untrusted_types[:20])
        super().__init__(
            reason
            or f"Unsupported model ({len(untrusted_types)} untrusted types): {names}."
        )


class InvalidModelError(MLBundleError):
    """Native model content is malformed or violates the adapter contract."""


class PredictionInputError(InvalidModelError, ValueError):
    """Bounded invalid-input error retaining the shape-error ValueError contract."""


class InvalidSchemaError(MLBundleError):
    """Training inputs or schema declarations violate the schema contract."""


class InvalidSamplesError(MLBundleError):
    """Sample selection or row declarations violate the samples contract."""


class MissingOptionalDependencyError(MLBundleError):
    def __init__(self, extra: str) -> None:
        self.extra = _identifier(extra)
        super().__init__(
            f'Install this feature with pip install "mlbundle[{self.extra}]".'
        )


class AdapterNotImplementedError(MLBundleError):
    def __init__(self, library: str) -> None:
        self.library = _identifier(library)
        super().__init__(f"The {self.library} adapter is not implemented yet.")
