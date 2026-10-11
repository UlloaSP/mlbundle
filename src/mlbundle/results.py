"""Metadata-only and fully validated native-model result contracts."""

from dataclasses import dataclass, field

from pydantic import JsonValue

from .adapters import Predictor, ProbabilityOutput, RegressionOutput
from .claims import ClassLabels
from .contract import FeatureOrigin, Library, MemberName, ModelKind
from .manifest import Manifest


@dataclass(frozen=True)
class MetadataView:
    """Integrity-checked metadata; model claims remain unverified by inspection."""

    manifest: Manifest
    members: tuple[MemberName, ...]


@dataclass(frozen=True)
class LoadedBundle:
    """A reader constructs this only after validating model claims and structure."""

    manifest: Manifest
    members: tuple[MemberName, ...]
    native_model: object = field(repr=False)
    predictor: Predictor = field(repr=False)
    schema: dict[str, JsonValue] | None = None
    structure: dict[str, JsonValue] | None = None
    samples: list[dict[str, JsonValue]] | None = None

    @property
    def kind(self) -> ModelKind:
        return self.manifest.model.kind

    @property
    def library(self) -> Library:
        return self.manifest.model.library

    @property
    def class_name(self) -> str:
        return self.manifest.model.estimator_class

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.manifest.features.names)

    @property
    def feature_origin(self) -> FeatureOrigin:
        return self.manifest.features.origin

    @property
    def class_labels(self) -> ClassLabels | None:
        labels = self.manifest.target.class_labels
        return None if labels is None else [list(output) for output in labels]

    def predict_proba(self, data: object) -> ProbabilityOutput:
        if self.kind != ModelKind.CLASSIFIER:
            raise ValueError("predict_proba requires a classifier.")
        return self.predictor.predict_proba(data)

    def predict(self, data: object) -> RegressionOutput:
        if self.kind != ModelKind.REGRESSOR:
            raise ValueError("predict requires a regressor.")
        return self.predictor.predict(data)
