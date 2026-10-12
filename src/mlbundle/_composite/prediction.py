"""Transform original inputs, then delegate to the cached native predictor."""

from typing import Any

from sklearn import config_context
from sklearn.pipeline import Pipeline

from mlbundle._prediction_input import tabular
from mlbundle._sklearn.prediction import _string_inputs
from mlbundle.adapters import (
    Predictor as NativePredictor,
    ProbabilityOutput,
    RegressionOutput,
)
from mlbundle.claims import ModelClaims
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError, MLBundleError, PredictionInputError


class Predictor:
    def __init__(
        self,
        prefix: Pipeline | None,
        native: NativePredictor,
        claims: ModelClaims,
        *,
        categorical: bool = False,
    ) -> None:
        self.prefix, self.native, self.claims = prefix, native, claims
        self.strings = prefix is not None and _string_inputs(prefix)
        self.categorical = categorical

    def _transform(self, data: Any) -> Any:
        # A single-step composite preserves the leaf's categorical admission.
        if self.prefix is None:
            return data
        data, rows = tabular(
            data,
            self.claims.n_features,
            strings=self.strings,
            categorical=self.categorical,
        )
        names = getattr(data, "columns", None)
        if (
            names is not None
            and self.claims.feature_names is not None
            and list(names) != self.claims.feature_names
        ):
            raise PredictionInputError(
                "Prediction columns must match model feature names and order."
            )
        try:
            with config_context(
                transform_output="default", enable_metadata_routing=False
            ):
                result = self.prefix.transform(data)
            if len(result.shape) != 2 or result.shape[0] != rows:
                raise InvalidModelError(
                    "Composite transformation must preserve prediction rows."
                )
            return result
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError(
                "Composite Pipeline transformation failed."
            ) from None

    def predict(self, data: object) -> RegressionOutput:
        if self.claims.kind != ModelKind.REGRESSOR:
            raise ValueError("predict requires a regressor.")
        return self.native.predict(self._transform(data))

    def predict_proba(self, data: object) -> ProbabilityOutput:
        if self.claims.kind != ModelKind.CLASSIFIER:
            raise ValueError("predict_proba requires a classifier.")
        return self.native.predict_proba(self._transform(data))
