"""Numeric CatBoost input and output validation without native exception contents."""

from typing import Any, cast

import numpy as np

from mlbundle._prediction_input import tabular
from mlbundle.adapters import ProbabilityOutput, RegressionOutput
from mlbundle.claims import ModelClaims
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError, MLBundleError, PredictionInputError


class Predictor:
    def __init__(self, model: Any, claims: ModelClaims) -> None:
        self.model, self.claims = model, claims

    def _call(self, data: object, *, probabilities: bool) -> tuple[np.ndarray, int]:
        try:
            inputs, rows = tabular(data, self.claims.n_features)
            if rows < 1:
                raise PredictionInputError(
                    "CatBoost prediction requires at least one row."
                )
            if (
                hasattr(inputs, "columns")
                and list(inputs.columns) != self.claims.feature_names
            ):
                raise PredictionInputError(
                    "CatBoost prediction columns must match native feature order."
                )
            raw = (
                self.model.predict_proba(inputs, thread_count=1)
                if probabilities
                else self.model.predict(inputs, thread_count=1)
            )
            return np.asarray(raw, dtype=float), rows
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("CatBoost prediction failed.") from None

    def predict(self, data: object) -> RegressionOutput:
        if self.claims.kind != ModelKind.REGRESSOR:
            raise ValueError("predict requires a regressor.")
        values, rows = self._call(data, probabilities=False)
        if values.shape != (rows,) or not np.isfinite(values).all():
            raise InvalidModelError(
                "Invalid regression prediction dimensions or values."
            )
        return cast(RegressionOutput, values.tolist())

    def predict_proba(self, data: object) -> ProbabilityOutput:
        if self.claims.kind != ModelKind.CLASSIFIER:
            raise ValueError("predict_proba requires a classifier.")
        values, rows = self._call(data, probabilities=True)
        if (
            values.shape != (rows, (self.claims.n_classes or [0])[0])
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7)
        ):
            raise InvalidModelError(
                "Invalid probability dimensions, values or row sums."
            )
        return cast(ProbabilityOutput, values.tolist())
