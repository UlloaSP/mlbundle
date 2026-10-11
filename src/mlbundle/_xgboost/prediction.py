"""XGBoost tabular inference with bounded-error and probability validation."""

from typing import Any, cast

import numpy as np

from mlbundle.adapters import ProbabilityOutput, RegressionOutput
from mlbundle.claims import ModelClaims
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError


class Predictor:
    def __init__(self, model: Any, claims: ModelClaims) -> None:
        self.model = model
        self.claims = claims

    def _call(self, data: object, *, probabilities: bool) -> tuple[np.ndarray, int]:
        try:
            shape = getattr(data, "shape", None)
            if shape is None:
                data = np.asarray(data)
                shape = data.shape
            if len(shape) != 2 or shape[1] != self.claims.n_features:
                raise ValueError(
                    "Prediction input must have the model's feature width."
                )
            rows = int(shape[0])
            raw = (
                self.model.predict_proba(data)
                if probabilities
                else self.model.predict(data)
            )
            return np.asarray(raw, dtype=float), rows
        except ValueError as error:
            if str(error) == "Prediction input must have the model's feature width.":
                raise
            raise InvalidModelError("Invalid XGBoost prediction input.") from None
        except Exception:
            raise InvalidModelError("XGBoost prediction failed.") from None

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
