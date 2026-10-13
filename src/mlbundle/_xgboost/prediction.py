"""XGBoost tabular inference with bounded-error and probability validation."""

from typing import Any, cast

import numpy as np

from mlbundle._prediction_decisions import class_indices
from mlbundle._prediction_input import tabular
from mlbundle.adapters import ProbabilityOutput, RegressionOutput
from mlbundle.claims import ModelClaims
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError, MLBundleError


class Predictor:
    def __init__(self, model: Any, claims: ModelClaims) -> None:
        self.model = model
        self.claims = claims

    def _call(
        self, data: object, *, probabilities: bool, decisions: bool = False
    ) -> tuple[np.ndarray, int]:
        try:
            data, rows = tabular(
                data, self.claims.n_features, categorical=self.model.enable_categorical
            )
            raw = (
                self.model.predict_proba(data)
                if probabilities
                else self.model.predict(data)
            )
            return np.asarray(raw, dtype=object if decisions else float), rows
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("XGBoost prediction failed.") from None

    def predict_class_indices(self, data: object) -> list[list[int]]:
        if self.claims.kind != ModelKind.CLASSIFIER:
            raise ValueError("Class prediction requires a classifier.")
        raw, rows = self._call(data, probabilities=False, decisions=True)
        # The closed wrapper predicts encoded indices, including with bound labels.
        labels = [list(range((self.claims.n_classes or [0])[0]))]
        return class_indices(raw, labels, rows)

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
