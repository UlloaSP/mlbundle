"""Tabular prediction with explicit output dimensions and probability validation."""

import warnings
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

    def _data(self, data: object) -> object:
        shape = getattr(data, "shape", None)
        if shape is None:
            data = np.asarray(data)
            shape = data.shape
        if len(shape) != 2 or shape[1] != self.claims.n_features:
            raise ValueError("Prediction input must have the model's feature width.")
        return data

    def _call(self, data: object, *, probabilities: bool) -> Any:
        # Positional arrays/lists remain supported for named models; sklearn
        # still enforces DataFrame name/order when those names are supplied.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but .*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore", message="X has feature names, but .*", category=UserWarning
            )
            return (
                self.model.predict_proba(data)
                if probabilities
                else self.model.predict(data)
            )

    def predict(self, data: object) -> RegressionOutput:
        if self.claims.kind != ModelKind.REGRESSOR:
            raise ValueError("predict requires a regressor.")
        data = self._data(data)
        result = np.asarray(self._call(data, probabilities=False), dtype=float)
        rows = len(data)  # type: ignore[arg-type]
        expected = (
            (rows,) if self.claims.n_outputs == 1 else (rows, self.claims.n_outputs)
        )
        if self.claims.n_outputs == 1 and result.shape == (rows, 1):
            result = result[:, 0]
        if result.shape != expected or not np.isfinite(result).all():
            raise InvalidModelError(
                "Invalid regression prediction dimensions or values."
            )
        return cast(RegressionOutput, result.tolist())

    def predict_proba(self, data: object) -> ProbabilityOutput:
        if self.claims.kind != ModelKind.CLASSIFIER:
            raise ValueError("predict_proba requires a classifier.")
        data = self._data(data)
        raw = self._call(data, probabilities=True)
        matrices = raw if isinstance(raw, list) else [raw]
        if not isinstance(matrices, list) or len(matrices) != self.claims.n_outputs:
            raise InvalidModelError("Invalid probability output count.")
        result: list[list[list[float]]] = []
        for matrix, classes in zip(matrices, self.claims.n_classes or [], strict=True):
            values = np.asarray(matrix, dtype=float)
            if (
                values.shape != (len(data), classes)  # type: ignore[arg-type]
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values > 1)
                or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8)
            ):
                raise InvalidModelError(
                    "Invalid probability dimensions, values or row sums."
                )
            result.append(values.tolist())
        return result[0] if self.claims.n_outputs == 1 else result
