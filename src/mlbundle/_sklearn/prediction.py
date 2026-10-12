"""Tabular prediction with explicit output dimensions and probability validation."""

import warnings
from typing import Any, cast

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from mlbundle._prediction_input import tabular
from mlbundle.adapters import ProbabilityOutput, RegressionOutput
from mlbundle.claims import ModelClaims
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError, MLBundleError


class Predictor:
    def __init__(self, model: Any, claims: ModelClaims) -> None:
        self.model = model
        self.claims = claims
        self.strings = isinstance(self.model, Pipeline) and any(
            isinstance(value, (OneHotEncoder, OrdinalEncoder))
            and hasattr(value, "categories_")
            for value in self.model.get_params(deep=True).values()
        )

    def _data(self, data: object) -> tuple[object, int]:
        return tabular(data, self.claims.n_features, strings=self.strings)

    def _call(self, data: object, *, probabilities: bool) -> Any:
        # Positional arrays/lists remain supported for named models; sklearn
        # still enforces DataFrame name/order when those names are supplied.
        try:
            return self._native_call(data, probabilities=probabilities)
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Sklearn prediction failed.") from None

    def _native_call(self, data: object, *, probabilities: bool) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but .*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore", message="X has feature names, but .*", category=UserWarning
            )
            raw = (
                self.model.predict_proba(data)
                if probabilities
                else self.model.predict(data)
            )
            if probabilities:
                return (
                    [np.asarray(item, dtype=float) for item in raw]
                    if isinstance(raw, list)
                    else np.asarray(raw, dtype=float)
                )
            return np.asarray(raw, dtype=float)

    def predict(self, data: object) -> RegressionOutput:
        if self.claims.kind != ModelKind.REGRESSOR:
            raise ValueError("predict requires a regressor.")
        data, rows = self._data(data)
        result = np.asarray(self._call(data, probabilities=False), dtype=float)
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
        data, rows = self._data(data)
        raw = self._call(data, probabilities=True)
        matrices = raw if isinstance(raw, list) else [raw]
        if not isinstance(matrices, list) or len(matrices) != self.claims.n_outputs:
            raise InvalidModelError("Invalid probability output count.")
        result: list[list[list[float]]] = []
        for matrix, classes in zip(matrices, self.claims.n_classes or [], strict=True):
            values = np.asarray(matrix, dtype=float)
            if (
                values.shape != (rows, classes)
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
