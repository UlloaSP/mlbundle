"""Numeric tabular feeds and validated tensor/ZipMap outputs, entirely in memory."""

import math
import warnings
from typing import Any, cast

import numpy as np

from mlbundle.adapters import ProbabilityOutput, RegressionOutput
from mlbundle.contract import ModelKind
from mlbundle.errors import InvalidModelError, PredictionInputError

from .description import Description

_InputShapeError = PredictionInputError


def _array(data: Any, dtype: str) -> np.ndarray:
    raw = np.asarray(data, dtype=object)
    if np.dtype(dtype).kind in "iu":
        limits = np.iinfo(dtype)
        for value in raw.flat:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value)
                or int(value) != value
                or not limits.min <= int(value) <= limits.max
            ):
                raise InvalidModelError(
                    "ONNX integer inputs require whole numbers within the dtype range."
                )
    elif any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in raw.flat
    ):
        raise InvalidModelError("ONNX inputs require finite numeric values.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values = np.asarray(raw, dtype=dtype)
    if not np.isfinite(values).all():
        raise InvalidModelError("ONNX inputs require finite numeric values.")
    return values


class Predictor:
    def __init__(self, session: Any, description: Description) -> None:
        self.session = session
        self.description = description
        self.claims = description.claims

    def _feeds(self, data: object) -> tuple[dict[str, np.ndarray], int]:
        frame = None
        if any(
            base.__module__.split(".")[0] == "pandas" for base in type(data).__mro__
        ):
            import pandas as pd  # noqa: PLC0415

            if isinstance(data, pd.DataFrame):
                frame = data
        values = np.asarray(data, dtype=object)
        if values.ndim != 2 or values.shape[1] != self.claims.n_features:
            raise _InputShapeError(
                "Prediction input must have the model's feature width."
            )
        rows = values.shape[0]
        feeds = {}
        for index, value in enumerate(self.description.inputs):
            shape = value["shape"]
            if isinstance(shape[0], int) and rows != shape[0]:
                raise _InputShapeError(
                    "Prediction row count must match the ONNX static batch."
                )
            if len(self.description.inputs) == 1:
                column = values
            elif frame is not None:
                if not frame.columns.is_unique or value["name"] not in frame.columns:
                    raise InvalidModelError(
                        "ONNX DataFrame columns must match model input names."
                    )
                column = frame[[value["name"]]].to_numpy(dtype=object)
            else:
                column = values[:, index : index + 1]
            feeds[value["name"]] = _array(column, value["dtype"])
        return feeds, rows

    def _run(self, data: object) -> tuple[Any, int]:
        try:
            feeds, rows = self._feeds(data)
            return self.session.run([self.description.output], feeds)[0], rows
        except (InvalidModelError, _InputShapeError):
            raise
        except Exception:
            raise InvalidModelError(
                "ONNX prediction failed or input values are invalid."
            ) from None

    def predict(self, data: object) -> RegressionOutput:
        if self.claims.kind != ModelKind.REGRESSOR:
            raise ValueError("predict requires a regressor.")
        raw, rows = self._run(data)
        try:
            values = np.asarray(raw, dtype=float)
            if values.shape == (rows, 1):
                values = values[:, 0]
            if values.shape != (rows,) or not np.isfinite(values).all():
                raise ValueError
            return cast(RegressionOutput, values.tolist())
        except Exception:
            raise InvalidModelError("Invalid ONNX regression output.") from None

    def predict_proba(self, data: object) -> ProbabilityOutput:
        if self.claims.kind != ModelKind.CLASSIFIER:
            raise ValueError("predict_proba requires a classifier.")
        raw, rows = self._run(data)
        try:
            if self.description.maps:
                labels = (self.claims.class_labels or [[]])[0]
                if not isinstance(raw, list) or any(
                    not isinstance(row, dict) or set(row) != set(labels) for row in raw
                ):
                    raise ValueError
                raw = [[row[label] for label in labels] for row in raw]
            values = np.asarray(raw, dtype=float)
            if (
                values.shape != (rows, (self.claims.n_classes or [0])[0])
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values > 1)
                or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-5)
            ):
                raise ValueError
            return cast(ProbabilityOutput, values.tolist())
        except Exception:
            raise InvalidModelError("Invalid ONNX probability output.") from None
