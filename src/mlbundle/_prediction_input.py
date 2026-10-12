"""Numeric input checks shared by optional sklearn and XGBoost predictors."""

import math
from typing import Any

import numpy as np
from scipy.sparse import issparse

from .errors import InvalidModelError, MLBundleError, PredictionInputError


def tabular(
    data: Any, width: int, *, categorical: bool = False, strings: bool = False
) -> tuple[Any, int]:
    try:
        shape = getattr(data, "shape", None)
        if shape is None:
            data = np.asarray(data)
            shape = data.shape
        if len(shape) != 2 or shape[1] != width:
            raise PredictionInputError(
                "Prediction input must have the model's feature width."
            )
        if issparse(data):
            arrays = [data.data]
        elif categorical and hasattr(data, "dtypes"):
            arrays = [
                np.asarray(data.iloc[:, index])
                for index, dtype in enumerate(data.dtypes)
                if str(dtype) != "category"
            ]
        else:
            arrays = [np.asarray(data)]
        for values in arrays:
            _numeric(values, strings=strings)
        return data, int(shape[0])
    except MLBundleError:
        raise
    except Exception:
        raise InvalidModelError("Invalid tabular prediction input.") from None


def _numeric(values: np.ndarray, *, strings: bool) -> None:
    if values.dtype.kind == "O":
        if any(
            not (strings and isinstance(value, str))
            and (
                not isinstance(
                    value, (int, float, np.integer, np.floating, bool, np.bool_)
                )
                or math.isinf(value)
            )
            for value in values.flat
        ):
            raise InvalidModelError(
                "Prediction input must contain numeric values without infinity."
            )
    elif strings and values.dtype.kind == "U":
        return
    elif values.dtype.kind not in "biuf" or np.isinf(values).any():
        raise InvalidModelError(
            "Prediction input must contain numeric values without infinity."
        )
