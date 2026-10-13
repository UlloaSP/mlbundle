"""Fitted dimensions that inference uses must agree with displayed counts."""

from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm._base import BaseLibSVM

from mlbundle.errors import InvalidModelError


def validate_fitted(model: Any, width: int | None) -> None:
    _statistics(model, width)
    _dummy(model)
    if width is None:
        return
    if hasattr(model, "_fit_X") and hasattr(model, "_y"):
        rows = model._fit_X.shape[0]
        if (
            model._fit_X.shape[1] != width
            or len(model._y) != rows
            or getattr(model, "n_samples_fit_", rows) != rows
        ):
            raise InvalidModelError("Neighbor fitted dimensions disagree.")
    if isinstance(model, (MLPClassifier, MLPRegressor)) and hasattr(model, "coefs_"):
        _mlp(model, width)


def _mlp(model: Any, width: int) -> None:
    sizes = [width, *np.atleast_1d(model.hidden_layer_sizes).tolist()]
    sizes.append(model.n_outputs_)
    if (
        model.n_layers_ != len(sizes)
        or len(model.coefs_) != len(sizes) - 1
        or len(model.intercepts_) != len(model.coefs_)
        or any(
            coef.shape != (left, right) or intercept.shape != (right,)
            for coef, intercept, left, right in zip(
                model.coefs_, model.intercepts_, sizes, sizes[1:], strict=False
            )
        )
    ):
        raise InvalidModelError("MLP fitted layer dimensions disagree.")
    if isinstance(model, MLPClassifier):
        classes = len(model.classes_)
        if (
            classes < 2
            or model.n_outputs_ != (1 if classes <= 2 else classes)
            or model.out_activation_ != ("logistic" if classes <= 2 else "softmax")
        ):
            raise InvalidModelError("MLP class count and output activation disagree.")
    elif model.out_activation_ != ("exp" if model.loss == "poisson" else "identity"):
        raise InvalidModelError("MLP regression loss and output activation disagree.")


def _statistics(model: Any, width: int | None) -> None:
    if not hasattr(model, "classes_") or isinstance(model.classes_, list):
        return
    classes = len(model.classes_)
    if width is not None and hasattr(model, "coef_"):
        _coefficient(model, classes, width)
    for name in ("theta_", "var_", "means_", "centroids_"):
        if (
            width is not None
            and hasattr(model, name)
            and np.asarray(getattr(model, name)).shape
            != (
                classes,
                width,
            )
        ):
            raise InvalidModelError(
                "Classifier fitted statistics and dimensions disagree."
            )
    for name in ("class_prior_", "class_log_prior_", "class_count_"):
        if hasattr(model, name) and np.asarray(getattr(model, name)).shape != (
            classes,
        ):
            raise InvalidModelError(
                "Classifier fitted priors and class counts disagree."
            )
    if hasattr(model, "feature_log_prob_"):
        values = model.feature_log_prob_
        shapes = (
            [np.asarray(value).shape for value in values]
            if isinstance(values, list)
            else [np.asarray(values).shape]
        )
        if any(len(shape) != 2 or shape[0] != classes for shape in shapes):
            raise InvalidModelError("Classifier likelihoods and class counts disagree.")
        if (isinstance(values, list) and len(values) != width) or (
            not isinstance(values, list) and shapes[0][1] != width
        ):
            raise InvalidModelError(
                "Classifier likelihoods and feature counts disagree."
            )


def _coefficient(model: Any, classes: int, width: int) -> None:
    if isinstance(model, BaseLibSVM):
        return
    if np.asarray(model.coef_).shape != (1 if classes <= 2 else classes, width):
        raise InvalidModelError("Classifier coefficients and class counts disagree.")


def _dummy(model: Any) -> None:
    if (
        isinstance(model, DummyRegressor)
        and hasattr(model, "constant_")
        and np.asarray(model.constant_).shape != (1, model.n_outputs_)
    ):
        raise InvalidModelError("Dummy regressor constant and output counts disagree.")
