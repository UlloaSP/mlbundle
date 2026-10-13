"""Readable composition facts supplement complete fitted-state commitments."""

from typing import Any

import numpy as np
import skops.io as sio
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor

from .parameters import parameters
from .payload import validate_payload


def prediction_state(model: Any) -> dict[str, Any]:
    state = {}
    if hasattr(model, "estimators_"):
        estimators = np.asarray(model.estimators_, dtype=object)
        state["estimator_labels"] = [
            getattr(estimator, "classes_", None) for estimator in estimators.flat
        ]
    for attribute in (
        "estimator_weights_",
        "estimators_features_",
        "_x_mean",
        "_x_std",
    ):
        if hasattr(model, attribute):
            state[attribute] = getattr(model, attribute)
    loss = getattr(model, "_loss", getattr(model, "_base_loss", None))
    if loss is not None:
        state["loss"] = {
            "class": type(loss).__name__,
            "link": type(loss.link).__name__,
            "n_classes": loss.n_classes,
        }
    if hasattr(model, "init_"):
        initializer = model.init_
        if isinstance(initializer, BaseEstimator):
            state["initializer"] = {
                "class": type(initializer).__name__,
                "parameters": parameters(initializer),
                "state_sha256": validate_payload(sio.dumps(initializer)),
            }
            if isinstance(initializer, (DummyClassifier, DummyRegressor)):
                state["initializer"]["values"] = {
                    name: getattr(initializer, name)
                    for name in (
                        "classes_",
                        "class_prior_",
                        "constant_",
                        "n_classes_",
                        "n_outputs_",
                    )
                    if hasattr(initializer, name)
                }
        else:
            state["initializer"] = initializer
        state["learning_rate"] = model.learning_rate
    if hasattr(model, "_probA"):
        state["probability_calibration"] = {"A": model._probA, "B": model._probB}
    return state
