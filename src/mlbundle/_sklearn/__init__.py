"""Skops-only sklearn adapter; importing the base package never imports this module."""

from typing import Any, cast

import numpy as np
import skops.io as sio
from pydantic import JsonValue, ValidationError
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import ClassifierChain, MultiOutputRegressor, RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from mlbundle.canonical import canonical_value
from mlbundle.claims import ClassLabels, FeatureNames, ModelClaims
from mlbundle.contract import Library, MemberName, ModelKind
from mlbundle.errors import (
    InvalidJSONError,
    InvalidModelError,
    MLBundleError,
    UnsupportedModelError,
)
from mlbundle.limits import MAX_CLASSES, MAX_OUTPUTS

from .parameters import parameters
from .payload import validate_payload
from .prediction import Predictor
from .structure import extract_structure, validate_tree_storage

# Native numeric node storage for DecisionTree, RandomForest, ExtraTrees and
# GradientBoosting; no functions/I/O constructors. Indices are validated on load.
TRUSTED_TYPES = ("sklearn.tree._tree.Tree",)


def _untrusted(data: bytes) -> None:
    unsupported = tuple(
        sorted(set(sio.get_untrusted_types(data=data)) - set(TRUSTED_TYPES))
    )
    if unsupported:
        raise UnsupportedModelError(unsupported)


def _outputs(model: Any, classifier: bool) -> tuple[int, ClassLabels | None]:
    if classifier:
        labels = _labels(model)
        return len(labels), labels
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "n_outputs_"):
        outputs = estimator.n_outputs_
        if (
            not isinstance(outputs, (int, np.integer))
            or isinstance(outputs, (bool, np.bool_))
            or not 1 <= outputs <= MAX_OUTPUTS
        ):
            raise InvalidModelError("Invalid regressor output count.")
        return int(outputs), None
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim not in (1, 2) or coef.shape[-1] != estimator.n_features_in_:
            raise InvalidModelError("Invalid regressor coefficient dimensions.")
        return 1 if coef.ndim == 1 else coef.shape[0], None
    if hasattr(estimator, "_y"):
        target = np.asarray(estimator._y)
        return 1 if target.ndim == 1 else target.shape[1], None
    if isinstance(estimator, (MultiOutputRegressor, RegressorChain)):
        return len(list(estimator.estimators_)), None
    if not get_tags(estimator).target_tags.multi_output:
        return 1, None
    raise UnsupportedModelError(
        reason="Regressor output count cannot be recovered faithfully."
    )


def _labels(model: Any) -> ClassLabels:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise UnsupportedModelError(reason="Classifier must expose original classes_.")
    outputs = classes if isinstance(classes, list) else [classes]
    if not 1 <= len(outputs) <= MAX_OUTPUTS or any(
        np.asarray(output).ndim != 1 or not 1 <= len(output) <= MAX_CLASSES
        for output in outputs
    ):
        raise InvalidModelError("Invalid classifier label dimensions.")
    return cast(ClassLabels, canonical_value(outputs))


class Adapter:
    library = Library.SKLEARN
    model_member = MemberName.SKOPS

    def recognizes(self, model: object) -> bool:
        return (
            isinstance(model, BaseEstimator)
            and not type(model).__module__.startswith("xgboost.")
            and (is_classifier(model) or is_regressor(model))
        )

    def _validate(self, model: Any) -> None:
        if not self.recognizes(model):
            raise UnsupportedModelError(
                reason="Sklearn payload must be a classifier or regressor."
            )
        try:
            check_is_fitted(model)
        except NotFittedError:
            raise UnsupportedModelError(
                reason="Sklearn estimator must be fitted before export/load."
            ) from None
        if is_classifier(model) and not callable(getattr(model, "predict_proba", None)):
            raise UnsupportedModelError(reason="Classifier must expose predict_proba.")
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
        binarizer = getattr(estimator, "_label_binarizer", None)
        if (
            isinstance(estimator, ClassifierChain)
            or getattr(estimator, "multilabel_", False)
            or getattr(binarizer, "y_type_", None) == "multilabel-indicator"
        ):
            raise UnsupportedModelError(
                reason="Multilabel/chain probabilities cannot satisfy the output-major contract."
            )
        if not hasattr(model, "n_features_in_"):
            raise UnsupportedModelError(
                reason="Estimator must expose fitted n_features_in_."
            )
        validate_tree_storage(model)

    def serialize(self, model: object) -> bytes:
        try:
            data = sio.dumps(model)
            validate_payload(data)
            _untrusted(data)
            self._validate(model)
            return data
        except MLBundleError:
            raise
        except Exception:
            raise UnsupportedModelError(
                reason="Model cannot be serialized safely with skops."
            ) from None

    def deserialize(self, data: bytes) -> object:
        try:
            validate_payload(data)
            _untrusted(data)
            model = sio.loads(data, trusted=list(TRUSTED_TYPES))
            self._validate(model)
            return model
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid skops model payload.") from None

    def extract_claims(self, model: Any) -> ModelClaims:
        try:
            self._validate(model)
            classifier = is_classifier(model)
            outputs, labels = _outputs(model, classifier)
            names = getattr(model, "feature_names_in_", None)
            if names is not None and (
                np.asarray(names).ndim != 1 or len(names) != model.n_features_in_
            ):
                raise InvalidModelError("Invalid model feature-name dimensions.")
            return ModelClaims(
                estimator_class=type(model).__name__,
                kind=ModelKind.CLASSIFIER if classifier else ModelKind.REGRESSOR,
                hyperparameters=parameters(model),
                n_features=int(model.n_features_in_),
                n_classes=None
                if labels is None
                else [len(output) for output in labels],
                n_outputs=outputs,
                feature_names=None
                if names is None
                else cast(FeatureNames, canonical_value(names)),
                feature_dtypes=None,
                class_labels=labels,
            )
        except (ValidationError, InvalidJSONError):
            raise UnsupportedModelError(
                reason="Model claims cannot be represented faithfully."
            ) from None
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid sklearn model claims.") from None

    def extract_structure(self, model: object) -> dict[str, JsonValue] | None:
        try:
            self._validate(model)
            return extract_structure(model)
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid sklearn model structure.") from None

    def predictor(self, model: object) -> Predictor:
        return Predictor(model, self.extract_claims(model))
