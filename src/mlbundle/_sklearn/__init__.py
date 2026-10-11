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
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise UnsupportedModelError(
                reason="Classifier must expose original classes_."
            )
        outputs = classes if isinstance(classes, list) else [classes]
        labels = cast(ClassLabels, canonical_value(outputs))
        return len(labels), labels
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "n_outputs_"):
        return int(estimator.n_outputs_), None
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
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
        self._validate(model)
        try:
            classifier = is_classifier(model)
            outputs, labels = _outputs(model, classifier)
            names = getattr(model, "feature_names_in_", None)
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

    def extract_structure(self, model: object) -> dict[str, JsonValue] | None:
        self._validate(model)
        return extract_structure(model)

    def predictor(self, model: object) -> Predictor:
        return Predictor(model, self.extract_claims(model))
