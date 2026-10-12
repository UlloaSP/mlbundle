"""Check alternate inference state against the estimator's displayed parameters."""

from typing import Any

import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.linear_model._glm.glm import _GeneralizedLinearRegressor
from sklearn.svm._base import BaseLibSVM

from mlbundle.errors import InvalidModelError


def validate_consistency(model: Any) -> None:
    if isinstance(
        model, (GradientBoostingClassifier, GradientBoostingRegressor)
    ) and hasattr(model, "estimators_"):
        columns = (
            model.n_classes_
            if isinstance(model, GradientBoostingClassifier) and model.n_classes_ > 2
            else 1
        )
        trees = model.estimators_
        if (
            not isinstance(trees, np.ndarray)
            or trees.ndim != 2
            or trees.shape != (model.n_estimators_, columns)
            or model.n_trees_per_iteration_ != columns
        ):
            raise InvalidModelError("Invalid gradient boosting stage arrangement.")
        expected = model._get_loss(None)
        _loss(model._loss, expected)
        _initializer(model)
    if isinstance(model, (ForestClassifier, ForestRegressor)) and hasattr(
        model, "estimators_"
    ):
        for estimator in model.estimators_:
            if estimator.n_outputs_ != model.n_outputs_ or (
                isinstance(model, ForestClassifier)
                and not np.array_equal(
                    np.atleast_1d(estimator.n_classes_), np.atleast_1d(model.n_classes_)
                )
            ):
                raise InvalidModelError(
                    "Forest and constituent output counts disagree."
                )
    if isinstance(model, _GeneralizedLinearRegressor) and hasattr(model, "_base_loss"):
        _loss(model._base_loss, model._get_loss())
    if isinstance(model, BaseLibSVM) and hasattr(model, "_dual_coef_"):
        _svm(model)


def _loss(actual: Any, expected: Any) -> None:
    if (
        type(actual) is not type(expected)
        or type(actual.link) is not type(expected.link)
        or getattr(actual, "n_classes", None) != getattr(expected, "n_classes", None)
    ):
        raise InvalidModelError("Fitted loss and declared parameters disagree.")


def _initializer(model: Any) -> None:
    initializer = model.init_
    if (
        hasattr(initializer, "n_features_in_")
        and initializer.n_features_in_ != model.n_features_in_
    ):
        raise InvalidModelError("Gradient boosting initializer input width disagrees.")


def _same(left: Any, right: Any, sign: int) -> bool:
    if issparse(left) or issparse(right):
        return (
            issparse(left)
            and issparse(right)
            and left.format == right.format
            and left.shape == right.shape
            and np.array_equal(left.indices, right.indices)
            and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.data, sign * right.data)
        )
    return np.array_equal(left, sign * right)


def _svm(model: Any) -> None:
    if (
        type(model._sparse) is not bool
        or issparse(model.support_vectors_) != model._sparse
        or issparse(model._dual_coef_) != model._sparse
    ):
        raise InvalidModelError("SVM sparse flag and fitted storage disagree.")
    classes = len(model.classes_) if hasattr(model, "classes_") else None
    count = len(model.support_)
    supports = np.asarray(model._n_support)
    rows = 1 if classes is None else classes - 1
    intercepts = 1 if classes is None else classes * (classes - 1) // 2
    if (
        supports.dtype.kind not in "iu"
        or supports.shape != ((2,) if classes is None else (classes,))
        or np.any(supports < 0)
        or np.any(supports > count)
        or int(supports.sum()) != count * (2 if classes is None else 1)
        or model._dual_coef_.shape != (rows, count)
        or model._intercept_.shape != (intercepts,)
        or model.support_.dtype.kind not in "iu"
        or np.any(model.support_ < 0)
        or np.any(model.support_ >= model.shape_fit_[0])
        or model.shape_fit_[1] != model.n_features_in_
    ):
        raise InvalidModelError("Invalid SVM fitted dimensions or support indices.")
    if model.kernel != "precomputed" and model.support_vectors_.shape != (
        count,
        model.n_features_in_,
    ):
        raise InvalidModelError("Invalid SVM support-vector dimensions.")
    if (
        classes is not None
        and (model._probA.size or callable(getattr(model, "predict_proba", None)))
        and (model._probA.shape != (intercepts,) or model._probB.shape != (intercepts,))
    ):
        raise InvalidModelError("Invalid SVM probability calibration dimensions.")
    sign = -1 if model._impl in ("c_svc", "nu_svc") and len(model.classes_) == 2 else 1
    if not _same(model.dual_coef_, model._dual_coef_, sign) or not np.array_equal(
        model.intercept_, sign * model._intercept_
    ):
        raise InvalidModelError("SVM public and inference coefficients disagree.")
