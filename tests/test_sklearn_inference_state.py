"""Prediction-relevant fitted state must match claims and native structure."""

import numpy as np
import pytest
import skops.io as sio
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle import ClaimMismatchError, InvalidModelError
from mlbundle._sklearn import Adapter
from sklearn_helpers import rewrite

pytestmark = pytest.mark.filterwarnings(
    "ignore:The `probability` parameter:FutureWarning"
)


def test_missing_route_is_verified(tmp_path):
    model = DecisionTreeClassifier(max_depth=1).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(model, path)
    before = model.predict_proba([[np.nan]])
    state = model.tree_.__getstate__()
    state["nodes"]["missing_go_to_left"][0] ^= 1
    model.tree_.__setstate__(state)
    assert not np.array_equal(before, model.predict_proba([[np.nan]]))
    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, sio.dumps(model)))


@pytest.mark.parametrize("classifier", [False, True])
def test_gradient_boosting_initializer_is_verified(classifier, tmp_path):
    model = (
        GradientBoostingClassifier(n_estimators=2)
        if classifier
        else GradientBoostingRegressor(n_estimators=2)
    ).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    path = tmp_path / "gb.mlbundle"
    mlbundle.export(model, path)
    if classifier:
        model.init_.class_prior_ = np.array([0.9, 0.1])
    else:
        model.init_.constant_ += 10
    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, sio.dumps(model)))


@pytest.mark.parametrize("attribute", ["n_trees_per_iteration_", "n_estimators_"])
def test_gradient_boosting_stage_dimensions_are_consistent(attribute):
    model = GradientBoostingClassifier(n_estimators=2).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    setattr(model, attribute, 3)
    with pytest.raises(InvalidModelError, match="stage arrangement"):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize(
    "model", [GradientBoostingClassifier(n_estimators=2), PoissonRegressor()]
)
def test_private_loss_cannot_disagree_with_declared_parameters(model):
    from sklearn._loss.link import HalfLogitLink

    model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    loss = model._loss if hasattr(model, "_loss") else model._base_loss
    loss.link = HalfLogitLink()
    with pytest.raises(InvalidModelError, match="loss"):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize(
    "model,attribute",
    [
        (AdaBoostClassifier(n_estimators=3), "estimator_weights_"),
        (
            BaggingClassifier(n_estimators=3, max_features=1, random_state=7),
            "estimators_features_",
        ),
    ],
)
def test_ensemble_weights_and_feature_assignment_are_verified(
    model, attribute, tmp_path
):
    x = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]]
    y = [0, 0, 1, 1, 1, 0]
    if attribute == "estimator_weights_":
        x, y = [[index] for index in range(8)], [0, 0, 1, 1, 0, 0, 1, 1]
    model.fit(x, y)
    path = tmp_path / "ensemble.mlbundle"
    mlbundle.export(model, path)
    if attribute == "estimator_weights_":
        before = model.predict_proba(x)
        model.estimator_weights_[0] += 1
        assert not np.array_equal(before, model.predict_proba(x))
    else:
        model.estimators_features_[0][0] ^= 1
    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, sio.dumps(model)))


def test_centered_linear_prediction_state_is_verified(tmp_path):
    model = PLSRegression(n_components=1).fit(
        [[0, 0], [1, 2], [2, 1], [3, 3]], [0, 1, 2, 3]
    )
    path = tmp_path / "pls.mlbundle"
    mlbundle.export(model, path)
    model._x_mean += 1
    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, sio.dumps(model)))


def test_pipeline_transform_state_is_claimed(tmp_path):
    model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression())]
    ).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    path = tmp_path / "pipeline.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.structure is None
    model.steps[0][1].mean_ += 1
    with pytest.raises(ClaimMismatchError, match="state_sha256"):
        mlbundle.load(rewrite(path, sio.dumps(model)))


@pytest.mark.parametrize("attribute", ["_dual_coef_", "_intercept_"])
def test_svm_displayed_coefficients_must_match_inference(attribute):
    model = SVC(kernel="linear", probability=True).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    setattr(model, attribute, getattr(model, attribute) + 1)
    with pytest.raises(InvalidModelError, match="coefficients"):
        Adapter().deserialize(sio.dumps(model))


def test_svm_calibration_state_is_verified(tmp_path):
    model = SVC(kernel="linear", probability=True, random_state=0).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    path = tmp_path / "svm.mlbundle"
    mlbundle.export(model, path)
    model._probA += 1
    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, sio.dumps(model)))
