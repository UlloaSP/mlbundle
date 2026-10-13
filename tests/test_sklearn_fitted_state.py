"""Fitted counts, label mappings, architecture and class identity are consistent."""

import numpy as np
import pytest
import skops.io as sio
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle import InvalidModelError, UnsupportedModelError
from mlbundle._sklearn import Adapter

pytestmark = [
    pytest.mark.filterwarnings("ignore:The `probability` parameter:FutureWarning"),
    pytest.mark.filterwarnings("ignore:.*:sklearn.exceptions.ConvergenceWarning"),
]


@pytest.mark.parametrize(
    "model",
    [
        MLPClassifier(hidden_layer_sizes=[3], solver="lbfgs", random_state=0),
        MLPRegressor(hidden_layer_sizes=[3, 2], solver="lbfgs", random_state=0),
    ],
)
def test_fitted_mlp_architecture_roundtrip(model, tmp_path):
    x = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2]]
    model.fit(x, [0, 0, 1, 1, 2, 2])
    path = tmp_path / "mlp.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    method = "predict_proba" if isinstance(model, MLPClassifier) else "predict"
    assert np.array_equal(getattr(loaded, method)(x), getattr(model, method)(x))


def test_mlp_public_labels_must_match_prediction_labels():
    model = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(2,), random_state=0).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    model.classes_ = model.classes_[::-1].copy()
    with pytest.raises(InvalidModelError, match="binarizer"):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize(
    "attribute,value",
    [("n_layers_", 1000000000), ("out_activation_", "identity"), ("n_outputs_", 2)],
)
def test_mlp_fitted_output_state_is_consistent(attribute, value):
    model = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(2,), random_state=0).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    setattr(model, attribute, value)
    with pytest.raises(InvalidModelError, match="MLP"):
        Adapter().deserialize(sio.dumps(model))


def test_neighbor_sample_count_cannot_drive_work_from_a_declaration():
    model = KNeighborsRegressor(algorithm="brute", n_neighbors=1).fit(
        [[0], [1]], [0, 1]
    )
    model.n_samples_fit_ = 1000000000
    with pytest.raises(InvalidModelError, match="Neighbor fitted dimensions"):
        Adapter().deserialize(sio.dumps(model))


def test_classifier_labels_cannot_overstate_fitted_statistics():
    model = GaussianNB().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    model.classes_ = np.array([0, 1, 2])
    with pytest.raises(InvalidModelError, match="statistics"):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_n_support", np.array([-1, 1], dtype=np.int32)),
        ("support_", np.array([1000000000] * 2)),
        ("_sparse", True),
        ("_probA", np.array([0.0, 0.0])),
    ],
)
def test_svm_fitted_shapes_and_indices_are_checked(attribute, value):
    model = SVC(kernel="linear", probability=True).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    setattr(model, attribute, value)
    with pytest.raises(InvalidModelError, match="SVM"):
        Adapter().deserialize(sio.dumps(model))


def test_nominal_sklearn_class_impersonation_is_rejected(tmp_path):
    class RenamedTree(DecisionTreeClassifier):
        pass

    RenamedTree.__name__ = "DecisionTreeClassifier"
    RenamedTree.__module__ = "sklearn.tree._classes"
    model = RenamedTree().fit([[0], [1]], [0, 1])
    with pytest.raises(UnsupportedModelError, match="exact"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")


@pytest.mark.parametrize("classifier", [False, True])
def test_initializer_fitted_dimensions_are_checked(classifier):
    model = (
        GradientBoostingClassifier(n_estimators=2, init=DummyClassifier())
        if classifier
        else GradientBoostingRegressor(n_estimators=2, init=DummyRegressor())
    ).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    if classifier:
        model.init_.class_prior_ = np.array([0.3, 0.3, 0.4])
    else:
        model.init_.constant_ = np.array([[0.0, 1.0]])
    with pytest.raises(InvalidModelError, match=r"priors|constant"):
        Adapter().deserialize(sio.dumps(model))


def test_initializer_input_width_matches_parent():
    from sklearn.linear_model import LogisticRegression

    model = GradientBoostingClassifier(n_estimators=2, init=LogisticRegression()).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    model.init_.n_features_in_ = 2
    with pytest.raises(InvalidModelError, match="initializer input width"):
        Adapter().deserialize(sio.dumps(model))
