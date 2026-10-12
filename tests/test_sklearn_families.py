"""Other previously admitted prediction families retain legitimate round trips."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    GammaRegressor,
    HuberRegressor,
    Lasso,
    PoissonRegressor,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.multioutput import RegressorChain
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor

import mlbundle

pytestmark = [
    pytest.mark.filterwarnings("ignore:The `probability` parameter:FutureWarning"),
    pytest.mark.filterwarnings("ignore:.*:sklearn.exceptions.ConvergenceWarning"),
]


@pytest.mark.parametrize(
    "model",
    [
        ExtraTreeClassifier(random_state=0),
        ExtraTreeRegressor(random_state=0),
        AdaBoostClassifier(n_estimators=3, random_state=0),
        AdaBoostRegressor(n_estimators=3, random_state=0),
        BaggingClassifier(n_estimators=3, random_state=0),
        BaggingRegressor(n_estimators=3, random_state=0),
        BayesianRidge(),
        ARDRegression(),
        ElasticNet(),
        Lasso(),
        HuberRegressor(),
        PoissonRegressor(),
        GammaRegressor(),
        SGDClassifier(loss="log_loss", random_state=0),
        SGDRegressor(random_state=0),
        LinearSVR(random_state=0),
        SVR(),
        NuSVR(),
        SVC(kernel="linear", probability=True, random_state=0),
        PLSRegression(n_components=1),
        GaussianNB(),
        MultinomialNB(),
        ComplementNB(),
        BernoulliNB(),
        CategoricalNB(),
        KNeighborsClassifier(algorithm="brute", n_neighbors=2),
        KNeighborsRegressor(algorithm="brute", n_neighbors=2),
        NearestCentroid(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(reg_param=0.1),
        MLPClassifier(
            hidden_layer_sizes=(3,), solver="lbfgs", random_state=0, max_iter=300
        ),
        MLPRegressor(
            hidden_layer_sizes=(3,), solver="lbfgs", random_state=0, max_iter=300
        ),
    ],
)
def test_additional_family_roundtrip(model, tmp_path):
    x = np.asarray(
        [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2], [1, 2]], dtype=float
    )
    y = np.asarray([0, 0, 1, 1, 1, 0, 1, 0])
    from sklearn.base import is_classifier

    target = y if is_classifier(model) else y + 0.1
    model.fit(x, target)
    path = tmp_path / "family.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    method = "predict_proba" if is_classifier(model) else "predict"
    expected = np.asarray(getattr(model, method)(x)).squeeze()
    assert np.array_equal(np.asarray(getattr(loaded, method)(x)).squeeze(), expected)
    assert (
        loaded.structure is not None or loaded.manifest.model.state_sha256 is not None
    )


def test_regressor_chain_roundtrip(tmp_path):
    x = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y = np.column_stack([x[:, 0], x[:, 1]])
    model = RegressorChain(Ridge()).fit(x, y)
    path = tmp_path / "chain.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(mlbundle.load(path).predict(x), model.predict(x))


def test_sparse_fitted_neighbor_state_remains_outside_trust_boundary(tmp_path):
    from scipy.sparse import csr_matrix

    x = csr_matrix([[0, 0], [1, 0], [0, 1], [1, 1]])
    model = KNeighborsRegressor(algorithm="brute", n_neighbors=1).fit(x, [0, 1, 2, 3])
    path = tmp_path / "sparse.mlbundle"
    with pytest.raises(mlbundle.UnsupportedModelError, match="csr_matrix"):
        mlbundle.export(model, path)


def test_fitted_categorical_pipeline_roundtrip(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    rows = [["a"], ["b"], ["a"], ["b"]]
    model = Pipeline(
        [
            ("encode", OneHotEncoder(sparse_output=False, dtype="float64")),
            ("model", LogisticRegression()),
        ]
    ).fit(rows, [0, 1, 0, 1])
    path = tmp_path / "categories.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))
    with pytest.raises(mlbundle.MLBundleError):
        loaded.predict_proba([[float("inf")]])
