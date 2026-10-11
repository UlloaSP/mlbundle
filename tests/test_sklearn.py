"""Sklearn native round trips, claimed metadata and hostile model payloads."""

import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

import mlbundle
from mlbundle import InvalidModelError, UnsupportedModelError
from mlbundle._sklearn import TRUSTED_TYPES, Adapter
from mlbundle.canonical import canonical_dumps


class CustomTree(DecisionTreeClassifier):
    pass


def user_function(data):
    return data


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeClassifier(random_state=1),
        DecisionTreeRegressor(random_state=1),
        RandomForestClassifier(n_estimators=3, random_state=1),
        RandomForestRegressor(n_estimators=3, random_state=1),
        ExtraTreesClassifier(n_estimators=3, random_state=1),
        ExtraTreesRegressor(n_estimators=3, random_state=1),
        GradientBoostingClassifier(n_estimators=3, random_state=1),
        GradientBoostingRegressor(n_estimators=3, random_state=1),
        LogisticRegression(max_iter=400),
        LinearRegression(),
        Ridge(),
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=400)),
            ]
        ),
    ],
)
def test_roundtrip(model, data, tmp_path):
    x, y = data
    model.fit(x, y)
    path = tmp_path / "model.mlbundle"
    view = mlbundle.export(model, path, target_name="species", metadata={"run": 1})
    loaded = mlbundle.load(path)
    assert type(loaded.native_model) is type(model)
    assert np.array_equal(model.predict(x), loaded.native_model.predict(x))
    method = "predict_proba" if loaded.kind == "classifier" else "predict"
    assert np.array_equal(
        getattr(model, method)(x), getattr(loaded, method)(x.tolist())
    )
    adapter = Adapter()
    assert view.manifest == loaded.manifest
    assert loaded.manifest.verified_claims() == adapter.extract_claims(model)
    assert canonical_dumps(loaded.structure) == canonical_dumps(
        adapter.extract_structure(model)
    )
    assert loaded.feature_names == tuple(f"feature_{i}" for i in range(1, 5))
    assert (
        loaded.manifest.target.name == "species"
        and loaded.manifest.user_metadata == {"run": 1}
    )
    assert (
        isinstance(loaded.native_model, DecisionTreeClassifier)
        if type(model) is DecisionTreeClassifier
        else True
    )


@pytest.mark.parametrize(
    "labels", [[False, True], [1, 2], ["no", "yes"], ["a", "b", "c"]]
)
def test_label_types_and_names(labels, data, tmp_path):
    x, y = data
    y = np.asarray(labels)[y % len(labels)]
    frame = pd.DataFrame(x, columns=list("abcd"))
    model = LogisticRegression(max_iter=400).fit(frame, y)
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model, path, feature_names=["ignored"])
    loaded = mlbundle.load(path)
    assert loaded.feature_origin == "model" and loaded.feature_names == tuple("abcd")
    assert loaded.class_labels == [model.classes_.tolist()]
    assert type(loaded.class_labels[0][0]) is type(labels[0])
    for value in [frame, x, x.tolist()]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but .*",
                category=UserWarning,
            )
            expected = model.predict_proba(value)
        assert np.array_equal(loaded.predict_proba(value), expected)


@pytest.mark.parametrize("classifier", [True, False])
def test_multioutput(classifier, data, tmp_path):
    x, y = data
    model = DecisionTreeClassifier() if classifier else LinearRegression()
    model.fit(x, np.column_stack([y, y % 2]))
    path = tmp_path / "multi.mlbundle"
    mlbundle.export(model, path, feature_names=list("abcd"))
    loaded = mlbundle.load(path)
    assert loaded.manifest.model.n_outputs == 2 and loaded.feature_origin == "dataframe"
    if classifier:
        assert loaded.class_labels == [[0, 1, 2], [0, 1]]
        assert all(
            np.array_equal(a, b)
            for a, b in zip(
                loaded.predict_proba(x), model.predict_proba(x), strict=True
            )
        )
    else:
        assert np.array_equal(loaded.predict(x), model.predict(x))


@pytest.mark.parametrize(
    "model",
    [
        MultiOutputClassifier(DecisionTreeClassifier()),
        MultiOutputRegressor(Ridge()),
    ],
)
def test_multioutput_wrappers(model, data, tmp_path):
    x, y = data
    model.fit(x, np.column_stack([y, y % 2]))
    path = tmp_path / "multi.mlbundle"
    mlbundle.export(model, path)
    assert mlbundle.load(path).manifest.model.n_outputs == 2


def test_multilabel_chain_is_rejected(data, tmp_path):
    x, y = data
    model = ClassifierChain(LogisticRegression()).fit(
        x, np.column_stack([y % 2, y % 2])
    )
    with pytest.raises(UnsupportedModelError, match="output-major"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")


@pytest.mark.parametrize(
    "model,match",
    [
        (CustomTree(), "CustomTree"),
        (
            Pipeline(
                [
                    ("function", FunctionTransformer(user_function)),
                    ("tree", DecisionTreeClassifier()),
                ]
            ),
            "user_function",
        ),
        (LinearSVC(), "predict_proba"),
        (HistGradientBoostingClassifier(), "TreePredictor"),
        (Pipeline([("xgb", XGBClassifier(n_estimators=2, n_jobs=1))]), "xgboost"),
    ],
)
def test_export_rejections(model, match, data, tmp_path):
    x, y = data
    model.fit(x, y)
    with pytest.raises(UnsupportedModelError, match=match):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert not (tmp_path / "bad.mlbundle").exists()


def test_import_and_inspect_are_lazy(data, tmp_path):
    x, y = data
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(DecisionTreeClassifier().fit(x, y), path)
    code = "import sys,mlbundle; mlbundle.inspect(sys.argv[1]); assert not {'sklearn','skops','numpy','pandas','xgboost','onnx','onnxruntime'} & sys.modules.keys(); print('lazy import and inspect: OK')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lazy import and inspect: OK" in result.stdout


def test_fixed_trust_and_unsafe_tree(data, monkeypatch):
    x, y = data
    model = DecisionTreeClassifier().fit(x, y)
    original, seen = sio.loads, []

    def loads(data, *, trusted):
        seen.append(trusted)
        return original(data, trusted=trusted)

    monkeypatch.setattr(sio, "loads", loads)
    Adapter().deserialize(sio.dumps(model))
    assert seen == [list(TRUSTED_TYPES)]
    model.tree_.children_left[0] = 1_000_000
    with pytest.raises(InvalidModelError, match="indices"):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize("invalid", [np.nan, -0.1, 1.1, 0.4])
def test_prediction_validation(invalid, monkeypatch):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    predictor = Adapter().predictor(model)
    monkeypatch.setattr(
        model, "predict_proba", lambda data: [[invalid, 0.0]] * len(data)
    )
    with pytest.raises(InvalidModelError, match="probability"):
        predictor.predict_proba([[0]])
    with pytest.raises(ValueError, match="width"):
        predictor.predict_proba([[0, 1]])
    with pytest.raises(ValueError, match="regressor"):
        predictor.predict([[0]])
    from sklearn.linear_model import Ridge

    regressor = Ridge().fit([[0], [1]], [0, 1])
    prediction = Adapter().predictor(regressor)
    monkeypatch.setattr(regressor, "predict", lambda data: [np.inf] * len(data))
    with pytest.raises(InvalidModelError, match="regression"):
        prediction.predict([[0]])
    with pytest.raises(ValueError, match="classifier"):
        prediction.predict_proba([[0]])


def test_parameter_projection(tmp_path):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from mlbundle._sklearn.parameters import _parameter

    state = np.random.RandomState(7)
    model = DecisionTreeClassifier(random_state=state).fit([[0], [1]], [0, 1])
    path = tmp_path / "state.mlbundle"
    mlbundle.export(model, path)
    assert (
        "$random_state"
        in mlbundle.load(path).manifest.model.hyperparameters["random_state"]
    )
    assert _parameter((np.float32(1.5), np.array([1, 2]), None), frozenset()) == [
        1.5,
        [1, 2],
        None,
    ]
    for value in [float, lambda x: x, {1: "bad"}]:
        with pytest.raises(UnsupportedModelError):
            _parameter(value, frozenset())
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(UnsupportedModelError, match="Cyclic"):
        _parameter(cyclic, frozenset())
    model = Pipeline(
        [("identity", FunctionTransformer()), ("tree", DecisionTreeClassifier())]
    ).fit([[0], [1]], [0, 1])
    path = tmp_path / "pipeline.mlbundle"
    mlbundle.export(model, path)
    params = mlbundle.load(path).manifest.model.hyperparameters
    assert params["tree"]["$estimator"]["class"] == "DecisionTreeClassifier"
    assert "tree__max_depth" in params
