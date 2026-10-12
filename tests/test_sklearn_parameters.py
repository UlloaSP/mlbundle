"""Faithful native parameter projections and bounded recursive work."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle import UnsupportedModelError


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


def test_parameter_work_is_bounded(monkeypatch):
    from importlib import import_module

    project = import_module("mlbundle._sklearn.parameters")

    monkeypatch.setattr(project, "MAX_VISITS", 10, raising=False)
    shared = [1]
    value = [shared] * 20
    with pytest.raises(mlbundle.LimitExceededError, match="parameter work"):
        project._parameter(value, frozenset())
