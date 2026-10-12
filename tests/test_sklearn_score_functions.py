"""Native univariate score references are closed, projected and committed."""

import numpy as np
import pytest
import skops.io as sio
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    chi2,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import mlbundle
from mlbundle._sklearn import Adapter
from mlbundle._sklearn.trusted import SCORE_FUNCTIONS
from sklearn_helpers import alter_schema, rewrite


def user_score(rows, y):
    return chi2(rows, y)


def fitted(selector):
    y = np.array([0, 1] * 30)
    rows = np.random.default_rng(7).uniform(0.01, 1, (60, 12))
    rows[:, 0] = y + rows[:, 0] * 0.01
    return make_pipeline(selector, LogisticRegression()).fit(rows, y), rows


@pytest.mark.parametrize(
    "selector",
    [
        SelectKBest,
        SelectPercentile,
        SelectFpr,
        SelectFdr,
        SelectFwe,
        GenericUnivariateSelect,
    ],
)
def test_default_univariate_selector_roundtrip(selector, tmp_path):
    model, rows = fitted(selector())
    path = tmp_path / "selector.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))


@pytest.mark.parametrize("function", SCORE_FUNCTIONS)
def test_all_closed_score_functions(function, tmp_path):
    model, rows = fitted(SelectKBest(function, k=2))
    path = tmp_path / "score.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.native_model[0].score_func is function
    assert loaded.manifest.model.hyperparameters["selectkbest__score_func"] == {
        "$function": f"{function.__module__}.{function.__name__}"
    }
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))


@pytest.mark.parametrize("function", [user_score, lambda x, y: chi2(x, y)[0]])
def test_user_scores_and_lambdas_rejected(function, tmp_path):
    model, _ = fitted(SelectKBest(function, k=2))
    with pytest.raises(mlbundle.UnsupportedModelError) as error:
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert any(function.__name__ in name for name in error.value.untrusted_types)
    with pytest.raises(mlbundle.UnsupportedModelError) as error:
        Adapter().deserialize(sio.dumps(model))
    assert any(function.__name__ in name for name in error.value.untrusted_types)


def test_score_reference_and_statistics_committed(tmp_path):
    for attribute in ("score_func", "scores_", "pvalues_"):
        model, _ = fitted(SelectKBest(k=2))
        path = tmp_path / "original.mlbundle"
        mlbundle.export(model, path, overwrite=True)
        if attribute == "score_func":
            model[0].score_func = chi2
        else:
            getattr(model[0], attribute)[0] += 1
        with pytest.raises(mlbundle.ClaimMismatchError):
            mlbundle.load(rewrite(path, sio.dumps(model)))


@pytest.mark.parametrize(
    "loader", ["ObjectNode", "TypeNode", "ConstructorFromReduceNode", "NdArrayNode"]
)
def test_score_name_cannot_select_calling_loader(loader, monkeypatch):
    model, _ = fitted(SelectKBest(k=2))

    def change(schema, entries):
        from mlbundle._sklearn.references import References

        node = next(
            node
            for node in References(schema).nodes
            if node["__loader__"] == "FunctionNode"
        )
        node["__loader__"] = loader

    payload = alter_schema(sio.dumps(model), change)
    monkeypatch.setattr(sio, "loads", lambda *a, **kw: pytest.fail("skops called"))
    with pytest.raises(mlbundle.InvalidModelError, match="function loader"):
        Adapter().deserialize(payload)
