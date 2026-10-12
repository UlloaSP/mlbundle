"""Bounded names, selector dimensionality and categorical native boundaries."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

import mlbundle
from mlbundle._sklearn.transform_names import output_names
from mlbundle._sklearn.transform_width import output_width
from pipeline_helpers import data, estimator, pipeline


@pytest.mark.parametrize("degree", [1, 2, (2, 3)])
@pytest.mark.parametrize("interaction", [False, True])
def test_polynomial_names_match_native_without_powers(degree, interaction, monkeypatch):
    frame, _ = data()
    model = PolynomialFeatures(degree, interaction_only=interaction).fit(frame)
    expected = model.get_feature_names_out().tolist()
    monkeypatch.setattr(
        PolynomialFeatures,
        "powers_",
        property(lambda _: pytest.fail("dense powers_ allocation")),
    )
    assert output_names(model, list("abc")) == expected


def test_large_polynomial_names_and_name_work_budget(monkeypatch):
    model = PolynomialFeatures(1, include_bias=False).fit(np.zeros((1, 100_000)))
    monkeypatch.setattr(
        PolynomialFeatures,
        "powers_",
        property(lambda _: pytest.fail("80 GB powers_ allocation")),
    )
    names = [f"x{i}" for i in range(100_000)]
    assert output_width(model, 100_000) == 100_000
    assert output_names(model, names) == names
    model = PolynomialFeatures(500).fit([[1]])
    with pytest.raises(mlbundle.LimitExceededError, match="feature-name work"):
        output_names(model, ["x"])
    model = PolynomialFeatures(100_000, interaction_only=True, include_bias=False).fit(
        [[1]]
    )
    assert output_names(model, ["x"]) == ["x"]


@pytest.mark.parametrize("formatter", ["{feature_name:>10000000000}", "{feature_name}"])
def test_formatter_rejected_before_expansion(formatter, tmp_path, monkeypatch):
    frame, y = data()
    transform = ColumnTransformer(
        [("sc", StandardScaler(), [0])], verbose_feature_names_out=formatter
    )
    model = Pipeline([("ct", transform), ("est", estimator())]).fit(frame, y)
    monkeypatch.setattr(
        ColumnTransformer,
        "get_feature_names_out",
        lambda *_: pytest.fail("unbounded formatter"),
    )
    with pytest.raises(
        mlbundle.InvalidModelError, match="Invalid transformed feature names"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")


def test_numpy_string_selectors_and_scalar_cache_rejection(tmp_path):
    frame, y = data()
    transform = ColumnTransformer([("sc", StandardScaler(), np.array(["a", "b"]))]).fit(
        frame
    )
    model = Pipeline(
        [("ct", transform), ("est", estimator().fit(transform.transform(frame), y))]
    )
    path = tmp_path / "numpy-selector.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(
        mlbundle.load(path).predict_proba(frame), model.predict_proba(frame)
    )
    transform = ColumnTransformer([("sc", StandardScaler(), [0])]).fit(frame)
    model.steps[0] = ("ct", transform)
    model.steps[1] = ("est", estimator().fit(transform.transform(frame), y))
    transform.transformers[0] = ("sc", transform.transformers[0][1], 0)
    transform.transformers_[0] = ("sc", transform.transformers_[0][1], 0)
    transform._columns = [0]
    with pytest.raises(
        mlbundle.InvalidModelError, match="ColumnTransformer input mapping"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")


def test_missing_ordinal_cache_is_rejected(tmp_path):
    frame, y = data()
    model = Pipeline([("enc", OrdinalEncoder()), ("est", estimator())]).fit(frame, y)
    del model.steps[0][1]._missing_indices
    with pytest.raises(
        mlbundle.InvalidModelError, match="Invalid composite Pipeline state"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")


def test_fitted_selector_rank_and_manual_name_boundaries(tmp_path):
    frame, y = data()
    transform = ColumnTransformer([("sc", StandardScaler(), [0])]).fit(frame)
    model = Pipeline(
        [("ct", transform), ("est", estimator().fit(transform.transform(frame), y))]
    )
    transform.transformers_[0] = ("sc", transform.transformers_[0][1], 0)
    with pytest.raises(
        mlbundle.InvalidModelError, match="ColumnTransformer input mapping"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")
    for transform in [
        PolynomialFeatures(2),
        ColumnTransformer([("sc", StandardScaler(), [0])]),
    ]:
        model = Pipeline(
            [
                ("sc", StandardScaler().set_output(transform="pandas")),
                ("tr", transform),
                ("est", estimator()),
            ]
        ).fit(frame, y)
        transform.feature_names_in_ = np.array(list("xyz"), dtype=object)
        with pytest.raises(
            mlbundle.InvalidModelError, match="Invalid transformed feature names"
        ):
            mlbundle.export(model, tmp_path / "never.mlbundle")


def test_encoder_drop_cache_native_storage_is_required(tmp_path):
    frame, y = data()
    model = Pipeline([("enc", OneHotEncoder(drop="first")), ("est", estimator())]).fit(
        frame, y
    )
    model.steps[0][1]._drop_idx_after_grouping = model.steps[0][
        1
    ]._drop_idx_after_grouping.tolist()
    with pytest.raises(mlbundle.InvalidModelError, match="fitted dimensions disagree"):
        mlbundle.export(model, tmp_path / "never.mlbundle")


@pytest.mark.parametrize("identity", [None, "passthrough", FunctionTransformer()])
def test_categorical_xgboost_identity_prefix(identity, tmp_path):
    frame, y = data()
    frame["b"] = pd.Categorical(np.where(y, "high", "low"))
    model = Pipeline([("identity", identity), ("est", estimator())]).fit(frame, y)
    path = tmp_path / "categorical.mlbundle"
    mlbundle.export(model, path)
    bundle = mlbundle.load(path)
    assert np.array_equal(bundle.predict_proba(frame), model.predict_proba(frame))
    assert np.array_equal(
        bundle.native_model.predict_proba(frame), model.predict_proba(frame)
    )


@pytest.mark.parametrize("dependency", ["sklearn", "skops"])
def test_missing_composite_dependency_is_bounded(dependency, tmp_path):
    model, _ = pipeline()
    path = tmp_path / "dependency.mlbundle"
    mlbundle.export(model, path)
    script = """
import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split(".")[0] == sys.argv[2]:
        raise ModuleNotFoundError(name=sys.argv[2])
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import mlbundle
view = mlbundle.inspect(sys.argv[1])
assert view.manifest.model.pipeline is not None
try:
    mlbundle.load(sys.argv[1])
except mlbundle.MissingOptionalDependencyError as error:
    assert error.extra == "sklearn"
else:
    raise AssertionError("missing dependency was loaded")
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path), dependency],
        capture_output=True,
        text=True,
        check=True,
    )
