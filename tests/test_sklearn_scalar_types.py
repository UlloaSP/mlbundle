"""Closed NumPy scalar type parameter records and fitted categorical composition."""

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

import mlbundle
from mlbundle._sklearn import Adapter
from mlbundle._sklearn.parameters import _parameter
from mlbundle._sklearn.trusted import SCALAR_TYPES
from sklearn_helpers import alter_schema, rewrite


@pytest.mark.parametrize("scalar", SCALAR_TYPES)
def test_scalar_type_parameter_projection(scalar):
    assert _parameter(scalar, frozenset()) == {"$type": f"numpy.{scalar.__name__}"}


@pytest.mark.parametrize("encoder", [OneHotEncoder, OrdinalEncoder, TargetEncoder])
def test_default_encoder_in_column_transformer(encoder, tmp_path):
    rows = pd.DataFrame({"category": ["red", "blue"] * 10, "number": np.arange(20)})
    model = make_pipeline(
        ColumnTransformer([("cat", encoder(), ["category"])], remainder="passthrough"),
        LogisticRegression(),
    ).fit(rows, [0, 1] * 10)
    path = tmp_path / "encoder.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))
    if encoder is not TargetEncoder:
        assert loaded.manifest.model.hyperparameters[
            "columntransformer__cat__dtype"
        ] == {"$type": "numpy.float64"}


def test_nested_imputer_scaler_encoder_gradient_boosting(tmp_path):
    rows = pd.DataFrame(
        {"cat": ["red", "blue"] * 10, "num": np.arange(20, dtype=float)}
    )
    rows.loc[0, "num"] = np.nan
    model = make_pipeline(
        ColumnTransformer(
            [
                ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), ["num"]),
                ("cat", OneHotEncoder(), ["cat"]),
            ]
        ),
        GradientBoostingClassifier(n_estimators=2),
    ).fit(rows, [0, 1] * 10)
    path = tmp_path / "nested.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(
        mlbundle.load(path).predict_proba(rows), model.predict_proba(rows)
    )
    model[0].transformers_[0][1][0].statistics_[0] += 1
    with pytest.raises(mlbundle.ClaimMismatchError, match="state_sha256"):
        mlbundle.load(rewrite(path, sio.dumps(model)))


@pytest.mark.parametrize(
    "scalar",
    [float, object, np.str_, np.object_, np.void, np.datetime64, np.longdouble],
)
def test_other_type_hyperparameters_remain_rejected(scalar, tmp_path):
    model = make_pipeline(OneHotEncoder(), LogisticRegression()).fit(
        [[0], [1], [0], [1]], [0, 1, 0, 1]
    )
    model[0].dtype = scalar
    with pytest.raises(mlbundle.UnsupportedModelError):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    with pytest.raises(mlbundle.UnsupportedModelError):
        _parameter(scalar, frozenset())


def test_changed_type_parameter_rejected_on_load(tmp_path):
    model = make_pipeline(OneHotEncoder(), LogisticRegression()).fit(
        [[0], [1], [0], [1]], [0, 1, 0, 1]
    )
    path = tmp_path / "original.mlbundle"
    mlbundle.export(model, path)
    model[0].dtype = np.float32
    with pytest.raises(mlbundle.ClaimMismatchError):
        mlbundle.load(rewrite(path, sio.dumps(model)))


def test_dtype_instance_parameter_roundtrip(tmp_path):
    rows = [[0], [1], [0], [1]]
    model = make_pipeline(
        OneHotEncoder(dtype=np.dtype("float64")), LogisticRegression()
    ).fit(rows, [0, 1, 0, 1])
    path = tmp_path / "dtype-instance.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.manifest.model.hyperparameters["onehotencoder__dtype"] == {
        "$dtype": "<f8"
    }
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))


def test_type_node_cannot_resolve_other_numpy_name(monkeypatch):
    def change(schema, entries):
        schema["__class__"] = "load"

    payload = alter_schema(sio.dumps(np.float64), change)
    monkeypatch.setattr(sio, "loads", lambda *a, **kw: pytest.fail("skops called"))
    with pytest.raises(mlbundle.UnsupportedModelError, match=r"numpy.load"):
        Adapter().deserialize(payload)


def test_unused_unfitted_column_transformer_branch(tmp_path):
    rows = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=float)
    transform = ColumnTransformer(
        [
            ("active", StandardScaler(), [0]),
            ("unused", ColumnTransformer([("inner", StandardScaler(), [0])]), []),
        ]
    )
    model = make_pipeline(transform, LogisticRegression()).fit(rows, [0, 0, 1, 1])
    path = tmp_path / "unused.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(
        mlbundle.load(path).predict_proba(rows), model.predict_proba(rows)
    )
