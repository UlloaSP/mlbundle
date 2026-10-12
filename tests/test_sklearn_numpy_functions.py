"""FunctionTransformer uses only closed elementwise NumPy function references."""

import numpy as np
import pytest
import skops.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import mlbundle
from mlbundle._sklearn import Adapter
from mlbundle._sklearn.parameters import _parameter
from mlbundle._sklearn.trusted import NUMPY_FUNCTIONS
from sklearn_helpers import alter_schema, rewrite


def user_transform(rows):
    return np.log1p(rows)


def fitted(function=np.log1p, inverse=None):
    rows = np.random.default_rng(7).uniform(0.1, 1, (40, 3))
    y = (rows[:, 0] > 0.5).astype(int)
    model = make_pipeline(
        FunctionTransformer(function, inverse_func=inverse, check_inverse=False),
        LogisticRegression(),
    ).fit(rows, y)
    return model, rows


@pytest.mark.parametrize("function", [*NUMPY_FUNCTIONS, np.abs])
@pytest.mark.parametrize("inverse", [False, True])
def test_closed_numpy_functions_roundtrip(function, inverse, tmp_path):
    model, rows = fitted(
        np.log1p if inverse else function, function if inverse else None
    )
    path = tmp_path / "function.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    key = "inverse_func" if inverse else "func"
    assert getattr(loaded.native_model[0], key) is function
    assert loaded.manifest.model.hyperparameters[f"functiontransformer__{key}"] == {
        "$function": f"numpy.{function.__name__}"
    }
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))


@pytest.mark.parametrize(
    "function", [user_transform, lambda x: x, np.sin, np.mean, np.load]
)
@pytest.mark.parametrize("inverse", [False, True])
def test_other_functions_rejected_at_export_and_load(function, inverse, tmp_path):
    model, _ = fitted()
    setattr(model[0], "inverse_func" if inverse else "func", function)
    with pytest.raises(mlbundle.UnsupportedModelError) as error:
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert any(function.__name__ in name for name in error.value.untrusted_types)
    with pytest.raises(mlbundle.UnsupportedModelError) as error:
        Adapter().deserialize(sio.dumps(model))
    assert any(function.__name__ in name for name in error.value.untrusted_types)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("attribute", ["func", "inverse_func"])
def test_altered_function_reference_fails_claims(attribute, tmp_path):
    model, _ = fitted(inverse=np.expm1)
    path = tmp_path / "original.mlbundle"
    mlbundle.export(model, path)
    setattr(model[0], attribute, np.square)
    with pytest.raises(mlbundle.ClaimMismatchError):
        mlbundle.load(rewrite(path, sio.dumps(model)))


@pytest.mark.parametrize(
    "loader", ["ObjectNode", "TypeNode", "ConstructorFromReduceNode", "NdArrayNode"]
)
def test_numpy_function_cannot_choose_calling_loader(loader, monkeypatch):
    def change(schema, entries):
        schema["__loader__"] = loader

    payload = alter_schema(sio.dumps(np.log1p), change)
    monkeypatch.setattr(sio, "loads", lambda *a, **kw: pytest.fail("skops called"))
    with pytest.raises(mlbundle.InvalidModelError, match="function loader"):
        Adapter().deserialize(payload)


def test_spoofed_function_identity_rejected():
    from sklearn.feature_selection import f_classif

    user_transform.__module__ = f_classif.__module__
    user_transform.__name__ = f_classif.__name__
    try:
        with pytest.raises(mlbundle.UnsupportedModelError):
            _parameter(user_transform, frozenset())
    finally:
        user_transform.__module__ = __name__
        user_transform.__name__ = "user_transform"
