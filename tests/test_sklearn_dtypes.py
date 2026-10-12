"""Plain fitted dtypes round trip; compound and hostile dtype nodes stay closed."""

from io import BytesIO

import numpy as np
import pytest
import skops.io as sio
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

import mlbundle
from mlbundle._sklearn import Adapter
from mlbundle._sklearn.parameters import _parameter
from mlbundle._sklearn.trusted import SCALAR_TYPES
from sklearn_helpers import alter_schema, rewrite


def fitted():
    rows = np.array([[1, 2], [2, np.nan], [3, 4], [4, 5]], dtype=float)
    model = make_pipeline(SimpleImputer(), LogisticRegression()).fit(rows, [0, 0, 1, 1])
    return model, rows


@pytest.mark.parametrize("scalar", [*SCALAR_TYPES, ">f8", ">i4"])
def test_plain_dtype_state_roundtrip(scalar, tmp_path):
    model, rows = fitted()
    model[0]._fit_dtype = np.dtype(scalar)
    path = tmp_path / "dtype.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.native_model[0]._fit_dtype == np.dtype(scalar)
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))
    assert _parameter(np.dtype(scalar), frozenset()) == {"$dtype": np.dtype(scalar).str}


def test_object_imputation_and_kbins_dtype(tmp_path):
    rows = np.array([["red"], ["blue"], ["red"], ["blue"]], dtype=object)
    model = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(), LogisticRegression()
    ).fit(rows, [0, 1, 0, 1])
    path = tmp_path / "object.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.native_model[0]._fit_dtype == np.dtype(object)
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))
    rows = np.arange(40, dtype=float).reshape(20, 2)
    model = make_pipeline(KBinsDiscretizer(dtype=np.float64), LogisticRegression()).fit(
        rows, [0] * 10 + [1] * 10
    )
    mlbundle.export(model, path, overwrite=True)
    assert np.array_equal(
        mlbundle.load(path).predict_proba(rows), model.predict_proba(rows)
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype([("x", "f8")]),
        np.dtype(("f8", (2,))),
        np.dtype("U4"),
        np.dtype("S4"),
        np.dtype("datetime64[ns]"),
        np.dtype("V8"),
        np.dtype("f8", metadata={"hidden": 1}),
    ],
)
def test_unsupported_dtype_export_and_projection(dtype, tmp_path):
    model, _ = fitted()
    model[0]._fit_dtype = dtype
    with pytest.raises(mlbundle.MLBundleError):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    with pytest.raises(mlbundle.UnsupportedModelError, match="dtype"):
        _parameter(dtype, frozenset())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "dtype", [np.dtype([("x", "f8")]), np.dtype(("f8", (2,))), np.dtype("U4")]
)
def test_compound_dtype_rejected_before_skops_load(dtype, monkeypatch):
    model, _ = fitted()
    model[0]._fit_dtype = dtype
    payload = sio.dumps(model)
    monkeypatch.setattr(sio, "loads", lambda *a, **kw: pytest.fail("skops called"))
    with pytest.raises(mlbundle.MLBundleError):
        Adapter().deserialize(payload)


def test_dtype_value_is_committed(tmp_path):
    model, _ = fitted()
    path = tmp_path / "original.mlbundle"
    mlbundle.export(model, path)
    model[0]._fit_dtype = np.dtype("float32")
    with pytest.raises(mlbundle.ClaimMismatchError, match="state_sha256"):
        mlbundle.load(rewrite(path, sio.dumps(model)))


@pytest.mark.parametrize(
    "attack",
    ["loader", "identity", "storage", "nonempty", "object_shape", "object_content"],
)
def test_hostile_dtype_nodes_fail_before_skops(attack, monkeypatch):
    from mlbundle._sklearn.references import References

    model, _ = fitted()
    model[0]._fit_dtype = (
        np.dtype(object) if attack.startswith("object") else np.dtype("f8")
    )

    def change(schema, entries):
        refs = References(schema)
        dtype = next(node for node in refs.nodes if node["__loader__"] == "DTypeNode")
        storage = refs.resolve(dtype["content"])
        if attack == "loader":
            dtype["__loader__"] = "ObjectNode"
        elif attack == "identity":
            dtype["__class__"] = "float64"
        elif attack == "storage":
            storage["__class__"] = "float64"
        elif attack == "nonempty":
            buffer = BytesIO()
            np.save(buffer, np.ones(1), allow_pickle=False)
            entries[storage["file"]] = buffer.getvalue()
        elif attack == "object_shape":
            refs.resolve(storage["shape"])["content"] = []
        else:
            storage["content"] = [
                {
                    "__loader__": "JsonNode",
                    "__class__": "str",
                    "__module__": "builtins",
                    "content": "1",
                    "is_json": True,
                }
            ]

    payload = alter_schema(sio.dumps(model), change)
    monkeypatch.setattr(sio, "loads", lambda *a, **kw: pytest.fail("skops called"))
    with pytest.raises(mlbundle.InvalidModelError, match=r"dtype|references"):
        Adapter().deserialize(payload)
