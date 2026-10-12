"""Unknown estimator/nested state and unreferenced files are rejected on both paths."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import skops.io as sio
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import mlbundle
from mlbundle import InvalidModelError, MLBundleError
from mlbundle._sklearn import Adapter
from sklearn_helpers import alter_schema, pack, unpack


@pytest.mark.parametrize("extra", [1, [[], []], csr_matrix([[1.0]]), StandardScaler()])
def test_unknown_estimator_attributes_rejected(extra, tmp_path):
    model = LinearRegression().fit([[0], [1]], [0, 1])
    model.review_extra_ = extra
    with pytest.raises(InvalidModelError, match="Unknown sklearn estimator state"):
        Adapter().deserialize(sio.dumps(model))
    with pytest.raises(InvalidModelError, match="Unknown sklearn estimator state"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    assert not list(tmp_path.iterdir())


def test_unknown_nested_object_rejected():
    model = LinearRegression().fit([[0], [1]], [0, 1])
    model.rank_ = SimpleNamespace(hidden="state")
    with pytest.raises(MLBundleError):
        Adapter().deserialize(sio.dumps(model))


def test_unknown_nested_estimator_state_rejected():
    model = LinearRegression().fit([[0], [1]], [0, 1])
    model.rank_ = StandardScaler()
    model.rank_.review_extra_ = 1
    with pytest.raises(InvalidModelError, match="Unknown sklearn estimator state"):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize("numeric", [False, True])
def test_unreferenced_payload_files_rejected(numeric):
    entries = unpack(sio.dumps(LinearRegression().fit([[0], [1]], [0, 1])))
    buffer = io.BytesIO()
    np.save(buffer, np.array([1.0]), allow_pickle=False)
    entries["unreferenced.npy" if numeric else "unreferenced.bin"] = (
        buffer.getvalue() if numeric else b"extra"
    )
    with pytest.raises(InvalidModelError, match="Unreferenced"):
        Adapter().deserialize(pack(entries))


def test_state_key_types_cannot_silently_discard_attributes():
    model = LinearRegression().fit([[0], [1]], [0, 1])

    def change(schema, entries):
        schema["content"]["key_types"]["content"].pop()

    with pytest.raises(InvalidModelError, match="state key types"):
        Adapter().deserialize(alter_schema(sio.dumps(model), change))


def test_unused_schema_objects_are_rejected():
    model = LinearRegression().fit([[0], [1]], [0, 1])

    def change(schema, entries):
        schema["unused"] = {
            "__loader__": "ListNode",
            "__module__": "builtins",
            "__class__": "list",
            "content": [],
        }

    with pytest.raises(InvalidModelError, match="node state"):
        Adapter().deserialize(alter_schema(sio.dumps(model), change))
