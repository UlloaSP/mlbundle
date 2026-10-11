"""ONNX-ML trees, nested inference graphs and label/metadata boundaries."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto as T, helper as h

import mlbundle
from mlbundle import InvalidModelError, LimitExceededError
from mlbundle._onnx import Adapter
from onnx_helpers import ROWS, classifier, model, regressor


@pytest.mark.parametrize("labels", [[10, 20], ["", "é"]])
def test_tree_classifier(labels, tmp_path):
    attrs = {
        "classlabels_strings"
        if isinstance(labels[0], str)
        else "classlabels_int64s": labels
    }
    node = h.make_node(
        "TreeEnsembleClassifier",
        ["X"],
        ["label", "P"],
        domain="ai.onnx.ml",
        nodes_treeids=[0, 0, 0],
        nodes_nodeids=[0, 1, 2],
        nodes_featureids=[0, 0, 0],
        nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
        nodes_values=[0.5, 0.0, 0.0],
        nodes_truenodeids=[1, 0, 0],
        nodes_falsenodeids=[2, 0, 0],
        class_treeids=[0, 0, 0, 0],
        class_nodeids=[1, 1, 2, 2],
        class_ids=[0, 1, 0, 1],
        class_weights=[0.8, 0.2, 0.1, 0.9],
        **attrs,
    )
    native = model(
        [node],
        [h.make_tensor_value_info("X", T.FLOAT, [None, 2])],
        [
            h.make_tensor_value_info(
                "label", T.STRING if isinstance(labels[0], str) else T.INT64, [None]
            ),
            h.make_tensor_value_info("P", T.FLOAT, [None, 2]),
        ],
    )
    data = native.SerializeToString()
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(data, path)
    loaded = mlbundle.load(path)
    expected = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(
        ["P"], {"X": np.asarray(ROWS, dtype=np.float32)}
    )[0]
    assert np.array_equal(loaded.predict_proba(ROWS), expected)
    assert loaded.class_labels == [labels] and loaded.structure is None


def test_nested_inference_graph(tmp_path):
    branch = h.make_graph(
        [h.make_node("Identity", ["X"], ["Y"])],
        "branch",
        [],
        [h.make_tensor_value_info("Y", T.FLOAT, [None, 1])],
    )
    native = model(
        [
            h.make_node(
                "If", ["condition"], ["result"], then_branch=branch, else_branch=branch
            )
        ],
        [h.make_tensor_value_info("X", T.FLOAT, [None, 1])],
        [h.make_tensor_value_info("result", T.FLOAT, [None, 1])],
        initializers=[h.make_tensor("condition", T.BOOL, [], [True])],
    )
    path = tmp_path / "nested.mlbundle"
    mlbundle.export(native, path)
    assert mlbundle.load(path).predict([[1], [2]]) == [1, 2]


def test_verified_metadata_properties(tmp_path):
    native = regressor()
    native.metadata_props.add(key="experiment", value="one")
    path = tmp_path / "metadata.mlbundle"
    mlbundle.export(native, path)
    assert mlbundle.load(path).manifest.model.hyperparameters["metadata_props"] == {
        "experiment": "one"
    }
    native.metadata_props.add(key="experiment", value="two")
    with pytest.raises(InvalidModelError):
        mlbundle.export(native, tmp_path / "bad.mlbundle")


@pytest.mark.parametrize("labels", [[10, 10], ["x" * 1025, "y"], ["bad\ud800", "y"]])
def test_invalid_native_labels(labels, tmp_path):
    if any(isinstance(value, str) and "\ud800" in value for value in labels):
        native = classifier(["placeholder", "other"])
        native.graph.node[0].attribute[0].strings[0] = b"\xff"
    else:
        native = classifier(labels)
    with pytest.raises(InvalidModelError):
        mlbundle.export(native, tmp_path / "bad.mlbundle")


def test_value_and_sparse_bounds_before_session(monkeypatch):
    native = regressor()
    native.graph.value_info.append(h.make_tensor_value_info("unsafe", T.FLOAT, [2**50]))
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    with pytest.raises(LimitExceededError):
        Adapter().deserialize(native.SerializeToString())
    native = regressor()
    native.graph.sparse_initializer.append(
        onnx.SparseTensorProto(
            values=h.make_tensor("sparse", T.DOUBLE, [1], [0]),
            indices=h.make_tensor("indices", T.INT64, [1], [0]),
            dims=[100_000_000],
        )
    )
    with pytest.raises(LimitExceededError, match="sparse"):
        Adapter().deserialize(native.SerializeToString())
