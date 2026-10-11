"""Tabular layouts, numeric conversion and invalid ONNX prediction outputs."""

from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
from onnx import TensorProto as T, helper as h

import mlbundle
from mlbundle import InvalidModelError
from mlbundle._onnx import Adapter
from onnx_helpers import ROWS, classifier, model, regressor, softmax


@pytest.mark.parametrize(
    "invalid",
    [
        [[0]],
        [[0, 1, 2]],
        [0, 1],
        [["0", "1"]],
        [[True, False]],
        [[float("nan"), 0]],
        [[float("inf"), 0]],
        [[1e100, 0]],
        [[0, 1], [0]],
    ],
)
def test_invalid_inputs(invalid):
    predictor = Adapter().predictor(
        Adapter().deserialize(classifier().SerializeToString())
    )
    with pytest.raises((InvalidModelError, ValueError)):
        predictor.predict_proba(invalid)


@pytest.mark.parametrize("invalid", [0.5, float("nan"), "1", True, 2**63, -(2**63) - 1])
def test_integer_conversion_rejections(invalid):
    predictor = Adapter().predictor(
        Adapter().deserialize(regressor(columns=True).SerializeToString())
    )
    with pytest.raises(InvalidModelError):
        predictor.predict([[invalid, 0]])


def test_integer_precision_and_named_frame():
    predictor = Adapter().predictor(
        Adapter().deserialize(regressor(columns=True).SerializeToString())
    )
    predictor.predict([[2**63 - 1, 0]])
    with pytest.raises(InvalidModelError, match="columns"):
        predictor.predict(pd.DataFrame([[0, 1]], columns=["wrong", "b"]))
    with pytest.raises(InvalidModelError, match="columns"):
        predictor.predict(pd.DataFrame([[0, 1]], columns=["a", "a"]))

    class UserFrame(pd.DataFrame):
        pass

    frame = UserFrame({"b": [10.0], "a": [1]})
    assert predictor.predict(frame) == predictor.predict(pd.DataFrame(frame)) == [-7.5]


def test_static_batch(tmp_path):
    path = tmp_path / "static.mlbundle"
    mlbundle.export(regressor(batch=1), path)
    loaded = mlbundle.load(path)
    assert loaded.predict([[0, 1]]) == [-0.5]
    with pytest.raises(ValueError, match="static batch"):
        loaded.predict(ROWS)


@pytest.mark.parametrize(
    "raw",
    [
        [[float("nan"), 0]],
        [[0.5, 0.5, 0]],
        [[0.4, 0.4]],
        [[1.1, -0.1]],
        ["malformed"],
        {"other": 1},
    ],
)
def test_invalid_probabilities(raw):
    predictor = Adapter().predictor(
        Adapter().deserialize(classifier().SerializeToString())
    )
    predictor.session = SimpleNamespace(run=lambda *args: [raw])
    with pytest.raises(InvalidModelError, match="probability"):
        predictor.predict_proba([[0, 1]])
    with pytest.raises(ValueError, match="regressor"):
        predictor.predict([[0, 1]])


@pytest.mark.parametrize(
    "raw",
    [
        [{10: 1.0}],
        [{10: 0.5, 20: 0.5, 30: 0}],
        [{10: float("nan"), 20: 0}],
        [[0.5, 0.5]],
        [{10: "bad", 20: "value"}],
    ],
)
def test_invalid_zipmap(raw):
    predictor = Adapter().predictor(
        Adapter().deserialize(classifier(zipmap=True).SerializeToString())
    )
    predictor.session = SimpleNamespace(run=lambda *args: [raw])
    with pytest.raises(InvalidModelError, match="probability"):
        predictor.predict_proba([[0, 1]])


@pytest.mark.parametrize("raw", [[float("inf")], [[0, 1]], [], ["invalid"]])
def test_invalid_regression(raw):
    predictor = Adapter().predictor(
        Adapter().deserialize(regressor().SerializeToString())
    )
    predictor.session = SimpleNamespace(run=lambda *args: [raw])
    with pytest.raises(InvalidModelError, match="regression"):
        predictor.predict([[0, 1]])
    with pytest.raises(ValueError, match="classifier"):
        predictor.predict_proba([[0, 1]])


def test_native_prediction_error_is_bounded():
    predictor = Adapter().predictor(
        Adapter().deserialize(regressor().SerializeToString())
    )

    def fail(*args):
        raise RuntimeError("private " + "x" * 10000)

    predictor.session = SimpleNamespace(run=fail)
    with pytest.raises(InvalidModelError) as error:
        predictor.predict([[0, 1]])
    assert "private" not in str(error.value) and len(str(error.value)) < 100


def unsupported_layouts():
    width = regressor()
    width.graph.input[0].type.tensor_type.shape.dim[1].ClearField("dim_value")
    rank = regressor()
    rank.graph.input[0].type.tensor_type.shape.dim.add(dim_value=1)
    dtype = regressor()
    dtype.graph.input[0].type.tensor_type.elem_type = T.STRING
    batch = regressor(batch=2)
    columns = regressor(columns=True)
    columns.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 2
    ambiguous = model(
        [h.make_node("Identity", ["X"], ["Y"])],
        [h.make_tensor_value_info("X", T.FLOAT, [None, 2])],
        [h.make_tensor_value_info("Y", T.FLOAT, [None, 2])],
    )
    targets = regressor()
    targets.graph.output.append(h.make_tensor_value_info("X", T.FLOAT, [None, 2]))
    unknown_classes = softmax()
    unknown_classes.graph.output[0].type.tensor_type.shape.dim[1].ClearField(
        "dim_value"
    )
    axis = softmax()
    axis.graph.node[-1].attribute[0].i = 0
    return [
        width,
        rank,
        dtype,
        batch,
        columns,
        ambiguous,
        targets,
        unknown_classes,
        axis,
    ]


@pytest.mark.parametrize("native", unsupported_layouts())
def test_unsupported_layout_before_session(native, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    with pytest.raises(InvalidModelError):
        mlbundle.export(native, tmp_path / "bad.mlbundle")
    assert not list(tmp_path.iterdir())


def test_dynamic_class_dimension_with_native_labels(tmp_path):
    native = classifier()
    native.graph.output[1].type.tensor_type.shape.dim[1].ClearField("dim_value")
    path = tmp_path / "dynamic.mlbundle"
    mlbundle.export(native, path)
    assert mlbundle.load(path).class_labels == [[10, 20]]


def test_softmax_identity_output(tmp_path):
    native = softmax()
    native.graph.node.append(h.make_node("Identity", ["P"], ["probabilities"]))
    native.graph.output[0].name = "probabilities"
    path = tmp_path / "identity.mlbundle"
    mlbundle.export(native, path)
    assert mlbundle.load(path).predict_proba([[0, 0]]) == [[0.5, 0.5]]


def test_rank_one_regression_and_empty_batch(tmp_path):
    native = regressor()
    native.graph.node.append(h.make_node("Squeeze", ["Y", "axes"], ["values"]))
    native.graph.initializer.append(h.make_tensor("axes", T.INT64, [1], [1]))
    native.graph.output[0].CopyFrom(h.make_tensor_value_info("values", T.FLOAT, [None]))
    path = tmp_path / "rank-one.mlbundle"
    mlbundle.export(native, path)
    loaded = mlbundle.load(path)
    assert loaded.predict([[0, 1]]) == [-0.5]
    assert loaded.predict(np.empty((0, 2))) == []


def test_labeled_probability_permutation_is_rejected(tmp_path, monkeypatch):
    native = classifier()
    native.graph.node.append(
        h.make_node("Gather", ["P", "order"], ["reordered"], axis=1)
    )
    native.graph.initializer.append(h.make_tensor("order", T.INT64, [2], [1, 0]))
    native.graph.output[1].name = "reordered"
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    with pytest.raises(InvalidModelError, match="transformation"):
        mlbundle.export(native, tmp_path / "bad.mlbundle")


def test_zipmap_owns_its_native_labels(tmp_path):
    native = classifier(zipmap=True)
    node = native.graph.node[1]
    node.attribute.clear()
    node.attribute.append(h.make_attribute("classlabels_strings", ["cold", "warm"]))
    native.graph.output[1].type.sequence_type.elem_type.map_type.key_type = T.STRING
    path = tmp_path / "mapped.mlbundle"
    mlbundle.export(native, path)
    loaded = mlbundle.load(path)
    assert loaded.class_labels == [["cold", "warm"]]
    expected = ort.InferenceSession(
        native.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": np.asarray(ROWS, dtype=np.float32)})[1]
    assert loaded.predict_proba(ROWS) == [
        [row["cold"], row["warm"]] for row in expected
    ]
