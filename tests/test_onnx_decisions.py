"""ONNX uses an actual graph label output or reports no class decision."""

from types import SimpleNamespace

import numpy as np
import pytest
from onnx import TensorProto as T, helper as h

import mlbundle
from mlbundle._onnx import Adapter
from onnx_helpers import ROWS, classifier, regressor, softmax


@pytest.mark.parametrize("labels", [(10, 20), ("cold", "warm")])
@pytest.mark.parametrize("zipmap", [False, True])
def test_native_graph_label_output(labels, zipmap, tmp_path):
    path = tmp_path / "label.mlbundle"
    mlbundle.export(classifier(labels, zipmap=zipmap), path)
    loaded = mlbundle.load(path)
    expected = loaded.native_model.session.run(
        ["label"], {"X": np.asarray(ROWS, dtype=np.float32)}
    )[0]
    indices = [[list(labels).index(value)] for value in expected]
    assert loaded.predictor.predict_class_indices(ROWS) == indices
    for row, index in zip(ROWS, indices, strict=True):
        assert loaded.predict_record(dict(enumerate(row)))[0].predicted_label == str(
            labels[index[0]]
        )


@pytest.mark.parametrize("strings", [False, True])
@pytest.mark.parametrize("rank_two", [False, True])
def test_declared_labels_and_native_encoded_decisions(strings, rank_two, tmp_path):
    native = softmax(two_outputs=True)
    if rank_two:
        native.graph.node[-1].ClearField("attribute")
        native.graph.node[-1].attribute.extend(
            [h.make_attribute("axis", 1), h.make_attribute("keepdims", 1)]
        )
        native.graph.output[0].CopyFrom(
            h.make_tensor_value_info("label", T.INT64, [None, 1])
        )
    if strings:
        native.graph.node.append(
            h.make_node(
                "LabelEncoder",
                ["label"],
                ["named"],
                domain="ai.onnx.ml",
                keys_int64s=[0, 1],
                values_strings=["cold", "warm"],
            )
        )
        native.graph.output[0].CopyFrom(
            h.make_tensor_value_info(
                "named", T.STRING, [None, 1] if rank_two else [None]
            )
        )
    path = tmp_path / "declared.mlbundle"
    mlbundle.export(native, path, class_labels=[["cold", "warm"]])
    loaded = mlbundle.load(path)
    name = "named" if strings else "label"
    expected = loaded.native_model.session.run(
        [name], {"X": np.asarray(ROWS, dtype=np.float32)}
    )[0].reshape(-1)
    labels = [
        str(value) if strings else ["cold", "warm"][int(value)] for value in expected
    ]
    assert [
        loaded.predict_record(dict(enumerate(row)))[0].predicted_label for row in ROWS
    ] == labels


@pytest.mark.parametrize(
    "native",
    [softmax(), classifier(map_only=True), classifier(zipmap=True, map_only=True)],
)
def test_probabilities_without_label_output(native, tmp_path):
    path = tmp_path / "no-decision.mlbundle"
    mlbundle.export(native, path)
    loaded = mlbundle.load(path)
    assert loaded.predictor.predict_class_indices(ROWS) is None
    assert loaded.predict_record({"0": 0, "1": 1})[0].predicted_label is None


def test_native_decision_may_disagree_with_probability(tmp_path):
    native = softmax(two_outputs=True)
    native.graph.node.pop()
    native.graph.node.append(
        h.make_node(
            "Constant",
            [],
            ["label"],
            value=h.make_tensor("decision", T.INT64, [1], [1]),
        )
    )
    path = tmp_path / "independent.mlbundle"
    mlbundle.export(native, path, class_labels=[["cold", "warm"]])
    result = mlbundle.load(path).predict_record({"0": 0, "1": 1})[0]
    assert (
        result.predicted_label == "warm"
        and result.probabilities[0] > result.probabilities[1]
    )


@pytest.mark.parametrize(
    "raw",
    [
        [],
        [10, 20],
        [[10, 20]],
        [True],
        [10.0],
        [np.nan],
        [np.inf],
        [999],
        [0],
        ["private"],
        [None],
    ],
)
def test_invalid_native_label_output(raw):
    adapter = Adapter()
    predictor = adapter.predictor(adapter.deserialize(classifier().SerializeToString()))
    seen = []

    def run(outputs, _feeds):
        seen.append(outputs)
        return [raw]

    predictor.session = SimpleNamespace(run=run)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        predictor.predict_class_indices([[0, 1]])
    assert seen == [["label"]]
    assert "private" not in str(error.value) and len(str(error.value)) < 160


@pytest.mark.parametrize("raw", [["private"], [None], ["x" * 1025], [complex(1)]])
def test_invalid_string_label_output(raw):
    adapter = Adapter()
    predictor = adapter.predictor(
        adapter.deserialize(classifier(("cold", "warm")).SerializeToString())
    )
    predictor.session = SimpleNamespace(run=lambda *args: [raw])
    with pytest.raises(mlbundle.InvalidModelError) as error:
        predictor.predict_class_indices([[0, 1]])
    assert "private" not in str(error.value) and len(str(error.value)) < 160


def test_native_failure_and_regression_boundary():
    adapter = Adapter()
    predictor = adapter.predictor(adapter.deserialize(classifier().SerializeToString()))

    def fail(*args):
        raise ValueError("private native exception")

    predictor.session = SimpleNamespace(run=fail)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        predictor.predict_class_indices([[0, 1]])
    assert "private" not in str(error.value)
    with pytest.raises(ValueError, match="classifier"):
        adapter.predictor(
            adapter.deserialize(regressor().SerializeToString())
        ).predict_class_indices([[0, 1]])


@pytest.mark.parametrize("source", [(0, 1), (10, 20), ("encoded-a", "encoded-b")])
@pytest.mark.parametrize("renamed", [("cold", "warm"), (100, 200)])
def test_zipmap_renaming_preserves_native_decision_positions(source, renamed, tmp_path):
    native = classifier(source, zipmap=True)
    node = native.graph.node[1]
    node.ClearField("attribute")
    strings = isinstance(renamed[0], str)
    node.attribute.append(
        h.make_attribute(
            "classlabels_strings" if strings else "classlabels_int64s", list(renamed)
        )
    )
    native.graph.output[1].type.sequence_type.elem_type.map_type.key_type = (
        T.STRING if strings else T.INT64
    )
    native.graph.node.append(h.make_node("Identity", ["label"], ["decision"]))
    native.graph.output[0].name = "decision"
    native.graph.node.insert(1, h.make_node("Identity", ["P"], ["scores"]))
    node.input[0] = "scores"
    path = tmp_path / "renamed.mlbundle"
    mlbundle.export(native, path)
    loaded = mlbundle.load(path)
    expected = loaded.native_model.session.run(
        ["decision"], {"X": np.asarray(ROWS, dtype=np.float32)}
    )[0]
    for row, label in zip(ROWS, expected, strict=True):
        assert loaded.predict_record(dict(enumerate(row)))[0].predicted_label == str(
            renamed[list(source).index(label)]
        )


def test_zipmap_with_graph_encoded_argmax(tmp_path):
    native = softmax(two_outputs=True)
    native.graph.node.append(
        h.make_node(
            "ZipMap",
            ["P"],
            ["probabilities"],
            domain="ai.onnx.ml",
            classlabels_strings=["cold", "warm"],
        )
    )
    native.graph.output[1].CopyFrom(
        h.make_value_info(
            "probabilities",
            h.make_sequence_type_proto(
                h.make_map_type_proto(T.STRING, h.make_tensor_type_proto(T.FLOAT, []))
            ),
        )
    )
    path = tmp_path / "mapped-softmax.mlbundle"
    mlbundle.export(native, path)
    loaded = mlbundle.load(path)
    expected = loaded.native_model.session.run(
        ["label"], {"X": np.asarray(ROWS, dtype=np.float32)}
    )[0]
    assert [
        loaded.predict_record(dict(enumerate(row)))[0].predicted_label for row in ROWS
    ] == [["cold", "warm"][int(value)] for value in expected]


@pytest.mark.parametrize("label", [0, 20])
def test_independent_head_still_requires_a_known_native_label(label, tmp_path):
    native = classifier((10, 20), zipmap=True)
    native.graph.node[0].output[0] = "unused"
    native.graph.node.append(
        h.make_node(
            "Constant",
            [],
            ["label"],
            value=h.make_tensor("decision", T.INT64, [1], [label]),
        )
    )
    path = tmp_path / "independent-label.mlbundle"
    mlbundle.export(native, path)
    loaded = mlbundle.load(path)
    if label == 0:
        with pytest.raises(mlbundle.InvalidModelError):
            loaded.predict_record({"0": 0, "1": 1})
    else:
        assert loaded.predict_record({"0": 0, "1": 1})[0].predicted_label == "20"


def test_string_tensor_rejects_object_array_numeric_cells():
    native = classifier((10, 20))
    native.graph.node[0].output[0] = "unused"
    native.graph.node.append(
        h.make_node(
            "Constant",
            [],
            ["label"],
            value=h.make_tensor("decision", T.STRING, [1], ["10"]),
        )
    )
    native.graph.output[0].CopyFrom(h.make_tensor_value_info("label", T.STRING, [None]))
    adapter = Adapter()
    predictor = adapter.predictor(adapter.deserialize(native.SerializeToString()))
    predictor.session = SimpleNamespace(
        run=lambda *args: [np.array([10], dtype=object)]
    )
    with pytest.raises(
        mlbundle.InvalidModelError, match=r"^Invalid ONNX class prediction output\.$"
    ):
        predictor.predict_class_indices([[0, 1]])
