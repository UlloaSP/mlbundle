"""Real-exporter layouts remain bounded, unchanged and validated once per operation."""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto as T, helper as h

import mlbundle
from mlbundle._onnx import protobuf, validation
from onnx_helpers import classifier, regressor


def tree_ensemble():
    trees, nodes = 8192, 15
    count = trees * nodes
    branches = nodes // 2
    leaves = list(range(branches, nodes))
    tree = h.make_node(
        "TreeEnsembleClassifier",
        ["X"],
        ["label", "scores"],
        domain="ai.onnx.ml",
        classlabels_int64s=[0, 1],
        nodes_treeids=[tree for tree in range(trees) for _ in range(nodes)],
        nodes_nodeids=list(range(nodes)) * trees,
        nodes_featureids=[0] * count,
        nodes_values=[0.0] * count,
        nodes_modes=(["BRANCH_LEQ"] * branches + ["LEAF"] * len(leaves)) * trees,
        nodes_truenodeids=([2 * i + 1 for i in range(branches)] + [0] * len(leaves))
        * trees,
        nodes_falsenodeids=([2 * i + 2 for i in range(branches)] + [0] * len(leaves))
        * trees,
        nodes_missing_value_tracks_true=[0] * count,
        class_treeids=[tree for tree in range(trees) for _ in leaves],
        class_nodeids=leaves * trees,
        class_ids=[0] * (trees * len(leaves)),
        class_weights=[0.25 / trees] * (trees * len(leaves)),
    )
    probabilities = h.make_value_info(
        "probabilities",
        h.make_sequence_type_proto(
            h.make_map_type_proto(T.INT64, h.make_tensor_type_proto(T.FLOAT, []))
        ),
    )
    native = h.make_model(
        h.make_graph(
            [
                tree,
                h.make_node(
                    "ZipMap",
                    ["scores"],
                    ["probabilities"],
                    domain="ai.onnx.ml",
                    classlabels_int64s=[0, 1],
                ),
            ],
            "CatBoostModel",
            [h.make_tensor_value_info("X", T.FLOAT, [None, 2])],
            [h.make_tensor_value_info("label", T.INT64, [None]), probabilities],
        ),
        ir_version=3,
        opset_imports=[h.make_opsetid("ai.onnx.ml", 2)],
    )
    native.producer_name = "CatBoost"
    native.producer_version = "Git info:\n" + "a" * 353
    native.metadata_props.add(key="training", value="x" * 62946)
    return native


def test_large_unpacked_tree_exporter_roundtrip(tmp_path, monkeypatch):
    native = tree_ensemble()
    data = native.SerializeToString()
    budget = [0]
    protobuf._message(data, 0, len(data), native.DESCRIPTOR, depth=0, budget=budget)
    assert budget[0] > 1_000_000
    calls = []
    original = validation.validate_wire

    def scan(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(validation, "validate_wire", scan)
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(data, path)
    assert calls == [len(data)]
    calls.clear()
    loaded = mlbundle.load(path)
    assert calls == [len(data)]
    assert loaded.native_model.data == data
    assert loaded.class_labels == [[0, 1]]
    assert loaded.manifest.environment.onnx.producer_version == native.producer_version
    rows = np.array([[-1, 0], [1, 0]], dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
    expected = session.run(None, {"X": rows})[1]
    assert loaded.predict_proba(rows) == [[row[0], row[1]] for row in expected]


@pytest.mark.parametrize("field", ["producer_name", "producer_version"])
def test_producer_string_limit_names_field_without_value(field, tmp_path):
    native = regressor()
    setattr(native, field, "SECRET\n" + "x" * 65536)
    with pytest.raises(mlbundle.InvalidModelError) as error:
        mlbundle.export(native, tmp_path / "bad.mlbundle")
    message = str(error.value)
    assert field in message and "65536" in message
    assert "SECRET" not in message and len(message) < 200


@pytest.mark.parametrize("field", ["producer_name", "producer_version"])
def test_producer_strings_at_limit_roundtrip(field, tmp_path):
    native = regressor()
    value = "".join(map(chr, np.random.default_rng(7).integers(33, 127, 65536)))
    setattr(native, field, value)
    path = tmp_path / "producer.mlbundle"
    mlbundle.export(native, path)
    assert getattr(mlbundle.load(path).manifest.environment.onnx, field) == value


def test_oversized_metadata_names_projection_limit(tmp_path):
    native = regressor()
    native.metadata_props.add(key="SECRET", value="x" * (1024 * 1024))
    with pytest.raises(mlbundle.LimitExceededError) as error:
        mlbundle.export(native, tmp_path / "bad.mlbundle")
    assert "metadata_props" in str(error.value) and "1048576" in str(error.value)
    assert "SECRET" not in str(error.value)


def test_oversized_native_label_names_shared_limit(tmp_path):
    with pytest.raises(mlbundle.InvalidModelError) as error:
        mlbundle.export(
            classifier(("SECRET" + "x" * 1024, "other")), tmp_path / "bad.mlbundle"
        )
    assert "class_labels" in str(error.value) and "1024" in str(error.value)
    assert "SECRET" not in str(error.value)


def test_description_cache_is_scoped_to_bytes_identity(tmp_path, monkeypatch):
    calls = []
    original = validation.validate_wire

    def scan(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(validation, "validate_wire", scan)
    path = tmp_path / "small.mlbundle"
    mlbundle.export(regressor(), path)
    assert len(calls) == 1
    calls.clear()
    first = mlbundle.load(path).native_model
    mlbundle.load(path)
    assert len(calls) == 2
    calls.clear()
    mlbundle.export(first, tmp_path / "second.mlbundle")
    assert len(calls) == 1
