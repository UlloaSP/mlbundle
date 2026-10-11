"""Original-byte ONNX round trips, native output layouts and optional imports."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

import mlbundle
from mlbundle import MemberName, UnsupportedModelError
from mlbundle._onnx import OnnxModel
from mlbundle.container import read_container
from onnx_helpers import ROWS, classifier, regressor, softmax


@pytest.mark.parametrize("form", ["path", "str", "bytes", "proto"])
@pytest.mark.parametrize(
    "labels,zipmap,map_only",
    [
        ([10, 20], False, False),
        (["cold", "warm"], False, False),
        ([20, 10], True, False),
        (["warm", "cold"], True, True),
    ],
)
def test_roundtrip(form, labels, zipmap, map_only, tmp_path):
    native = classifier(labels, zipmap=zipmap, map_only=map_only)
    original = native.SerializeToString()
    source = tmp_path / "original.onnx"
    source.write_bytes(original)
    inputs = {"path": source, "str": str(source), "bytes": original, "proto": native}
    path = tmp_path / "classifier.mlbundle"
    mlbundle.export(inputs[form], path)
    loaded = mlbundle.load(path)
    assert isinstance(loaded.native_model, OnnxModel)
    assert isinstance(loaded.native_model.session, ort.InferenceSession)
    assert loaded.native_model.data == original
    assert loaded.class_labels == [labels]
    assert loaded.structure is None and loaded.class_name == "ONNX"
    assert loaded.manifest.features.model_dtypes == ["float32", "float32"]
    _, members = read_container(path)
    assert members[MemberName.ONNX] == source.read_bytes() == original
    direct = ort.InferenceSession(original, providers=["CPUExecutionProvider"])
    expected = direct.run(None, {"X": np.asarray(ROWS, dtype=np.float32)})[-1]
    if zipmap:
        expected = [[row[label] for label in labels] for row in expected]
    for rows in [ROWS, np.asarray(ROWS), pd.DataFrame(ROWS)]:
        assert np.array_equal(loaded.predict_proba(rows), expected)
    assert native.SerializeToString() == original


@pytest.mark.parametrize("two_outputs", [False, True])
@pytest.mark.parametrize(
    "labels", [None, [["cold", "warm"]], [[False, True]], [[1.0, 2.0]]]
)
def test_label_declarations(two_outputs, labels, tmp_path):
    original = softmax(two_outputs=two_outputs).SerializeToString()
    path = tmp_path / "model.mlbundle"
    mlbundle.export(original, path, class_labels=labels)
    loaded = mlbundle.load(path)
    assert loaded.class_labels == (labels or [[0, 1]])
    assert loaded.native_model.data == original
    expected = ort.InferenceSession(original, providers=["CPUExecutionProvider"]).run(
        ["P"], {"X": np.asarray(ROWS, dtype=np.float32)}
    )[0]
    assert np.array_equal(loaded.predict_proba(ROWS), expected)


@pytest.mark.parametrize("columns", [False, True])
def test_regressor(columns, tmp_path):
    native = regressor(columns=columns)
    original = native.SerializeToString()
    path = tmp_path / "regressor.mlbundle"
    mlbundle.export(native, path)
    loaded = mlbundle.load(path)
    rows = [[0, 1.0], [1, 0.0]]
    feeds = (
        {
            "a": np.array([[0], [1]], dtype=np.int64),
            "b": np.array([[1.0], [0.0]], dtype=np.float64),
        }
        if columns
        else {"X": np.asarray(rows, dtype=np.float32)}
    )
    expected = (
        ort.InferenceSession(original, providers=["CPUExecutionProvider"])
        .run(None, feeds)[0]
        .ravel()
    )
    for values in [
        rows,
        np.asarray(rows),
        pd.DataFrame(rows, columns=["a", "b"])[["b", "a"]]
        if columns
        else pd.DataFrame(rows),
    ]:
        assert np.array_equal(loaded.predict(values), expected)
    assert loaded.feature_origin == ("model" if columns else "generated")
    assert loaded.feature_names == (
        ("a", "b") if columns else ("feature_1", "feature_2")
    )
    assert loaded.manifest.features.model_dtypes == (
        ["int64", "float64"] if columns else ["float32"] * 2
    )


@pytest.mark.parametrize(
    "labels", [[["bad", "labels"]], [[10]], [[10, 10]], [[20, 10]], [["a", 1]], ["ab"]]
)
def test_native_label_rejection(labels, tmp_path):
    with pytest.raises(UnsupportedModelError):
        mlbundle.export(classifier(), tmp_path / "bad.mlbundle", class_labels=labels)
    assert not list(tmp_path.iterdir())


def test_native_label_agreement_and_regressor_rejection(tmp_path):
    path = tmp_path / "native.mlbundle"
    mlbundle.export(classifier(), path, class_labels=[[10, 20]])
    assert mlbundle.load(path).class_labels == [[10, 20]]
    for native, labels in [(regressor(), [[0, 1]]), (softmax(), [[0]])]:
        with pytest.raises(UnsupportedModelError):
            mlbundle.export(native, tmp_path / "bad.mlbundle", class_labels=labels)


def test_lazy_import_and_missing_extra(tmp_path, monkeypatch):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    code = "import sys,mlbundle; mlbundle.inspect(sys.argv[1]); assert not {'sklearn','skops','numpy','pandas','xgboost','onnx','onnxruntime'} & sys.modules.keys(); print('lazy import and inspect: OK')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lazy import and inspect: OK" in result.stdout
    original = mlbundle.registry.import_module

    def missing(name):
        if name == "mlbundle._onnx":
            raise ModuleNotFoundError(name="onnxruntime")
        return original(name)

    monkeypatch.setattr(mlbundle.registry, "import_module", missing)
    for native in [classifier(), classifier().SerializeToString(), Path("model.onnx")]:
        with pytest.raises(
            mlbundle.MissingOptionalDependencyError, match=r"mlbundle\[onnx\]"
        ):
            mlbundle.export(native, tmp_path / "missing.mlbundle")
    with pytest.raises(
        mlbundle.MissingOptionalDependencyError, match=r"mlbundle\[onnx\]"
    ):
        mlbundle.load(path)


def test_session_options(tmp_path, monkeypatch):
    calls = []
    original = ort.InferenceSession

    def session(data, *, sess_options, providers, **kwargs):
        assert isinstance(data, bytes)
        assert providers == ["CPUExecutionProvider"]
        assert kwargs["read_config_from_model"] is False
        assert (
            sess_options.intra_op_num_threads == sess_options.inter_op_num_threads == 1
        )
        assert sess_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert (
            sess_options.graph_optimization_level
            == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        assert (
            not sess_options.enable_profiling
            and not sess_options.optimized_model_filepath
        )
        assert (
            not sess_options.enable_mem_pattern
            and not sess_options.enable_cpu_mem_arena
        )
        calls.append(True)
        return original(data, sess_options=sess_options, providers=providers, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", session)
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    mlbundle.load(path)
    assert len(calls) == 3


def test_readme_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from onnx import TensorProto, helper

    from mlbundle import export, load

    graph = helper.make_graph(
        [helper.make_node("Softmax", ["features"], ["probabilities"], axis=1)],
        "classifier",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
    )
    original = model.SerializeToString()
    Path("trained.onnx").write_bytes(original)
    export("trained.onnx", "onnx.mlbundle", class_labels=[["cold", "warm"]])
    bundle = load("onnx.mlbundle")
    assert bundle.native_model.data == original
    assert bundle.class_labels == [["cold", "warm"]]
    assert bundle.predict_proba([[0.0, 0.0]]) == [[0.5, 0.5]]
