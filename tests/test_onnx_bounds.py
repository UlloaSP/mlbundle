"""Count consistency, storage representations and ONNX preflight ceilings."""

import onnxruntime as ort
import pytest
from onnx import TensorProto as T, helper as h

import mlbundle
from mlbundle import InvalidModelError, LimitExceededError, MemberName
from mlbundle._onnx import Adapter
from onnx_helpers import classifier, regressor, rewrite, softmax


@pytest.mark.parametrize("raw", [False, True])
def test_initializer_storage_roundtrip(raw, tmp_path):
    native = softmax()
    if raw:
        import numpy as np
        from onnx import numpy_helper

        native.graph.initializer[0].CopyFrom(
            numpy_helper.from_array(
                np.asarray([[-1, 1], [1, -1]], dtype=np.float32), name="W"
            )
        )
    path = tmp_path / "storage.mlbundle"
    mlbundle.export(native, path)
    assert mlbundle.load(path).predict_proba([[0, 0]]) == [[0.5, 0.5]]


@pytest.mark.parametrize(
    "type_,values", [(T.INT64, [0] * 1001), (T.FLOAT, [0.0] * 1001)]
)
def test_packed_scalars_count_before_parse(type_, values, monkeypatch):
    from mlbundle._onnx import protobuf

    native = regressor()
    native.graph.initializer.append(
        h.make_tensor("packed", type_, [len(values)], values)
    )
    monkeypatch.setattr(protobuf, "MAX_FIELDS", 100)
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    with pytest.raises(LimitExceededError, match="scalar count"):
        Adapter().deserialize(native.SerializeToString())


def inconsistent_models():
    targets = regressor()
    targets.graph.node[0].attribute.clear()
    targets.graph.node[0].attribute.extend(
        [
            h.make_attribute("coefficients", [2.0, -1.0, 1.0, 2.0]),
            h.make_attribute("intercepts", [0.5, 0.5]),
            h.make_attribute("targets", 2),
        ]
    )
    classes = softmax()
    classes.graph.initializer[0].CopyFrom(
        h.make_tensor("W", T.FLOAT, [2, 3], [1.0, 0.0, -1.0, 0.0, 1.0, -1.0])
    )
    classes.graph.initializer[1].CopyFrom(
        h.make_tensor("B", T.FLOAT, [3], [0.0, 0.0, 0.0])
    )
    return [targets, classes]


@pytest.mark.parametrize("native", inconsistent_models())
def test_actual_output_counts_must_match_graph(native, tmp_path):
    path = tmp_path / "valid.mlbundle"
    mlbundle.export(classifier(), path)
    hostile = rewrite(
        path, lambda m, b: b.update({MemberName.ONNX: native.SerializeToString()})
    )
    for action in [
        lambda: mlbundle.export(native, tmp_path / "bad.mlbundle"),
        lambda: mlbundle.load(hostile),
    ]:
        with pytest.raises(InvalidModelError):
            action()


@pytest.mark.parametrize(
    "alter",
    [
        lambda m: m.graph.initializer[0].dims.append(-1),
        lambda m: m.graph.initializer[0].dims.extend([1] * 33),
        lambda m: m.graph.initializer[0].ClearField("float_data"),
        lambda m: m.graph.initializer[0].int64_data.append(1),
        lambda m: m.graph.initializer[0].segment.SetInParent(),
        lambda m: m.graph.initializer[0].ClearField("dims"),
        lambda m: setattr(m.graph.initializer[0], "data_type", T.FLOAT8E4M3FN),
    ],
)
def test_invalid_tensor_storage_before_session(alter, monkeypatch):
    native = softmax()
    alter(native)
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    with pytest.raises(mlbundle.MLBundleError):
        Adapter().deserialize(native.SerializeToString())


def test_onnx_source_errors_and_lower_limits(tmp_path):
    with pytest.raises(InvalidModelError, match="read"):
        mlbundle.export(tmp_path / "missing.onnx", tmp_path / "bad.mlbundle")
    source = tmp_path / "model.onnx"
    source.write_bytes(regressor().SerializeToString())
    with pytest.raises(LimitExceededError):
        mlbundle.export(
            source,
            tmp_path / "bad.mlbundle",
            limits=mlbundle.ArchiveLimits(max_member_bytes=1),
        )
    assert not (tmp_path / "bad.mlbundle").exists()


def test_native_record_can_be_exported(tmp_path):
    path = tmp_path / "first.mlbundle"
    mlbundle.export(classifier(), path)
    native = mlbundle.load(path).native_model
    second = tmp_path / "second.mlbundle"
    mlbundle.export(native, second)
    assert mlbundle.load(second).native_model.data == native.data


def test_runtime_configuration_environment_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("ORT_LOAD_CONFIG_FROM_MODEL", "1")
    with pytest.raises(InvalidModelError, match="session creation"):
        mlbundle.export(regressor(), tmp_path / "bad.mlbundle")
