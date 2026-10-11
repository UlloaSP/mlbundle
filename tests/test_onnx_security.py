"""ONNX graph and protobuf attacks fail before any native session or disk access."""

from pathlib import Path

import onnxruntime as ort
import pytest
from onnx import TensorProto as T, helper as h

import mlbundle
from mlbundle import InvalidModelError, LimitExceededError, MemberName
from mlbundle._onnx import Adapter
from onnx_helpers import classifier, external_model, external_tensor, regressor, rewrite


@pytest.mark.parametrize(
    "where", ["initializer", "attribute", "subgraph", "sparse", "function"]
)
def test_external_data_before_session(where, tmp_path, monkeypatch):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"server secret")
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    data = external_model(where, outside).SerializeToString()
    hostile = rewrite(path, lambda m, b: b.update({MemberName.ONNX: data}))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail(
            "External references must be rejected before native session creation"
        )

    original = Path.open

    def guarded(self, *args, **kwargs):
        assert self != outside
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    for action in [
        lambda: mlbundle.export(data, tmp_path / "bad.mlbundle"),
        lambda: mlbundle.load(hostile),
    ]:
        with pytest.raises(InvalidModelError, match="external"):
            action()
    assert not calls


@pytest.mark.parametrize("external_flag", [False, True])
def test_either_external_marker_is_rejected(external_flag, monkeypatch):
    native = regressor()
    tensor = external_tensor(entries=not external_flag)
    if not external_flag:
        tensor.data_location = T.DEFAULT
    native.graph.initializer.append(tensor)
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    with pytest.raises(InvalidModelError, match="external"):
        Adapter().deserialize(native.SerializeToString())


def hostile_models():
    domain = regressor()
    domain.graph.node[0].domain = "evil"
    nested = external_model("subgraph")
    nested.graph.node[0].attribute[0].g.node[0].domain = "evil"
    nested.graph.node[0].attribute[0].g.node[0].attribute.clear()
    nested.graph.node[0].attribute[1].g.node[0].attribute.clear()
    huge = regressor()
    huge.graph.initializer.append(h.make_tensor("huge", T.FLOAT, [1], [0.0]))
    huge.graph.initializer[0].dims[0] = 2**50
    mismatch = regressor()
    mismatch.graph.initializer.append(h.make_tensor("short", T.FLOAT, [1], [0.0]))
    mismatch.graph.initializer[0].dims[0] = 5
    function = regressor()
    function.functions.append(
        h.make_function(
            "ai.onnx.ml",
            "LinearRegressor",
            ["X"],
            ["Y"],
            [h.make_node("Identity", ["X"], ["Y"])],
            [h.make_opsetid("", 21)],
        )
    )
    evil_function = regressor()
    evil_function.functions.append(
        h.make_function(
            "evil",
            "Hidden",
            ["X"],
            ["Y"],
            [h.make_node("Identity", ["X"], ["Y"])],
            [h.make_opsetid("", 21)],
        )
    )
    training = regressor()
    training.training_info.add()
    config = regressor()
    config.metadata_props.add(
        key="ort_config",
        value='{"session_options":{"optimized_model_filepath":"outside.onnx"}}',
    )
    malformed = regressor()
    malformed.graph.node[0].input[0] = "missing"
    return [
        (domain, InvalidModelError),
        (nested, InvalidModelError),
        (huge, LimitExceededError),
        (mismatch, InvalidModelError),
        (function, InvalidModelError),
        (evil_function, InvalidModelError),
        (training, InvalidModelError),
        (config, InvalidModelError),
        (malformed, InvalidModelError),
    ]


@pytest.mark.parametrize("native,error", hostile_models())
def test_hostile_graphs_at_export_and_load(native, error, tmp_path, monkeypatch):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    payload = native.SerializeToString()
    hostile = rewrite(path, lambda m, b: b.update({MemberName.ONNX: payload}))
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    for action in [
        lambda: mlbundle.export(payload, tmp_path / "bad.mlbundle"),
        lambda: mlbundle.load(hostile),
    ]:
        with pytest.raises(error):
            action()


@pytest.mark.parametrize(
    "data",
    [
        b"random bytes",
        b"",
        b"\x08" + b"\x80" * 11,
        b"\x08\x01\x08\x02",
        b"\xf8\x7f\x01",
        regressor().SerializeToString()[:-7],
    ],
)
def test_hostile_protobuf(data, tmp_path, monkeypatch):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    hostile = rewrite(path, lambda m, b: b.update({MemberName.ONNX: data}))
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("session called")
    )
    with pytest.raises(InvalidModelError):
        mlbundle.load(hostile)


def test_bounded_native_error(tmp_path, monkeypatch):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)

    def invalid(*args, **kwargs):
        raise RuntimeError("secret file path " + "x" * 10000)

    monkeypatch.setattr(ort, "InferenceSession", invalid)
    for action in [
        lambda: mlbundle.export(regressor(), tmp_path / "bad.mlbundle"),
        lambda: mlbundle.load(path),
    ]:
        with pytest.raises(InvalidModelError, match="session creation") as error:
            action()
        assert "secret" not in str(error.value) and len(str(error.value)) < 100


def test_protobuf_depth_and_field_limits(monkeypatch):
    from mlbundle._onnx import protobuf

    native = regressor().SerializeToString()
    monkeypatch.setattr(protobuf, "MAX_DEPTH", 2)
    with pytest.raises(LimitExceededError, match="nesting"):
        Adapter().deserialize(native)
    monkeypatch.setattr(protobuf, "MAX_DEPTH", 32)
    monkeypatch.setattr(protobuf, "MAX_FIELDS", 1)
    with pytest.raises(LimitExceededError, match="field count"):
        Adapter().deserialize(native)


def test_graph_and_tensor_limits(monkeypatch):
    from mlbundle._onnx import validation

    monkeypatch.setattr(validation, "MAX_NODES", 0)
    with pytest.raises(LimitExceededError, match="node count"):
        Adapter().deserialize(regressor().SerializeToString())
    monkeypatch.setattr(validation, "MAX_NODES", 100_000)
    native = regressor()
    native.graph.initializer.append(h.make_tensor("unused", T.DOUBLE, [2], [0, 1]))
    monkeypatch.setattr(validation, "BYTE_CAP", 8)
    with pytest.raises(LimitExceededError, match="storage"):
        Adapter().deserialize(native.SerializeToString())
