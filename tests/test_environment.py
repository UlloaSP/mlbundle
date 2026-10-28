"""Training environment declarations; package versions live in the lock."""

import sysconfig

import pytest
from pydantic import ValidationError

from mlbundle import OnnxInfo, Opset, environment as module


def test_installed_environment():
    environment = module.capture_environment()
    assert environment.free_threaded is bool(
        sysconfig.get_config_var("Py_GIL_DISABLED")
    )
    assert environment.python_version.startswith("3.14")
    assert environment.python_implementation
    assert environment.os and environment.os_version and environment.architecture
    assert environment.onnx is None


def test_free_threaded_build_and_onnx(monkeypatch):
    monkeypatch.setattr(module.sysconfig, "get_config_var", lambda name: 1)
    graph = OnnxInfo(
        opsets=[Opset(domain="", version=21)],
        ir_version=10,
        producer_name="",
        producer_version="",
    )
    environment = module.capture_environment(onnx=graph)
    assert environment.free_threaded is True
    assert environment.onnx == graph


@pytest.mark.parametrize("domains", [["ai.onnx.ml", ""], ["", ""]])
def test_onnx_opset_order_and_uniqueness(domains):
    with pytest.raises(ValidationError):
        OnnxInfo(
            opsets=[Opset(domain=domain, version=21) for domain in domains],
            ir_version=10,
            producer_name="",
            producer_version="",
        )
