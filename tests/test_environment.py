"""Environment declarations use installed distribution metadata."""

from importlib import metadata

import pytest
from pydantic import ValidationError

from mlbundle import OnnxInfo, Opset, environment as module


def test_installed_environment():
    environment = module.capture_environment()
    assert environment.packages["mlbundle"] == metadata.version("mlbundle")
    assert environment.python_version.startswith("3.14")
    assert environment.python_implementation
    assert environment.os and environment.os_version and environment.architecture
    assert environment.onnx is None


def test_missing_distributions_and_onnx(monkeypatch):
    def version(name):
        if name == "mlbundle":
            return "0.1.0a2"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(module.metadata, "version", version)
    graph = OnnxInfo(
        opsets=[Opset(domain="", version=21)],
        ir_version=10,
        producer_name="",
        producer_version="",
    )
    environment = module.capture_environment(onnx=graph)
    assert environment.packages == {"mlbundle": "0.1.0a2"}
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
