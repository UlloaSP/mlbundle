"""Smoke test for the installed package."""

import subprocess
import sys
from importlib.metadata import version

import mlbundle


def test_package_version() -> None:
    assert mlbundle.__version__ == version("mlbundle")
    assert all(hasattr(mlbundle, name) for name in mlbundle.__all__)


def test_base_import_does_not_touch_model_libraries():
    script = """
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'xgboost', 'onnx', 'onnxruntime', 'skops', 'numpy', 'pandas', 'scipy'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOptional())
import mlbundle
assert mlbundle.__version__
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
