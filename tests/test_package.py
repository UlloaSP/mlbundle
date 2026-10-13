"""Smoke test for the installed package."""

import subprocess
import sys
from importlib.metadata import version

import pytest

import mlbundle


def test_package_version() -> None:
    assert mlbundle.__version__ == version("mlbundle")
    assert all(hasattr(mlbundle, name) for name in mlbundle.__all__)


def test_base_import_does_not_touch_model_libraries():
    script = """
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'xgboost', 'catboost', 'onnx', 'onnxruntime', 'skops', 'numpy', 'pandas', 'scipy', 'mlschema'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOptional())
import mlbundle
assert mlbundle.__version__
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("library", ["sklearn", "xgboost", "onnx"])
def test_reader_works_without_schema_dependencies(library, tmp_path):
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    from onnx_helpers import regressor

    frame = pd.DataFrame({"a": [0, 1], "b": [1.0, 0.0]})
    if library == "onnx":
        model = regressor()
    elif library == "sklearn":
        model = DecisionTreeRegressor().fit(frame, [0, 1])
    else:
        model = XGBRegressor(n_estimators=1, n_jobs=1).fit(frame, [0, 1])
    path = tmp_path / "schema.mlbundle"
    mlbundle.export(model, path, training_data=frame, sample_rows=1)
    script = """
import sys
class WithoutSchema:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'mlschema'}:
            raise ModuleNotFoundError(name=fullname)
sys.meta_path.insert(0, WithoutSchema())
import mlbundle
assert not {'pandas', 'mlschema'} & sys.modules.keys()
view = mlbundle.inspect(sys.argv[1])
assert view.schema and view.samples == [{'a': 0, 'b': 1.0}]
loaded = mlbundle.load(sys.argv[1])
assert loaded.schema == view.schema and loaded.samples == view.samples
assert len(loaded.predict([[0, 1]])) == 1
assert not {'pandas', 'mlschema'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
