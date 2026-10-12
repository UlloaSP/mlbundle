"""Wheel checks run in separate environments, shared by CI and local verification."""

import argparse
import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from tarfile import open as open_tar
from zipfile import ZipFile

_OPTIONAL = {
    "sklearn",
    "skops",
    "xgboost",
    "onnx",
    "onnxruntime",
    "numpy",
    "scipy",
    "pandas",
    "mlschema",
}


def _console(*args: str) -> str:
    executable = Path(sys.executable).with_name(
        "mlbundle.exe" if sys.platform == "win32" else "mlbundle"
    )
    result = subprocess.run(
        [executable, *args], capture_output=True, text=True, check=True
    )
    assert result.stderr == "", result.stderr
    return result.stdout


def _sklearn(path: Path) -> None:
    from sklearn.tree import DecisionTreeRegressor

    import mlbundle

    model = DecisionTreeRegressor(random_state=0).fit([[0, 1], [1, 0]], [0, 1])
    mlbundle.export(model, path, feature_names=["temperature", "humidity"])
    bundle = mlbundle.load(path)
    assert isinstance(bundle.native_model, DecisionTreeRegressor)
    assert bundle.predict([[0, 1], [1, 0]]) == [0.0, 1.0]
    sys.stdout.write("sklearn wheel: export/load/predict OK\n")


def _base(path: Path) -> None:
    assert all(find_spec(name) is None for name in _OPTIONAL)

    import mlbundle

    assert mlbundle.__version__ == "0.1.0"
    view = mlbundle.inspect(path)
    assert view.manifest.model.library == "sklearn"
    assert "cannot attest" in _console("--help")
    assert "model claims NOT verified" in _console("inspect", str(path))
    assert json.loads(
        _console("inspect", str(path), "--json")
    ) == view.manifest.model_dump(mode="json", by_alias=True)
    try:
        mlbundle.load(path)
    except mlbundle.MissingOptionalDependencyError as error:
        assert error.extra == "sklearn"
        sys.stdout.write(f"load: {type(error).__name__}: {error}\n")
    else:
        raise AssertionError("base installation unexpectedly loaded sklearn")
    assert not _OPTIONAL & sys.modules.keys()
    sys.stdout.write("base wheel: import/help/inspect OK; optional libraries absent\n")


def _contents(directory: Path) -> None:
    (wheel,) = directory.glob("mlbundle-*.whl")
    (sdist,) = directory.glob("mlbundle-*.tar.gz")
    with ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        assert "mlbundle/py.typed" in wheel_names
        assert "mlbundle/_sklearn/state_fields.json" in wheel_names
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in wheel_names)
        assert any(name.endswith(".dist-info/entry_points.txt") for name in wheel_names)
        assert all(name.startswith(("mlbundle/", "mlbundle-")) for name in wheel_names)
    with open_tar(sdist) as archive:
        sdist_names = archive.getnames()
        assert any(name.endswith("/src/mlbundle/py.typed") for name in sdist_names)
        assert any(name.endswith("/LICENSE") for name in sdist_names)
    for name in [*wheel_names, *sdist_names]:
        assert not set(Path(name).parts) & {"tests", ".venv", ".git", "__pycache__"}, (
            name
        )
        assert not name.endswith((".pyc", ".mlbundle", "coverage.xml")), name
    sys.stdout.write(
        f"artifacts: {wheel.name}, {sdist.name}; py.typed/catalog/license present; no tests or stray files\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "base", "sklearn", "contents"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "contents":
        _contents(args.path)
    elif args.action == "base":
        _base(args.path)
    else:
        if args.action == "sklearn":
            absent = _OPTIONAL - {"sklearn", "skops", "numpy", "scipy"}
            assert all(find_spec(name) is None for name in absent)
        _sklearn(args.path)


if __name__ == "__main__":
    main()
