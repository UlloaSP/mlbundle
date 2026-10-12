"""Real CLI reads, verification, safe display, errors and lazy inspection."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import mlbundle
from mlbundle import MemberName
from mlbundle.cli import main
from mlbundle.container import read_container
from mlbundle.writing import write_container
from onnx_helpers import softmax

_CONSOLE = Path(sys.executable).with_name(
    "mlbundle.exe" if sys.platform == "win32" else "mlbundle"
)


@pytest.fixture(params=["sklearn", "xgboost", "onnx"])
def bundle_path(request, tmp_path):
    rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
    model = {
        "sklearn": lambda: DecisionTreeClassifier(random_state=0).fit(
            rows, [0, 1, 0, 1]
        ),
        "xgboost": lambda: XGBClassifier(n_estimators=1, n_jobs=1).fit(
            rows, [0, 1, 0, 1]
        ),
        "onnx": softmax,
    }[request.param]()
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model, path, metadata={"name": "CLI fixture"})
    return path


def test_inspect_and_verify(bundle_path, capsys):
    view = mlbundle.inspect(bundle_path)
    assert main(["inspect", str(bundle_path)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    for text in (
        "model claims NOT verified",
        "Format: mlbundle 1",
        str(view.manifest.bundle_id),
        "Writer:",
        view.manifest.model.library.value,
        "Counts: features=2, outputs=1, classes=[2]",
        "Features (generated):",
        "Class labels (output-major):",
        "Training:",
        "Packages:",
        "schema=no",
        "samples=no",
    ):
        assert text in output.out
    for member, record in view.manifest.integrity.items():
        assert (
            f"{member.value}: {record.size} bytes sha256={record.sha256}" in output.out
        )
    assert main(["verify", str(bundle_path)]) == 0
    output = capsys.readouterr()
    assert output.err == "" and len(output.out.splitlines()) == 1
    assert (
        "Valid bundle:" in output.out and "native model claims/structure" in output.out
    )


def test_console_entry_and_json(bundle_path):
    result = subprocess.run(
        [_CONSOLE, "inspect", str(bundle_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and result.stderr == ""
    assert (
        mlbundle.Manifest.from_bytes(result.stdout.encode())
        == mlbundle.inspect(bundle_path).manifest
    )


@pytest.mark.parametrize(
    "args",
    [[], ["unknown"], ["inspect"], ["verify", "file", "--json"], ["\x1b[31m\n" * 1000]],
)
def test_usage_error_is_one_safe_line(args, capsys):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    output = capsys.readouterr()
    assert (
        output.out == "" and output.err == "mlbundle: invalid arguments; use --help.\n"
    )


@pytest.mark.parametrize(
    "args", [["--help"], ["inspect", "--help"], ["verify", "--help"]]
)
def test_help_explains_inspection_limits(args, capsys):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 0
    text = capsys.readouterr().out
    assert "cannot attest" in text and "Only verify" in text


@pytest.mark.parametrize("command", ["inspect", "verify"])
def test_io_and_rejected_bundle(command, tmp_path, capsys):
    assert main([command, str(tmp_path / "missing")]) == 2
    output = capsys.readouterr()
    assert (
        output.out == "" and output.err == "mlbundle: FileNotFoundError: I/O failed.\n"
    )
    path = tmp_path / "invalid.mlbundle"
    path.write_bytes(b"secret\x1b[31m\n" * 1000)
    assert main([command, str(path)]) == 1
    output = capsys.readouterr()
    assert (
        output.out == ""
        and output.err == "mlbundle: InvalidContainerError: bundle rejected.\n"
    )


def test_verify_rejects_lie_that_inspect_accepts(bundle_path, capsys):
    manifest, members = read_container(bundle_path)
    parts = manifest.model_dump(mode="json")
    parts["model"]["estimator_class"] = "secret\x1b[31m"
    del members[MemberName.MANIFEST]
    bundle_path.write_bytes(write_container(parts, members))
    assert main(["inspect", str(bundle_path)]) == 0
    capsys.readouterr()
    assert main(["verify", str(bundle_path)]) == 1
    output = capsys.readouterr()
    assert (
        output.out == ""
        and output.err == "mlbundle: ClaimMismatchError: bundle rejected.\n"
    )


def test_missing_extra_in_subprocess(bundle_path):
    script = """
import sys
class WithoutModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'skops', 'xgboost', 'onnx', 'onnxruntime'}:
            raise ModuleNotFoundError(name=fullname)
sys.meta_path.insert(0, WithoutModels())
from mlbundle.cli import main
sys.exit(main(['verify', sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(bundle_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and result.stdout == ""
    assert (
        result.stderr
        == f"mlbundle: missing optional dependency; install mlbundle[{mlbundle.inspect(bundle_path).manifest.model.library.value}].\n"
    )


@pytest.mark.parametrize("json_output", [False, True])
def test_inspect_never_imports_model_libraries(bundle_path, json_output):
    script = """
import sys
blocked = {'sklearn', 'skops', 'numpy', 'scipy', 'pandas', 'mlschema', 'xgboost', 'onnx', 'onnxruntime'}
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockModels())
from mlbundle.cli import main
assert main(['inspect', *sys.argv[1:]]) == 0
assert not blocked & sys.modules.keys()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(bundle_path),
            *(["--json"] if json_output else []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_terminal_controls_and_display_bounds(tmp_path, capsys):
    label = "label\n\x1b[31m\r\x7f\x9b\u202e\u2028"
    model = DecisionTreeClassifier().fit([[0], [1]], [label, "other"])
    path = tmp_path / "controls.mlbundle"
    metadata = {"\x1b[31m": "x" * 10_000 + "\n\r\x9b"}
    mlbundle.export(model, path, feature_names=[label], metadata=metadata)
    assert main(["inspect", str(path)]) == 0
    output = capsys.readouterr().out
    assert max(map(len, output.splitlines())) < 600
    assert "..." in output and "\\u001b" in output and "\\u202e" in output
    assert all(ord(char) >= 32 and ord(char) < 127 for char in output if char != "\n")
    assert main(["inspect", str(path), "--json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["user_metadata"] == metadata
    assert json.loads(output)["target"]["class_labels"] == [[label, "other"]]
    assert all(32 <= ord(char) < 127 for char in output if char != "\n")


@pytest.mark.parametrize("help_output", [False, True])
def test_closed_output_pipe_is_one_io_error(bundle_path, help_output):
    args = ["--help"] if help_output else ["inspect", str(bundle_path)]
    with subprocess.Popen(
        [_CONSOLE, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        assert process.stdout is not None and process.stderr is not None
        process.stdout.close()
        assert process.wait(timeout=30) == 2
        error = process.stderr.read()
    assert error in {
        "mlbundle: OSError: I/O failed.\n",
        "mlbundle: BrokenPipeError: I/O failed.\n",
    }


@pytest.mark.parametrize("failure", ["usage", "io", "bundle"])
def test_closed_error_pipe_exits_two(failure, tmp_path):
    path = tmp_path / "invalid.mlbundle"
    path.write_bytes(b"invalid bundle")
    args = {
        "usage": ["unknown"],
        "io": ["inspect", str(tmp_path / "missing")],
        "bundle": ["verify", str(path)],
    }[failure]
    with subprocess.Popen(
        [_CONSOLE, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        assert process.stdout is not None and process.stderr is not None
        process.stderr.close()
        assert process.wait(timeout=30) == 2
        assert process.stdout.read() == b""


def test_summary_includes_schema_and_samples(tmp_path, capsys):
    import pandas as pd

    frame = pd.DataFrame({"a": [0, 1]})
    path = tmp_path / "schema.mlbundle"
    mlbundle.export(
        DecisionTreeClassifier().fit(frame, [0, 1]),
        path,
        training_data=frame,
        sample_rows=1,
    )
    assert main(["inspect", str(path)]) == 0
    output = capsys.readouterr().out
    assert "Features (model):" in output
    assert "schema=yes, structure=yes, samples=yes" in output
