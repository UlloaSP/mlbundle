"""Execute the actual fenced README examples, including the console commands."""

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from onnx_helpers import softmax

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(
    r"```(\w+)\n(.*?)\n```", README.read_text(encoding="utf-8"), re.DOTALL
)
PYTHON = [code for language, code in BLOCKS if language == "python"]


@pytest.mark.parametrize(
    "code",
    PYTHON,
    ids=[
        "sklearn",
        "records",
        "xgboost",
        "catboost",
        "composite",
        "onnx",
        "schema-samples",
    ],
)
def test_python_example(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if 'record = {"0"' in code:
        exec(compile(PYTHON[0], str(README), "exec"), {})
    if 'Path("trained.onnx")' in code:
        # The user's prerequisite file: a self-contained, two-column classifier.
        Path("trained.onnx").write_bytes(softmax().SerializeToString())
    exec(compile(code, str(README), "exec"), {})


def test_console_examples(tmp_path, monkeypatch):
    assert len(PYTHON) == 7
    assert all(language in {"python", "shell"} for language, _ in BLOCKS)
    monkeypatch.chdir(tmp_path)
    exec(compile(PYTHON[0], str(README), "exec"), {})
    executable = Path(sys.executable).with_name(
        "mlbundle.exe" if sys.platform == "win32" else "mlbundle"
    )
    for language, code in BLOCKS:
        if language != "shell":
            continue
        for line in code.splitlines():
            command, *args = shlex.split(line)
            assert command == "mlbundle"
            result = subprocess.run(
                [executable, *args], capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout and not result.stderr
