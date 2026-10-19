"""Execute the README quick start and check the links PyPI renders."""

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(
    r"```(\w+)\n(.*?)\n```", README.read_text(encoding="utf-8"), re.DOTALL
)
PYTHON = [code for language, code in BLOCKS if language == "python"]
LINKS = re.findall(
    r"\]\(([^)\s]+)\)|(?:href|src|srcset)=\"([^\"]+)\"",
    README.read_text(encoding="utf-8"),
)


def test_links_are_absolute():
    # PyPI renders the README without the repository, so relative links break.
    targets = [markdown or html for markdown, html in LINKS]
    assert targets
    assert [url for url in targets if not url.startswith("https://")] == []


@pytest.mark.parametrize("code", PYTHON)
def test_python_example(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scope = {}
    exec(compile(code, str(README), "exec"), scope)
    assert scope["metadata"].samples is None
    assert scope["bundle"].feature_names == ("temperature", "humidity")
    assert scope["result"].predicted_label == "cold"
    assert scope["result"].probabilities == (1.0, 0.0)


def test_console_examples(tmp_path, monkeypatch):
    assert PYTHON
    assert all(language in {"python", "shell", "bash"} for language, _ in BLOCKS)
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
