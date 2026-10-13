"""Manual publish preflight rejects other refs, existing versions and outages."""

import runpy
from contextlib import nullcontext
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_check.py"


@pytest.fixture
def preflight():
    return runpy.run_path(str(SCRIPT))["check"]


@pytest.mark.parametrize("ref", ["refs/heads/develop", "refs/tags/v0.1.0a3", ""])
def test_other_refs_fail_before_network(preflight, monkeypatch, ref):
    def unexpected(*args, **kwargs):
        raise AssertionError("Network must not be used for other refs")

    monkeypatch.setitem(preflight.__globals__, "urlopen", unexpected)
    with pytest.raises(SystemExit, match="requires refs/heads/main"):
        preflight(ref, {"version": "0.1.0a3"})


@pytest.mark.parametrize("code", [200, 404, 403, 429, 500])
def test_exact_version_response(preflight, monkeypatch, capsys, code):
    def response(url, timeout):
        assert url == "https://pypi.org/pypi/mlbundle/0.1.0a3/json"
        assert timeout == 30
        if code != 200:
            raise HTTPError(url, code, "response", None, None)
        return nullcontext()

    monkeypatch.setitem(preflight.__globals__, "urlopen", response)
    if code == 404:
        preflight("refs/heads/main", {"version": "0.1.0a3"})
        assert "HTTP 404" in capsys.readouterr().out
    else:
        message = "already exists" if code == 200 else f"HTTP {code}"
        with pytest.raises(SystemExit, match=message):
            preflight("refs/heads/main", {"version": "0.1.0a3"})


@pytest.mark.parametrize("error", [URLError("offline"), TimeoutError("timeout")])
def test_network_failure_is_closed(preflight, monkeypatch, error):
    def response(*args, **kwargs):
        raise error

    monkeypatch.setitem(preflight.__globals__, "urlopen", response)
    with pytest.raises(SystemExit, match="PyPI version check failed"):
        preflight("refs/heads/main", {"version": "0.1.0a3"})
