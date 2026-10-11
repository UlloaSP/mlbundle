"""Smoke test for the installed package."""

from importlib.metadata import version

import mlbundle


def test_package_version() -> None:
    assert mlbundle.__version__ == version("mlbundle")
    assert mlbundle.__all__ == ["__version__"]
