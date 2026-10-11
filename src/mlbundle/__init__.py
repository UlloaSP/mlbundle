"""Pickle-free model bundles."""

from importlib.metadata import version as _version

__version__ = _version("mlbundle")

__all__ = ["__version__"]
