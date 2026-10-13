"""Diagnose sklearn failures after closed loading has rejected the model."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

from ._display import display
from ._versions import safe_version
from .errors import EnvironmentMismatchError, InvalidModelError


@contextmanager
def sklearn_environment(declared_version: object) -> Iterator[None]:
    try:
        yield
    except InvalidModelError:
        declared = safe_version(declared_version)
        try:
            installed = safe_version(version("scikit-learn"))
        except PackageNotFoundError:
            installed = None
        if declared is None or installed is None or declared == installed:
            raise
        raise EnvironmentMismatchError(
            f"Bundle declares scikit-learn {display(declared)}; installed version is "
            f"{display(installed)}. The bundle must be re-exported with scikit-learn "
            f"{display(installed)} or loaded where scikit-learn {display(declared)} is installed."
        ) from None
