"""Strict bounded version text for failure diagnostics, never compatibility gates."""

import re
from importlib.metadata import version

from ._display import display

_VERSION = re.compile(
    r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-z0-9]+(?:[.-][a-z0-9]+)*)?"
)


def safe_version(value: object) -> str | None:
    if isinstance(value, str) and len(value) <= 64 and _VERSION.fullmatch(value):
        return value
    return None


def writer_hint(writer: object) -> str:
    declared = safe_version(writer)
    if declared is None:
        return ""
    reader = version("mlbundle")
    if declared == reader:
        return ""
    return (
        f" Bundle was written by mlbundle {display(declared)}; this reader is "
        f"{display(reader)}. Format 1 is not frozen; the model must be re-exported."
    )
