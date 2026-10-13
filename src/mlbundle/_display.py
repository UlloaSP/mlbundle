"""Bounded ASCII display shared by diagnostics and the CLI."""

import json


def display(value: object, *, limit: int = 512) -> str:
    # ASCII JSON escapes terminal controls, bidi controls and line separators.
    rendered = json.dumps(value, ensure_ascii=True)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."
