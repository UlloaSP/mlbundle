"""Execution harness for the mlbundle playground. Runs inside Pyodide only."""

import errno
import json
import linecache
import os
import shutil
import sys
import traceback
from importlib import import_module
from pathlib import Path
from types import TracebackType

# This module exists only in the Pyodide runtime exercised by test:runtime.
from pyodide.code import eval_code_async  # pyright: ignore[reportMissingImports]

FILENAME = "<playground>"
WORK_ROOT = Path("/playground")
# session.ts writes the lock that scripts/lock.py generated for this runtime.
LOCK = Path("/lock/pylock.toml")
MAX_BUNDLES = 8
MIB = 1024 * 1024
# Lowered with mlbundle's own ArchiveLimits: a browser tab is not a model server.
LIMITS_MIB = {
    "max_archive_bytes": 64,
    "max_compressed_bytes": 64,
    "max_member_bytes": 64,
    "max_total_bytes": 128,
}
_runs = 0


def _no_hard_links(*_args: object, **_kwargs: object) -> None:
    raise OSError(errno.ENOSYS, "Pyodide's file system has no hard links")


def runtime_versions(source_root: str) -> str:
    """Prepare the interpreter, import the repository's mlbundle and report versions."""
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    # Pyodide omits os.link. mlbundle then uses its documented exclusive-create
    # publication, exactly as on other file systems without hard links.
    if not hasattr(os, "link"):
        os.link = _no_hard_links  # pyright: ignore[reportAttributeAccessIssue]
    from importlib.metadata import version  # noqa: PLC0415

    import_module("mlbundle")
    names = ("mlbundle", "mlschema", "scikit-learn", "skops", "pandas", "numpy")
    return json.dumps(
        {"python": sys.version.split()[0]} | {name: version(name) for name in names}
    )


def _limits():
    from mlbundle import ArchiveLimits  # noqa: PLC0415

    return ArchiveLimits(**{key: value * MIB for key, value in LIMITS_MIB.items()})


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _metadata(view) -> dict:
    manifest = json.loads(view.manifest.to_bytes())
    integrity = manifest.get("integrity", {})
    members = [
        {"name": str(name), **integrity.get(str(name), {})} for name in view.members
    ]
    lock = view.environment_lock
    return {
        "manifest": manifest,
        "members": members,
        "schema": view.schema,
        "samples": view.samples,
        "environment": {
            "pythonVersion": lock.python_version,
            "platforms": list(lock.platforms),
            "environmentId": lock.environment_id,
            "packages": [[p.name, p.version, list(p.platforms)] for p in lock.packages],
        },
    }


def _structure(structure: object) -> object:
    if structure is None:
        return None
    from mlbundle.canonical import canonical_dumps  # noqa: PLC0415

    # Canonical JSON keeps non-finite numbers as {"$float": ...} tags.
    return json.loads(canonical_dumps(structure, max_bytes=256 * MIB, max_depth=64))


def describe_bundle(path: str, *, verify: bool) -> dict:
    """Inspect metadata; when verify is set, also load and recompute model claims.

    Uploaded files are never verified here: verification deserializes the model.
    """
    from mlbundle import inspect, load  # noqa: PLC0415

    target = Path(path)
    report: dict = {"fileName": target.name, "sizeBytes": target.stat().st_size}
    try:
        report |= _metadata(inspect(target, limits=_limits()))
    except Exception as error:  # Every rejection is shown verbatim.
        return report | {"status": "rejected", "error": _error_text(error)}
    if not verify:
        return report | {"status": "inspected", "structure": None}
    try:
        loaded = load(target, limits=_limits())
    except Exception as error:
        return report | {"status": "unverified", "error": _error_text(error)}
    return report | {"status": "verified", "structure": _structure(loaded.structure)}


def inspect_upload(path: str) -> str:
    """Metadata-only inspection of a file chosen by the visitor."""
    try:
        return json.dumps(describe_bundle(path, verify=False), allow_nan=False)
    finally:
        Path(path).unlink(missing_ok=True)


def _user_traceback(error: BaseException) -> TracebackType | None:
    tb = error.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != FILENAME:
        tb = tb.tb_next
    return tb


def _format_error(error: BaseException) -> str:
    lines = traceback.format_exception(type(error), error, _user_traceback(error))
    return "".join(lines).rstrip()


def _outcome(value: object) -> dict:
    if value is None:
        return {"kind": "none"}
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return {
            "kind": "repr",
            "typeName": type(value).__qualname__,
            "text": repr(value),
        }
    return {"kind": "json", "text": text}


def _bundles(directory: Path) -> tuple[list[dict], int]:
    found = sorted(p for p in directory.glob("*.mlbundle") if p.is_file())
    reports = [describe_bundle(str(p), verify=True) for p in found[:MAX_BUNDLES]]
    for report, path in zip(reports, found, strict=False):
        report["path"] = str(path)
    return reports, max(0, len(found) - MAX_BUNDLES)


async def run(code: str) -> str:
    """Run code in a fresh namespace and an empty working directory.

    Every *.mlbundle the code writes into that directory is inspected and then
    verified with mlbundle.load, because the visitor's own code produced it.
    """
    global _runs  # noqa: PLW0603
    _runs += 1
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    directory = WORK_ROOT / f"run-{_runs}"
    directory.mkdir(parents=True)
    shutil.copyfile(LOCK, directory / "pylock.toml")
    os.chdir(directory)
    linecache.cache[FILENAME] = (len(code), None, code.splitlines(True), FILENAME)
    namespace = {"__name__": "__main__"}
    try:
        outcome = _outcome(await eval_code_async(code, namespace, filename=FILENAME))
    except BaseException as error:
        outcome = {"kind": "error", "text": _format_error(error)}
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.chdir("/")
    bundles, skipped = _bundles(directory)
    return json.dumps(
        {"outcome": outcome, "bundles": bundles, "skipped": skipped}, allow_nan=False
    )
