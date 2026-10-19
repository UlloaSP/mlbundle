"""Run examples/train.py and store everything the landing page shows about it.

The page never hand-writes bundle output: members, hashes, manifest, schema, CLI
text and the tampered-bundle rejection all come from this run. Usage, from landing/:
uv run --project .. python scripts/example.py
"""

import contextlib
import io
import json
import os
import runpy
import tempfile
import zipfile
from pathlib import Path

import mlbundle
from mlbundle import inspect
from mlbundle.cli import main as cli

LANDING = Path(__file__).resolve().parents[1]
SOURCE = LANDING / "examples" / "train.py"
OUTPUT = LANDING / "src" / "data" / "example.json"


def run_cli(*argv: str) -> dict[str, object]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli(list(argv))
    return {
        "exit": code,
        "stdout": out.getvalue().rstrip(),
        "stderr": err.getvalue().rstrip(),
    }


def tamper(path: Path, target: Path) -> None:
    """Rewrite one model claim in manifest.json, keeping every other member intact."""
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(target, "w") as out:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["model"]["hyperparameters"]["max_depth"] = 3
                data = json.dumps(manifest).encode()
            out.writestr(info, data)


def main() -> None:
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as work:
        os.chdir(work)
        try:
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                runpy.run_path(str(SOURCE), run_name="__main__")
            bundle = Path("trained.mlbundle")
            view = inspect(bundle)
            tamper(bundle, Path("edited.mlbundle"))
            result = {
                "version": mlbundle.__version__,
                "code": SOURCE.read_text(encoding="utf-8"),
                "printed": printed.getvalue().rstrip(),
                "bundleBytes": bundle.stat().st_size,
                "members": [
                    {
                        "name": member.value,
                        "size": record.size if record else None,
                        "sha256": record.sha256 if record else None,
                    }
                    for member in view.members
                    for record in [view.manifest.integrity.get(member)]
                ],
                "manifest": view.manifest.model_dump(mode="json", by_alias=True),
                "schema": view.schema,
                "inspect": run_cli("inspect", "trained.mlbundle"),
                "verify": run_cli("verify", "trained.mlbundle"),
                "editedInspect": run_cli("inspect", "edited.mlbundle"),
                "editedVerify": run_cli("verify", "edited.mlbundle"),
            }
        finally:
            os.chdir(previous)
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT.relative_to(LANDING)}")


if __name__ == "__main__":
    main()
