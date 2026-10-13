"""Small inspection and verification CLI with bounded, terminal-safe output."""

import argparse
import io
import json
import sys
import warnings
from collections.abc import Sequence
from typing import TextIO

from . import inspect, load
from .contract import MemberName
from .errors import MissingOptionalDependencyError, MLBundleError
from .results import LoadedBundle, MetadataView

_DISPLAY_LIMIT = 512
_NOTICE = (
    "inspect validates metadata and integrity; it cannot attest that metadata "
    "matches the model. Only verify loads the model and checks that agreement."
)


class _Parser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        # argparse's default echoes arbitrary argv and prints several lines.
        self.exit(_write_error("mlbundle: invalid arguments; use --help.", 2))

    def print_help(self, file: TextIO | None = None) -> None:
        if file is None or file is sys.stdout:
            _write_output(self.format_help().rstrip("\n"))
        else:
            super().print_help(file)


def _display(value: object) -> str:
    # ASCII JSON escapes terminal controls, bidi controls and line separators.
    rendered = json.dumps(value, ensure_ascii=True)
    if len(rendered) > _DISPLAY_LIMIT:
        return rendered[: _DISPLAY_LIMIT - 3] + "..."
    return rendered


def _summary(view: MetadataView) -> str:
    manifest = view.manifest
    model = manifest.model
    environment = manifest.environment
    lines = [
        "Metadata and integrity validated; model claims NOT verified.",
        f"Format: mlbundle {manifest.format_version}",
        f"Bundle id: {manifest.bundle_id}",
        f"Created: {manifest.created_at.isoformat()}",
        f"Writer: {_display(manifest.writer_version)}",
        f"Model: {model.library.value} {_display(model.estimator_class)} ({model.kind.value})",
        f"Counts: features={model.n_features}, outputs={model.n_outputs}, classes={_display(model.n_classes)}",
        f"Features ({manifest.features.origin.value}): {_display(manifest.features.names)}",
        f"Class labels (output-major): {_display(manifest.target.class_labels)}",
        f"Target names (output-major): {_display(manifest.target.names)}",
    ]
    lines.extend(_pipeline_summary(view))
    lines.append("Members:")
    for member in view.members:
        record = manifest.integrity.get(member)
        if record is None:
            lines.append(f"  {member.value}: validated (no self-integrity record)")
        else:
            lines.append(
                f"  {member.value}: {record.size} bytes sha256={record.sha256}"
            )
    lines.extend(
        [
            "Training: "
            + _display(
                {
                    "python": environment.python_version,
                    "implementation": environment.python_implementation,
                    "os": environment.os,
                    "os_version": environment.os_version,
                    "architecture": environment.architecture,
                }
            ),
            f"Packages: {_display(environment.packages)}",
            f"ONNX environment: {_display(None if environment.onnx is None else environment.onnx.model_dump(mode='json'))}",
            "Present: "
            + ", ".join(
                f"{name}={'yes' if MemberName(name + '.json') in view.members else 'no'}"
                for name in ("schema", "structure", "samples")
            ),
            f"User metadata: {_display(manifest.user_metadata)}",
        ]
    )
    return "\n".join(lines)


def _pipeline_summary(view: MetadataView | LoadedBundle) -> list[str]:
    model = view.manifest.model
    recipe = model.pipeline
    if recipe is None:
        return []
    leaf = recipe.estimator
    boundary = recipe.transformed_features
    return [
        f"Steps: {_display(recipe.step_names)}",
        f"Estimator: {leaf.library.value} {_display(leaf.claims.estimator_class)} ({leaf.serialization.value})",
        f"Boundary: input={model.n_features}, transformed={boundary.n_features}",
        f"Transformed features ({boundary.origin}): {_display(boundary.names)}",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="mlbundle", description=_NOTICE)
    commands = parser.add_subparsers(dest="command", required=True)
    inspection = commands.add_parser(
        "inspect", help="validate metadata without loading a model", description=_NOTICE
    )
    inspection.add_argument("file", help="bundle path")
    inspection.add_argument(
        "--json",
        action="store_true",
        help="print the full validated manifest as escaped JSON",
    )
    verification = commands.add_parser(
        "verify",
        help="load and verify native model claims and structure",
        description=_NOTICE + " verify requires the matching library extra.",
    )
    verification.add_argument("file", help="bundle path")
    parser.epilog = (
        "Exit codes: 0 success; 1 bundle/dependency rejection; 2 usage or I/O error."
    )
    return parser


def _write_output(output: str) -> None:
    try:
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
    except OSError:
        # Prevent a second failed flush, with unbounded text/exit 120, at shutdown.
        sys.stdout = io.StringIO()
        raise


def _write_error(message: str, code: int) -> int:
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except OSError:
        sys.stderr = io.StringIO()
        return 2
    return code


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "inspect":
            view = inspect(args.file)
            output = (
                json.dumps(
                    view.manifest.model_dump(mode="json", by_alias=True),
                    ensure_ascii=True,
                    indent=2,
                )
                if args.json
                else _summary(view)
            )
        else:
            # Version warnings may include native content; CLI errors are bounded.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bundle = load(args.file)
            output = (
                f"Valid bundle: {bundle.library.value} {_display(bundle.class_name)}; "
                "verified format, integrity, schema/samples and native model claims/structure."
            )
            details = _pipeline_summary(bundle)
            if details:
                output += "\n" + "\n".join(details)
        _write_output(output)
    except MissingOptionalDependencyError as error:
        return _write_error(
            f"mlbundle: missing optional dependency; install mlbundle[{error.extra}].",
            1,
        )
    except MLBundleError as error:
        # Report the failure category, never exception text derived from a bundle.
        return _write_error(f"mlbundle: {type(error).__name__}: bundle rejected.", 1)
    except OSError as error:
        return _write_error(f"mlbundle: {type(error).__name__}: I/O failed.", 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
