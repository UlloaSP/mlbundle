# Version 1 runtime environment lock

Normative section of [the format contract](format.md). Version 1 is not frozen.

Every bundle contains `pylock.toml`, a [PEP 751](https://peps.python.org/pep-0751/)
lock of the distributions its model needs at prediction time, other than mlbundle
itself. The manifest's `writer_version` pins mlbundle: a runtime installs exactly
the lock plus `mlbundle==writer_version`. The lock is an author claim. mlbundle
parses and validates it, but never installs, imports or executes anything it
names; a runtime that installs it executes the named packages' code.

## Targets

Locks target CPython with the GIL on three platforms, named as
`sysconfig.get_platform()` reports them: `linux-x86_64` and `linux-aarch64` for
hosts with glibc 2.28 or newer (`manylinux_2_28`), and `win-amd64` (64-bit
Windows on x86). macOS, musl and Windows on ARM are not targets. The Python version is the exact
training version (`environment.python_version`), and the training
implementation must be CPython. Free-threaded wheels (`cp3Nt` ABI tags) never
match a target, and `environment.free_threaded` records only how training ran.
Other targets require a new format revision.

## Canonical file

The member is ASCII TOML in exactly the spelling the writer emits; readers
re-encode the parsed lock and require identical bytes. This closes parser
differentials between TOML implementations. The file is:

```toml
lock-version = "1.0"
created-by = "mlbundle"
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'linux' and platform_machine == 'aarch64'",
    "sys_platform == 'win32' and platform_machine == 'AMD64'",
]
requires-python = "==3.14.7"

[[packages]]
name = "numpy"
version = "2.5.4"
wheels = [
    { name = "numpy-2.5.4-cp314-...-manylinux_2_28_aarch64.whl", url = "https://.../numpy-2.5.4-cp314-...-manylinux_2_28_aarch64.whl", size = 15342001, hashes = { sha256 = "..." } },
    { name = "numpy-2.5.4-cp314-...-manylinux_2_28_x86_64.whl", url = "https://.../numpy-2.5.4-cp314-...-manylinux_2_28_x86_64.whl", size = 16710432, hashes = { sha256 = "..." } },
    { name = "numpy-2.5.4-cp314-cp314-win_amd64.whl", url = "https://.../numpy-2.5.4-cp314-cp314-win_amd64.whl", size = 12902112, hashes = { sha256 = "..." } },
]

[[packages]]
name = "nvidia-nccl-cu13"
version = "2.32.3"
marker = "(sys_platform == 'linux' and platform_machine == 'x86_64') or (sys_platform == 'linux' and platform_machine == 'aarch64')"
wheels = [ ... ]
```

Lines end with LF, the file ends with one LF, packages are separated by one
blank line and there are no comments. The only top-level keys are those shown;
each package has `name`, `version`, optional `marker` and `wheels`, in that
order. `marker` appears only for a package locked for some targets: it is that
target's `environments` entry, or for several targets each entry in parentheses,
in target order, joined by ` or `. Each wheel inline table has `name`,
`url`, `size` and `hashes` with only `sha256`, in that order. Index, sdist, VCS,
directory, archive, path, upload-time, extras, dependency groups, attestation
identities and tool tables are rejected.

## Validation

The same code validates the lock in the writer, `inspect` and `load`:

- Packages: 1–512, sorted and unique by PEP 503 normalized name (at most 128
  characters, `[a-z0-9]` runs joined by single hyphens). `mlbundle` is absent.
- Versions: public PEP 440 releases, optional `a`/`b`/`rc`, `.post` and `.dev`
  segments, at most 64 characters. Local versions such as `+cpu` are rejected,
  as no index serves them.
- Wheels: one to three per package, sorted and unique by filename. A filename
  names the package and version. URLs are HTTPS, end with `/` and the filename,
  contain no quotes, backslashes, spaces or controls and are at most 2,048
  characters. Sizes are 1–2^40 bytes; digests are lowercase hexadecimal SHA-256.
- Selection: on each of a package's targets, installers' tag order chooses one
  wheel: platform wheels before pure ones, then the `cp3N` ABI, `abi3` and
  `none`. Linux platform tags are `manylinux_X_Y` up to 2.28 (or the equivalent
  `manylinux1`, `manylinux2010`, `manylinux2014` aliases), Windows uses
  `win_amd64`, and `any` fits every target. The best
  wheel per target must be unique, and every listed wheel must be some target's
  choice. A lock therefore determines the exact file each target installs.
- Binding: `requires-python` is `==` plus the manifest's Python version. The
  lock contains pydantic and the model's library distributions on every target:
  `scikit-learn` and `skops`; `xgboost` or `xgboost-cpu` (not both) and
  `scikit-learn`; `catboost`; or `onnx` and `onnxruntime`. Composite pipelines
  combine their sklearn and leaf requirements.
- Bounds: the member cap is 1 MiB (`max_lock_bytes`). Invalid UTF-8, TOML
  errors, excessive nesting and non-canonical bytes raise
  `InvalidEnvironmentLockError` without echoing values.

Validation proves structure and consistency only. Hashes identify files the
author named; they do not show that the packages are safe or that the author
trained with them.

## Generation

`export` locks the running interpreter with mlbundle's `lock` extra, which
provides uv:

1. Pin pydantic and the model's library distributions at their installed
   versions. Exactly one distribution per requirement group must be installed;
   `xgboost-cpu` satisfies XGBoost. Editable, directory, VCS and local-version
   installs cannot be pinned; wheels installed from a file or URL keep their version.
2. Constrain every other installed distribution to its installed version.
3. For each target, resolve wheels only with
   `uv pip compile --python-version <training> --only-binary :all:` and
   `--python-platform` `x86_64-manylinux_2_28`, `aarch64-manylinux_2_28` or
   `x86_64-pc-windows-msvc`, then keep each package's preferred wheel. Versions
   must agree across targets; a package missing from some targets gets a marker.
4. Require every locked distribution installed on the training machine to have
   the installed version. Distributions needed only on other targets, such as
   NCCL for XGBoost on Linux, take uv's resolution.

Generation uses the index configured for uv on the training machine and fails
with `LockGenerationError` when a target lacks a wheel, a package publishes only
an sdist or the index is unreachable. There is no fallback to other versions.
`export(environment_lock=...)` instead accepts canonical lock bytes or a path,
such as another bundle's `EnvironmentLock.to_bytes()`; the same fidelity check
applies. Locks fix versions and files, not training binaries: a conda or macOS
build that trained a model can differ from the wheels that serve it.

## Consumers

`inspect(path).environment_lock` exposes the validated `EnvironmentLock` without
model libraries: `python_version`, `platforms`, `glibc`, `packages` (name,
version, targets and wheels) and `environment_id`, the SHA-256 of the Python
version and every package's name, version, targets, wheel filenames and digests.
URLs do not change the identity, so a mirror yields the same environment.
`to_bytes()` returns the canonical file for installers such as
`uv pip install -r pylock.toml`.

Installing a lock runs code from every package in it: import-time code, `.pth`
files and native extensions. mlbundle does not decide which packages, versions or
indexes a runtime trusts. Before installing, a runtime applies its own policy to
the complete package set (for example names, version ranges or reviewed wheel
digests), downloads only from indexes it trusts by name, version and digest, and
installs in an isolated environment separate from the process that loads models.

`check_environment(bundle)` runs in that loading process. It requires CPython
with the GIL at the lock's Python version on a target platform (glibc 2.28 or
newer on Linux),
and installed distributions equal to the target's locked packages plus mlbundle
at `writer_version`, with nothing else and no duplicate, editable, directory or
VCS installs. It reads distribution metadata only, imports no model library and
raises `EnvironmentMismatchError` with bounded differences. `load` does not run
this check, so a matching environment remains the caller's decision.
