"""Metadata-only inspection validates bytes without importing model libraries."""

import subprocess
import sys

import pytest

from lock_helpers import sklearn_lock_bytes
from mlbundle import (
    ArchiveLimits,
    InvalidContainerError,
    LimitExceededError,
    MemberName,
    MetadataView,
    UnsupportedFormatVersionError,
    inspect,
)
from mlbundle.writing import write_container


def test_inspect_and_optional_json(manifest_data):
    from mlbundle.canonical import canonical_dumps

    manifest_data["references"]["schema"] = "schema.json"
    schema = {
        "schema_version": "2",
        "source": "training_data",
        "onehot_separator": "__",
        "fields": [
            {"kind": "text", "label": name, "required": True, "mappedTo": name}
            for name in ("a", "b")
        ],
        "reports": [
            {
                "kind": "classifier",
                "label": "Predicted class",
                "mappedTo": 0,
                "labels": ["1", "positive"],
                "showClassProbabilities": True,
            }
        ],
    }
    bundle = write_container(
        manifest_data,
        {
            MemberName.SKOPS: b"model",
            MemberName.SCHEMA: canonical_dumps(schema),
            MemberName.LOCK: sklearn_lock_bytes(),
        },
    )
    view = inspect(bundle)
    assert isinstance(view, MetadataView)
    assert view.members == (
        MemberName.MANIFEST,
        MemberName.SKOPS,
        MemberName.SCHEMA,
        MemberName.LOCK,
    )
    assert view.environment_lock.to_bytes() == sklearn_lock_bytes()
    assert view.manifest.integrity[MemberName.SKOPS].size == 5
    assert not hasattr(view, "native_model")
    assert view.schema == schema and view.samples is None


def test_inspect_no_model_imports(manifest_data, tmp_path):
    from conftest import schema_for_manifest
    from mlbundle.canonical import canonical_dumps

    target = tmp_path / "model.mlbundle"
    write_container(
        manifest_data,
        {
            MemberName.SKOPS: b"model",
            MemberName.SCHEMA: canonical_dumps(schema_for_manifest(manifest_data)),
            MemberName.LOCK: sklearn_lock_bytes(),
        },
        destination=target,
    )
    script = """
import sys
from mlbundle import inspect
view = inspect(sys.argv[1])
assert view.manifest.model.library == "sklearn"
for name in ("sklearn", "skops", "xgboost", "catboost", "onnx", "onnxruntime", "numpy", "pandas", "mlschema"):
    assert not any(key == name or key.startswith(name + ".") for key in sys.modules), name
print("inspect imported no model libraries")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "inspect imported no model libraries"


def test_inspect_rejections(manifest_data, parts):
    from conftest import build_zip

    with pytest.raises(InvalidContainerError):
        inspect(b"not a zip")
    with pytest.raises(LimitExceededError):
        inspect(build_zip(parts), limits=ArchiveLimits(max_archive_bytes=1))
    manifest_data["format_version"] = "2"
    from mlbundle.canonical import canonical_dumps

    with pytest.raises(UnsupportedFormatVersionError):
        inspect(
            build_zip([("manifest.json", canonical_dumps(manifest_data)), *parts[1:]])
        )


def test_load_without_schema_libraries(tmp_path):
    from sklearn.tree import DecisionTreeClassifier

    import mlbundle

    path = tmp_path / "model.mlbundle"
    mlbundle.export(DecisionTreeClassifier().fit([[0], [1]], ["", "positive"]), path)
    script = """
import sys
class WithoutSchema:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'mlschema'}:
            raise ModuleNotFoundError(name=fullname)
sys.meta_path.insert(0, WithoutSchema())
from mlbundle import load
bundle = load(sys.argv[1])
assert bundle.predict_proba([[0], [1]]) == [[1.0, 0.0], [0.0, 1.0]]
assert bundle.schema['reports'][0]['labels'] == ['', 'positive']
assert not {'pandas', 'mlschema'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(path)], check=True, capture_output=True
    )
