"""Small wire-format fixture shared by contract feature tests."""

import pytest

from mlbundle import __version__
from mlbundle.integrity import hash_member


@pytest.fixture
def manifest_data() -> dict:
    return {
        "format_name": "mlbundle",
        "format_version": "1",
        "bundle_id": "f0d87d83-8cf9-4cf5-af08-e1c885156513",
        "created_at": "2026-10-11T10:00:00Z",
        "writer_version": __version__,
        "model": {
            "library": "sklearn",
            "estimator_class": "DecisionTreeClassifier",
            "kind": "classifier",
            "serialization": "skops",
            "hyperparameters": {"max_depth": 3, "random_state": 0},
            "n_features": 2,
            "n_classes": [2],
            "n_outputs": 1,
        },
        "features": {
            "names": ["a", "b"],
            "dtypes": ["float64", None],
            "model_dtypes": None,
            "origin": "model",
        },
        "target": {"class_labels": [[1, "positive"]], "name": "outcome"},
        "environment": {
            "python_version": "3.14.0",
            "python_implementation": "CPython",
            "os": "Windows",
            "os_version": "11",
            "architecture": "AMD64",
            "packages": {"mlbundle": __version__, "scikit-learn": "1.9.1"},
            "onnx": None,
        },
        "integrity": {"model.skops": hash_member(b"native model bytes").model_dump()},
        "references": {"schema": None, "structure": None, "samples": None},
        "user_metadata": {"name": "example", "metrics": {"accuracy": 0.95}},
    }
