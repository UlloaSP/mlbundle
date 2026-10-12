"""Strict manifest acceptance and rejection, including cross-field invariants."""

import pytest
from pydantic import ValidationError

from mlbundle import (
    FeatureOrigin,
    InvalidManifestError,
    LimitExceededError,
    Manifest,
    UnsupportedFormatVersionError,
)
from mlbundle.canonical import canonical_dumps
from mlbundle.integrity import hash_member


def read(data: dict) -> Manifest:
    return Manifest.from_bytes(canonical_dumps(data))


def test_round_trip_and_claim_projection(manifest_data):
    manifest = read(manifest_data)
    assert Manifest.from_bytes(manifest.to_bytes()) == manifest
    assert manifest.to_bytes() == canonical_dumps(manifest_data)
    claims = manifest.verified_claims()
    assert claims.feature_names == ["a", "b"]
    assert claims.class_labels == [[1, "positive"]]
    assert type(claims.class_labels[0][0]) is int
    with pytest.raises(ValidationError):
        manifest.writer_version = "new"


def test_utc_timestamp_and_empty_original_class_label(manifest_data):
    manifest_data["created_at"] = "2026-10-11T10:00:00.1234+00:00"
    manifest_data["target"]["class_labels"] = [["", "positive"]]
    manifest = read(manifest_data)
    assert manifest.target.class_labels == [["", "positive"]]
    assert manifest.created_at.microsecond == 123400
    assert b"2026-10-11T10:00:00.123400Z" in manifest.to_bytes()


@pytest.mark.parametrize(
    "section", [None, "model", "features", "target", "environment", "references"]
)
def test_unknown_fields(manifest_data, section):
    (manifest_data if section is None else manifest_data[section])["unexpected"] = (
        "x" * 1000
    )
    with pytest.raises(InvalidManifestError) as error:
        read(manifest_data)
    assert "x" * 100 not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_features", "2"),
        ("n_features", True),
        ("n_features", 2.0),
        ("n_classes", ["2"]),
        ("n_outputs", 0),
        ("hyperparameters", []),
        ("estimator_class", "x" * 257),
        ("kind", "clusterer"),
        ("library", "pytorch"),
        ("serialization", "unknown"),
        ("serialization", "onnx"),
        ("state_sha256", "short"),
        ("state_sha256", True),
    ],
)
def test_bad_model_fields(manifest_data, field, value):
    manifest_data["model"][field] = value
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


@pytest.mark.parametrize("version", ["2", "0", "1.0"])
def test_unknown_version(manifest_data, version):
    manifest_data["format_version"] = version
    with pytest.raises(UnsupportedFormatVersionError):
        read(manifest_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_name", "other"),
        ("format_version", 1),
        ("bundle_id", "invalid"),
        ("created_at", "2026-10-11T10:00:00"),
        ("created_at", "2026-10-11T10:00:00+02:00"),
        ("created_at", "0"),
        ("created_at", "1679616000.123"),
        ("created_at", "2026-10-11T10:00:00.1234567Z"),
        ("writer_version", 1),
    ],
)
def test_bad_identity(manifest_data, field, value):
    manifest_data[field] = value
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


@pytest.mark.parametrize(
    "name",
    [
        "../model.skops",
        "dir/model.skops",
        "dir\\model.skops",
        "unknown",
        "manifest.json",
        "model.onnx",
    ],
)
def test_bad_member_names(manifest_data, name):
    manifest_data["integrity"] = {name: hash_member(b"data").model_dump()}
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("names", ["a", "a"]),
        ("names", ["a"]),
        ("names", ["a", 1]),
        ("dtypes", ["float64"]),
        ("model_dtypes", ["float32"]),
        ("origin", "unknown"),
        ("origin", "generated"),
    ],
)
def test_bad_features(manifest_data, field, value):
    manifest_data["features"][field] = value
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


@pytest.mark.parametrize(
    "labels", [None, [[1]], [[1, 1]], [[1, 2], [3, 4]], [[None, "x"]]]
)
def test_bad_labels(manifest_data, labels):
    manifest_data["target"]["class_labels"] = labels
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


def test_multioutput_and_nonmodel_features(manifest_data):
    manifest_data["model"].update(n_outputs=2, n_classes=[2, 3])
    manifest_data["target"]["class_labels"] = [[False, True], [0.0, 1.0, "other"]]
    manifest_data["features"].update(
        names=["feature_1", "feature_2"], origin="generated"
    )
    manifest = read(manifest_data)
    assert manifest.verified_claims().feature_names is None
    assert type(manifest.target.class_labels[1][0]) is float
    assert manifest.features.origin == FeatureOrigin.GENERATED
    manifest_data["features"]["origin"] = "dataframe"
    assert read(manifest_data).verified_claims().feature_names is None


def test_regressor(manifest_data):
    manifest_data["model"].update(kind="regressor", n_classes=None)
    manifest_data["target"]["class_labels"] = None
    assert read(manifest_data).verified_claims().class_labels is None
    manifest_data["target"]["class_labels"] = [[0, 1]]
    with pytest.raises(InvalidManifestError):
        read(manifest_data)
    manifest_data["model"]["n_classes"] = [2]
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


def test_invalid_class_counts(manifest_data):
    manifest_data["model"]["n_classes"] = None
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


@pytest.mark.parametrize(
    "section,field,size",
    [("user_metadata", "large", 65536), ("model", "hyperparameters", 1024 * 1024)],
)
def test_bounded_metadata(manifest_data, section, field, size):
    if section == "model":
        manifest_data[section][field] = {"large": "x" * size}
    else:
        manifest_data[section][field] = "x" * size
    with pytest.raises(LimitExceededError):
        read(manifest_data)


def test_metadata_depth(manifest_data):
    nested = {}
    for _ in range(9):
        nested = {"nested": nested}
    manifest_data["user_metadata"] = nested
    with pytest.raises(LimitExceededError):
        read(manifest_data)


def test_top_level_metadata_count(manifest_data):
    manifest_data["user_metadata"] = {str(index): 0 for index in range(1025)}
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


def test_references(manifest_data):
    for name in ("schema", "structure", "samples"):
        member = f"{name}.json"
        manifest_data["references"][name] = member
        manifest_data["integrity"][member] = hash_member(b"{}").model_dump()
    manifest = read(manifest_data)
    assert manifest.references.schema_ == "schema.json"
    manifest_data["integrity"].pop("schema.json")
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


def test_member_cap_and_bad_integrity(manifest_data):
    manifest_data["references"]["structure"] = "structure.json"
    manifest_data["integrity"]["structure.json"] = {
        "size": 256 * 1024**2 + 1,
        "sha256": "a" * 64,
    }
    with pytest.raises(InvalidManifestError):
        read(manifest_data)
    manifest_data["integrity"]["model.skops"]["sha256"] = "bad"
    with pytest.raises(InvalidManifestError):
        read(manifest_data)


def test_environment_invariants_and_onnx(manifest_data):
    manifest_data["environment"]["packages"]["mlbundle"] = "different"
    with pytest.raises(InvalidManifestError):
        read(manifest_data)
    manifest_data["environment"]["packages"]["mlbundle"] = manifest_data[
        "writer_version"
    ]
    manifest_data["model"].update(library="onnx", serialization="onnx")
    manifest_data["integrity"] = {"model.onnx": hash_member(b"onnx").model_dump()}
    with pytest.raises(InvalidManifestError):
        read(manifest_data)
    manifest_data["environment"]["onnx"] = {
        "opsets": [{"domain": "", "version": 21}],
        "ir_version": 10,
        "producer_name": "producer",
        "producer_version": "1",
    }
    assert read(manifest_data).verified_claims().onnx.ir_version == 10


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}', b"NaN", b"[]", b"{}", b"\xff"])
def test_invalid_manifest_json(data):
    with pytest.raises(InvalidManifestError):
        Manifest.from_bytes(data)
