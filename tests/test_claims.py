"""Canonical claim comparison preserves order, types and numeric identity."""

import pytest
from pydantic import ValidationError

from mlbundle import ClaimMismatchError, Manifest, ModelClaims
from mlbundle.canonical import canonical_dumps
from mlbundle.claims import verify_claims, verify_structure


def test_equal_claims_and_structure(manifest_data):
    expected = Manifest.from_bytes(canonical_dumps(manifest_data)).verified_claims()
    actual = ModelClaims.model_validate(expected.model_dump())
    verify_claims(expected, actual)
    verify_structure(
        {"b": [1.0], "a": {"$float": "nan"}}, {"a": {"$float": "nan"}, "b": [1.0]}
    )
    verify_structure(None, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("estimator_class", "OtherClassifier"),
        ("hyperparameters", {"max_depth": 4}),
        ("feature_names", ["b", "a"]),
        ("feature_dtypes", ["float32", "int64"]),
        ("class_labels", [["positive", 1]]),
        ("class_labels", [[1.0, "positive"]]),
    ],
)
def test_claim_mismatch(manifest_data, field, value):
    expected = Manifest.from_bytes(canonical_dumps(manifest_data)).verified_claims()
    fields = expected.model_dump()
    fields[field] = value
    actual = ModelClaims.model_validate(fields)
    with pytest.raises(ClaimMismatchError, match=field):
        verify_claims(expected, actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_names", ["a"]),
        ("feature_names", ["a", "a"]),
        ("feature_dtypes", ["float32"]),
    ],
)
def test_invalid_extracted_features(manifest_data, field, value):
    claims = (
        Manifest.from_bytes(canonical_dumps(manifest_data))
        .verified_claims()
        .model_dump()
    )
    claims[field] = value
    with pytest.raises(ValidationError):
        ModelClaims.model_validate(claims)


@pytest.mark.parametrize("actual", [{"coef": [0.0]}, {"coef": [1]}, None])
def test_structure_mismatch(actual):
    with pytest.raises(ClaimMismatchError, match="structure"):
        verify_structure({"coef": [1.0]}, actual)
