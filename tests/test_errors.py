"""All contract errors are catchable together and diagnostics are bounded."""

import pytest

import mlbundle


@pytest.mark.parametrize(
    "name,args",
    [
        ("InvalidContainerError", ("Invalid zip.",)),
        ("LimitExceededError", ("Limit exceeded.",)),
        ("UnsupportedFormatVersionError", ()),
        ("InvalidManifestError", ()),
        ("InvalidJSONError", ()),
        ("IntegrityMismatchError", ("model.ubj",)),
        ("ClaimMismatchError", ("n_features",)),
        ("UnsupportedModelError", (("custom.Model",),)),
        ("MissingOptionalDependencyError", ("onnx",)),
        ("AdapterNotImplementedError", ("onnx",)),
    ],
)
def test_hierarchy(name, args):
    error = getattr(mlbundle, name)(*args)
    assert isinstance(error, mlbundle.MLBundleError)
    assert str(error)


def test_untrusted_types_and_sanitized_messages():
    types = ("custom.Model", "x" * 10000, "bad\nname")
    error = mlbundle.UnsupportedModelError(types)
    assert error.untrusted_types == types
    assert "custom.Model" in str(error)
    assert len(str(error)) < 200
    assert "\n" not in str(error)
    assert "unrecognized" in str(mlbundle.ClaimMismatchError("x" * 10000))
    assert "mlbundle[sklearn]" in str(
        mlbundle.MissingOptionalDependencyError("sklearn")
    )
