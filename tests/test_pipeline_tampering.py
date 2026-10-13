"""Rehashed attacker recipes, component state and native structure still fail."""

import copy

import numpy as np
import pytest
import skops.io as sio

import mlbundle
from conftest import build_zip
from mlbundle._sklearn import transformers
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.writing import write_container
from pipeline_helpers import parts, pipeline


@pytest.fixture
def composite(tmp_path):
    model, _ = pipeline()
    path = tmp_path / "composite.mlbundle"
    mlbundle.export(model, path)
    return parts(path)


def changed_record(parts, path, value):
    record = copy.deepcopy(parts)
    current = record
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    return record


@pytest.mark.parametrize(
    "path,value",
    [
        (("model", "pipeline", "unknown"), 1),
        (("model", "pipeline", "step_names"), ["same", "same"]),
        (("model", "pipeline", "step_names"), ["x__y", "est"]),
        (("model", "pipeline", "step_names"), ["memory", "est"]),
        (("model", "pipeline", "step_names"), ["x" * 257, "est"]),
        (("model", "pipeline", "transformers"), None),
        (("model", "pipeline", "transformers", "state_sha256"), None),
        (("model", "pipeline", "transformers", "member"), "model.ubj"),
        (("model", "pipeline", "estimator", "library"), "sklearn"),
        (("model", "pipeline", "estimator", "serialization"), "cbm"),
        (("model", "pipeline", "estimator", "member"), "model.cbm"),
        (("model", "pipeline", "estimator", "claims", "estimator_class"), "eval"),
        (("model", "pipeline", "transformed_features", "n_features"), True),
        (("model", "pipeline", "transformed_features", "names"), ["a", "a", "b"]),
        (("model", "pipeline", "transformed_features", "origin"), "declared"),
        (("model", "hyperparameters", "memory"), "cache"),
        (("model", "hyperparameters", "verbose"), 1),
        (("model", "state_sha256"), "a" * 64),
        (("model", "library"), "xgboost"),
        (("model", "serialization"), "skops"),
        (("model", "pipeline"), None),
    ],
)
def test_strict_recipe(composite, path, value):
    record, members = composite
    with pytest.raises(mlbundle.InvalidManifestError):
        mlbundle.inspect(write_container(changed_record(record, path, value), members))


@pytest.mark.parametrize(
    "path,value,claim",
    [
        (("model", "pipeline", "step_names", 0), "renamed", "pipeline.transformers"),
        (
            ("model", "pipeline", "transformers", "state_sha256"),
            "a" * 64,
            "pipeline.transformers",
        ),
        (
            ("model", "pipeline", "transformers", "hyperparameters", "scale__copy"),
            False,
            "pipeline.transformers",
        ),
        (
            ("model", "pipeline", "transformed_features", "names"),
            ["x", "y", "z"],
            "pipeline.transformed_features",
        ),
        (
            (
                "model",
                "pipeline",
                "estimator",
                "claims",
                "hyperparameters",
                "prediction",
                "importance_type",
            ),
            "weight",
            "pipeline.estimator.hyperparameters",
        ),
        (
            ("model", "pipeline", "estimator", "claims", "feature_names"),
            ["x", "y", "z"],
            "pipeline.estimator.feature_names",
        ),
        (
            ("model", "pipeline", "estimator", "claims", "feature_dtypes"),
            ["float", "float", "float"],
            "pipeline.estimator.feature_dtypes",
        ),
        (("model", "hyperparameters", "verbose"), True, None),
    ],
)
def test_recomputed_component_claims(composite, path, value, claim):
    record, members = composite
    data = write_container(changed_record(record, path, value), members)
    if claim is None:
        # Outer options are explicitly declarations; they do not restore model state.
        assert mlbundle.load(data).native_model.verbose is True
    else:
        with pytest.raises(mlbundle.ClaimMismatchError) as error:
            mlbundle.load(data)
        assert error.value.claim == claim


@pytest.mark.parametrize(
    "member_set",
    [
        {"model.skops", "model.onnx"},
        {"model.ubj", "model.cbm"},
        {"model.skops", "model.ubj", "model.cbm"},
        set(),
    ],
)
def test_invalid_member_candidates(composite, member_set):
    record, _ = composite
    with pytest.raises(
        mlbundle.InvalidContainerError, match="Invalid model member combination"
    ):
        mlbundle.inspect(
            build_zip(
                [
                    ("manifest.json", canonical_dumps(record)),
                    *[(name, b"native") for name in member_set],
                ]
            )
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing_skops",
        "missing_native",
        "extra_native",
        "pair_without_recipe",
        "recipe_without_pair",
    ],
)
def test_member_recipe_disagreement(composite, change):
    record, members = composite
    record, members = copy.deepcopy(record), dict(members)
    if change == "missing_skops":
        members.pop(mlbundle.MemberName.SKOPS)
    elif change == "missing_native":
        members.pop(mlbundle.MemberName.UBJSON)
    elif change == "extra_native":
        members[mlbundle.MemberName.CBM] = b"native"
    elif change == "pair_without_recipe":
        record["model"]["pipeline"] = None
        record["model"]["serialization"] = "skops"
    else:
        record["model"]["pipeline"]["transformers"] = None
        record["model"]["pipeline"]["step_names"] = ["est"]
    with pytest.raises((mlbundle.InvalidContainerError, mlbundle.InvalidManifestError)):
        mlbundle.inspect(write_container(record, members))


@pytest.mark.parametrize("alteration", ["state", "params", "names", "width"])
def test_rehashed_transformer_payload(composite, alteration):
    record, members = composite
    restored = transformers.Adapter().deserialize(members[mlbundle.MemberName.SKOPS])
    step = restored.steps[0][1]
    if alteration == "state":
        step.mean_[0] += 1
    elif alteration == "params":
        step.copy = False
    elif alteration == "names":
        restored.steps[0] = ("renamed", step)
    else:
        step.scale_ = np.ones(2)
    members = {**members, mlbundle.MemberName.SKOPS: sio.dumps(restored)}
    with pytest.raises((mlbundle.ClaimMismatchError, mlbundle.InvalidModelError)):
        mlbundle.load(write_container(record, members))


@pytest.mark.parametrize("alteration", ["tree", "boundary", "null", "extra"])
def test_rehashed_structure(composite, alteration):
    record, members = composite
    structure = canonical_loads(members[mlbundle.MemberName.STRUCTURE])
    if alteration == "tree":
        structure["trees"][0]["nodeid"] += 1
    elif alteration == "boundary":
        structure["pipeline"]["features"]["names"][0] = "changed"
    elif alteration == "extra":
        structure["invented"] = True
    else:
        structure = None
    members = {**members, mlbundle.MemberName.STRUCTURE: canonical_dumps(structure)}
    with pytest.raises(mlbundle.ClaimMismatchError, match="structure"):
        mlbundle.load(write_container(record, members))


def test_labels_counts_missing_structure_and_unknown_version(composite):
    record, members = composite
    changed = copy.deepcopy(record)
    changed["target"]["class_labels"] = changed["model"]["pipeline"]["estimator"][
        "claims"
    ]["class_labels"] = [["no", "yes"]]
    with pytest.raises(mlbundle.ClaimMismatchError, match="class_labels"):
        mlbundle.load(write_container(changed, members))
    changed = copy.deepcopy(record)
    changed["references"]["structure"] = None
    with pytest.raises(mlbundle.InvalidManifestError):
        write_container(
            changed,
            {
                name: value
                for name, value in members.items()
                if name != mlbundle.MemberName.STRUCTURE
            },
        )
    changed["format_version"] = "unknown"
    with pytest.raises(mlbundle.UnsupportedFormatVersionError):
        write_container(changed, members)


@pytest.mark.parametrize("library", ["xgboost", "catboost"])
def test_native_preflight_before_native_loader(library, tmp_path, monkeypatch):
    model, _ = pipeline(library)
    path = tmp_path / "native.mlbundle"
    mlbundle.export(model, path)
    record, members = parts(path)
    final_type = type(model.steps[-1][1])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native parser was called")

    monkeypatch.setattr(final_type, "load_model", forbidden)
    native_member = (
        mlbundle.MemberName.UBJSON if library == "xgboost" else mlbundle.MemberName.CBM
    )
    members[native_member] = b"not a native payload"
    with pytest.raises(mlbundle.InvalidModelError):
        mlbundle.load(write_container(record, members))
