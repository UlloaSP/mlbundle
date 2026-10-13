"""Output-major target declarations, exclusion and hostile reader records."""

import inspect

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mlbundle
from conftest import align_schema
from mlbundle.canonical import canonical_dumps
from mlbundle.cli import main
from mlbundle.container import read_container
from mlbundle.writing import write_container


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("training", [False, True])
def test_target_roundtrip_exclusion_and_reports(
    kind, outputs, training, tmp_path, capsys
):
    features = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    targets = pd.DataFrame({"first": [0, 1], "second": [10, 20]})
    names = list(targets.columns[:outputs])
    y = targets[names].to_numpy()
    model_type = (
        DecisionTreeClassifier if kind == "classifier" else DecisionTreeRegressor
    )
    model = model_type().fit(features, y[:, 0] if outputs == 1 else y)
    frame = features.assign(**targets.to_dict("list"))[["second", "a", "first", "b"]]
    if outputs == 1:
        frame = frame.drop(columns="second")
    path = tmp_path / "targets.mlbundle"
    view = mlbundle.export(
        model,
        path,
        target_names=tuple(names),
        training_data=frame if training else None,
        sample_rows=2 if training else None,
    )
    loaded = mlbundle.load(path)
    assert (
        view.manifest.target.names
        == mlbundle.inspect(path).manifest.target.names
        == loaded.manifest.target.names
        == names
    )
    assert loaded.feature_names == ("a", "b")
    assert loaded.schema == view.schema
    assert loaded.samples == (features.to_dict("records") if training else None)
    label = "Predicted class" if kind == "classifier" else "Predicted value"
    assert [report["label"] for report in loaded.schema["reports"]] == [
        label if outputs == 1 else f"{label} {index}" for index in range(outputs)
    ]
    assert [report["mappedTo"] for report in loaded.schema["reports"]] == list(
        range(outputs)
    )
    actual = (
        loaded.predict_proba(features)
        if kind == "classifier"
        else loaded.predict(features)
    )
    expected = (
        model.predict_proba(features)
        if kind == "classifier"
        else model.predict(features)
    )
    assert actual == (
        [value.tolist() for value in expected]
        if outputs > 1 and kind == "classifier"
        else expected.tolist()
    )
    assert main(["inspect", str(path)]) == 0
    assert (
        "Target names (output-major): " + str(names).replace("'", '"')
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "names,outputs,match",
    [
        ("target", 1, "list"),
        (b"target", 1, "list"),
        ([], 1, "per output"),
        (["a", "b"], 1, "per output"),
        (["a"], 2, "per output"),
        (["same", "same"], 2, "unique"),
        ([""], 1, "nonempty"),
        (["x" * 1025], 1, "bounded"),
        ([0], 1, "bounded"),
        ([True], 1, "bounded"),
    ],
)
def test_export_target_errors_do_not_publish(names, outputs, match, tmp_path):
    model = DecisionTreeClassifier().fit(
        [[0], [1]], [0, 1] if outputs == 1 else [[0, 10], [1, 20]]
    )
    path = tmp_path / "never.mlbundle"
    with pytest.raises(mlbundle.InvalidSchemaError, match=match) as error:
        mlbundle.export(model, path, target_names=names)
    assert len(str(error.value)) < 160 and not path.exists()


def test_missing_training_target_and_removed_alias(tmp_path):
    frame = pd.DataFrame({"a": [0, 1], "first": [0, 1]})
    model = DecisionTreeClassifier().fit(frame[["a"]], [0, 1])
    path = tmp_path / "never.mlbundle"
    with pytest.raises(mlbundle.InvalidSchemaError, match="training column"):
        mlbundle.export(model, path, training_data=frame, target_names=["missing"])
    assert "target_name" not in inspect.signature(mlbundle.export).parameters
    with pytest.raises(TypeError, match="target_name"):
        mlbundle.export(model, path, target_name="first")
    assert not path.exists()


@pytest.mark.parametrize("origin", ["model", "declared", "generated"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_target_feature_overlap_preserves_destination(origin, overwrite, tmp_path):
    frame = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    model = DecisionTreeClassifier().fit(
        frame if origin == "model" else frame.to_numpy(), [0, 1]
    )
    path = tmp_path / "never.mlbundle"
    if overwrite:
        path.write_bytes(b"original")
    names = ["feature_0"] if origin == "generated" else ["a"]
    with pytest.raises(mlbundle.InvalidSchemaError, match="targets"):
        mlbundle.export(
            model,
            path,
            target_names=names,
            feature_names=["a", "b"] if origin == "declared" else None,
            overwrite=overwrite,
        )
    assert path.read_bytes() == b"original" if overwrite else not path.exists()
    assert list(tmp_path.glob(".mlbundle-*")) == []


@pytest.mark.parametrize("names", [[], ["a", "b"], "outcome", [""], ["x" * 1025], [0]])
def test_manifest_target_bounds_and_count(names, manifest_data):
    manifest_data["target"]["names"] = names
    with pytest.raises(mlbundle.InvalidManifestError):
        mlbundle.Manifest.from_bytes(canonical_dumps(manifest_data))


def test_manifest_target_duplicates_and_old_key(manifest_data):
    manifest_data["model"].update(n_outputs=2, n_classes=[2, 2])
    manifest_data["target"].update(
        names=["same", "same"], class_labels=[[0, 1], [10, 20]]
    )
    with pytest.raises(mlbundle.InvalidManifestError):
        mlbundle.Manifest.from_bytes(canonical_dumps(manifest_data))
    manifest_data["target"].pop("names")
    manifest_data["target"]["name"] = "old"
    with pytest.raises(mlbundle.InvalidManifestError):
        mlbundle.Manifest.from_bytes(canonical_dumps(manifest_data))


@pytest.mark.parametrize("samples", [False, True])
def test_reader_checks_every_declared_target(samples, tmp_path):
    frame = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    model = DecisionTreeClassifier().fit(frame, [[0, 10], [1, 20]])
    path = tmp_path / "model.mlbundle"
    mlbundle.export(
        model, path, training_data=frame, sample_rows=1 if samples else None
    )
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    parts["target"]["names"] = ["first", "b"]
    del members[mlbundle.MemberName.MANIFEST]
    align_schema(parts, members)
    bundle = write_container(parts, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(
            mlbundle.InvalidSamplesError if samples else mlbundle.InvalidSchemaError,
            match="target",
        ):
            reader(bundle)
