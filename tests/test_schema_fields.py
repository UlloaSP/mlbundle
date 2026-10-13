"""Closed field vocabulary: native inference shapes and malicious alterations."""

import copy
from typing import get_args

import pandas as pd
import pytest
from mlschema import infer_schema
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle import InvalidSchemaError, MemberName
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.container import read_container
from mlbundle.writing import write_container


@pytest.fixture
def bundle(tmp_path):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    frame = pd.DataFrame(
        {
            "color__red": [1, 0],
            "age": [20, 30],
            "color__blue": [0, 1],
            "category": pd.Categorical(["low", "high"]),
            "pair": [(1.0, 2.0), (2.0, 3.0)],
        }
    )
    model = Pipeline(
        [
            ("select", ColumnTransformer([("age", "passthrough", ["age"])])),
            ("tree", DecisionTreeClassifier()),
        ]
    ).fit(frame, [0, 1])
    path = tmp_path / "fields.mlbundle"
    view = mlbundle.export(model, path, training_data=frame, target_names=None)
    assert view.schema["fields"] == infer_schema(frame, onehot_separator="__")
    return path


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["fields"][0]["options"][0].update(mappedTo="age"),
        lambda s: s["fields"][0]["options"][0].update(label="wrong"),
        lambda s: s["fields"][0]["options"].reverse(),
        lambda s: s["fields"][0]["options"].pop(),
        lambda s: s["fields"][0].update(label="color__red"),
        lambda s: s["fields"][1].update(label="wrong"),
        lambda s: s["fields"][2].update(options=["low", "low"]),
        lambda s: s["fields"][3]["field1"].update(mappedTo="age"),
        lambda s: s["fields"][3].update(field1=copy.deepcopy(s["fields"][3])),
        lambda s: s["fields"][1].update(kind="custom"),
        lambda s: s["fields"][1].update(step=-1),
        lambda s: s.update(fields=[], reports=[]),
    ],
)
def test_invalid_field_contracts(change, bundle):
    manifest, members = read_container(bundle)
    schema = canonical_loads(members[MemberName.SCHEMA])
    change(schema)
    members[MemberName.SCHEMA] = canonical_dumps(schema)
    del members[MemberName.MANIFEST]
    data = write_container(manifest, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidSchemaError):
            reader(data)


@pytest.mark.parametrize("schema", [None, [], 1, {"fields": [], "reports": []}])
def test_malformed_schema_envelope(schema, bundle):
    manifest, members = read_container(bundle)
    members[MemberName.SCHEMA] = canonical_dumps(schema)
    del members[MemberName.MANIFEST]
    data = write_container(manifest, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidSchemaError):
            reader(data)


def test_matching_manifest_and_report_lie_still_fails_native_claims(bundle):
    manifest, members = read_container(bundle)
    parts = manifest.model_dump(mode="json", by_alias=True)
    parts["target"]["class_labels"] = [["changed", "labels"]]
    schema = canonical_loads(members[MemberName.SCHEMA])
    schema["reports"][0]["labels"] = ["changed", "labels"]
    members[MemberName.SCHEMA] = canonical_dumps(schema)
    del members[MemberName.MANIFEST]
    data = write_container(parts, members)
    assert mlbundle.inspect(data).schema["reports"][0]["labels"] == [
        "changed",
        "labels",
    ]
    with pytest.raises(mlbundle.ClaimMismatchError, match="class_labels"):
        mlbundle.load(data)


def test_declared_target_cannot_be_a_schema_feature(bundle):
    manifest, members = read_container(bundle)
    parts = manifest.model_dump(mode="json", by_alias=True)
    parts["target"]["names"] = ["age"]
    del members[MemberName.MANIFEST]
    data = write_container(parts, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidSchemaError, match="target"):
            reader(data)


def test_report_comparison_uses_schema_budget(tmp_path):
    import random
    import string

    from mlbundle.limits import MANIFEST_BYTES

    rng = random.Random(7)
    labels = [
        f"{index:07d}" + "".join(rng.choices(string.ascii_letters, k=1001))
        for index in range(8)
    ]
    frame = pd.DataFrame({"a": range(8)})
    targets = [[label] * 1024 for label in labels]
    path = tmp_path / "large-reports.mlbundle"
    view = mlbundle.export(
        DecisionTreeClassifier().fit(frame, targets), path, training_data=frame
    )
    cap = mlbundle.ArchiveLimits().max_schema_bytes
    assert len(canonical_dumps(view.schema["reports"], max_bytes=cap)) > MANIFEST_BYTES
    assert len(view.manifest.to_bytes()) < MANIFEST_BYTES
    assert mlbundle.inspect(path).schema == mlbundle.load(path).schema == view.schema


def test_field_vocabulary_matches_mlschema_models():
    from mlschema.strategies.app import builtin_kinds
    from mlschema.strategies.domain.onehot_category_field import OneHotCategoryOption

    from mlbundle._schema_fields import OneHotOption, SchemaField

    ours = {
        kind: set(model.model_fields)
        for model in get_args(get_args(SchemaField)[0])
        for kind in get_args(model.model_fields["kind"].annotation)
    }
    theirs = {item.name: set(item.model.model_fields) for item in builtin_kinds()}
    assert ours.keys() == theirs.keys()
    for kind, keys in ours.items():
        assert keys <= theirs[kind], kind
    assert set(OneHotOption.model_fields) <= set(OneHotCategoryOption.model_fields)


@pytest.mark.parametrize("origin", ["model", "declared", "generated"])
def test_default_inference_and_positional_predictions(origin, tmp_path):
    frame = pd.DataFrame(
        {"color::red": [1.0, 0.0], "age": [20.0, 30.0], "color::blue": [0.0, 1.0]}
    )
    model = DecisionTreeClassifier().fit(
        frame if origin == "model" else frame.to_numpy(), [0, 1]
    )
    path = tmp_path / "default.mlbundle"
    view = mlbundle.export(
        model,
        path,
        feature_names=list(frame.columns) if origin == "declared" else None,
        onehot_separator="::",
    )
    loaded = mlbundle.load(path)
    assert view.schema == mlbundle.inspect(path).schema == loaded.schema
    assert view.schema["source"] == "default" and view.samples is None
    assert view.manifest.features.origin == origin
    assert view.manifest.features.dtypes == [None] * 3
    columns = range(3) if origin == "generated" else frame.columns
    expected = infer_schema(
        pd.DataFrame([[1.0] * 3], columns=columns), onehot_separator="::"
    )
    if origin == "declared":
        expected[0]["options"][0]["mappedTo"] = 0
        expected[0]["options"][1]["mappedTo"] = 2
        expected[1]["mappedTo"] = 1
    assert view.schema["fields"] == expected
    if origin == "generated":
        assert view.manifest.features.names == ["feature_0", "feature_1", "feature_2"]
        assert all(field["step"] == 0.1 for field in expected)
    rows = frame if origin == "model" else frame.to_numpy()
    assert loaded.predict_proba(rows) == model.predict_proba(rows).tolist()
    if origin != "model":
        assert loaded.predict_proba(rows.tolist()) == model.predict_proba(rows).tolist()


@pytest.mark.parametrize("mapping", ["color::red", True, -1, 3, 100_000, 0.0])
def test_positional_series_and_onehot_mapping_rejections(mapping, tmp_path):
    frame = pd.DataFrame(
        {"color::red": [1, 0], "pair": [(1, 2), (3, 4)], "color::blue": [0, 1]}
    )
    model = DecisionTreeClassifier().fit([[1, 2, 0], [0, 3, 1]], [0, 1])
    path = tmp_path / "positions.mlbundle"
    view = mlbundle.export(model, path, training_data=frame, onehot_separator="::")
    assert [option["mappedTo"] for option in view.schema["fields"][0]["options"]] == [
        0,
        2,
    ]
    series = view.schema["fields"][1]
    assert (
        series["mappedTo"]
        == series["field1"]["mappedTo"]
        == series["field2"]["mappedTo"]
        == 1
    )
    manifest, members = read_container(path)
    del members[MemberName.MANIFEST]
    for target in (view.schema["fields"][0]["options"][0], series["field1"]):
        original = target["mappedTo"]
        target["mappedTo"] = mapping
        members[MemberName.SCHEMA] = canonical_dumps(view.schema)
        data = write_container(manifest, members)
        for reader in (mlbundle.inspect, mlbundle.load):
            with pytest.raises(InvalidSchemaError):
                reader(data)
        target["mappedTo"] = original


@pytest.mark.parametrize("separator", [None, True, 1, [], b"__", "", "x" * 17])
def test_export_separator_bounds(separator, tmp_path):
    path = tmp_path / "never.mlbundle"
    with pytest.raises(InvalidSchemaError, match="onehot_separator"):
        mlbundle.export(object(), path, onehot_separator=separator)
    assert not path.exists()


def test_separator_limit_and_reader_first_group_boundary(tmp_path):
    separator = "x" * 16
    frame = pd.DataFrame(
        {
            f"color{separator}shade{separator}red": [1, 0],
            f"color{separator}shade{separator}blue": [0, 1],
        }
    )
    path = tmp_path / "separator.mlbundle"
    view = mlbundle.export(
        DecisionTreeClassifier().fit(frame, [0, 1]),
        path,
        training_data=frame,
        onehot_separator=separator,
    )
    assert mlbundle.load(path).schema == view.schema
    manifest, members = read_container(path)
    del members[MemberName.MANIFEST]
    group = view.schema["fields"][0]
    group["label"] += separator + "shade"
    for option in group["options"]:
        option["label"] = option["value"] = option["value"].split(separator, 1)[1]
    members[MemberName.SCHEMA] = canonical_dumps(view.schema)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidSchemaError):
            reader(write_container(manifest, members))


def test_report_string_collision_fails_before_publication(tmp_path, manifest_data):
    from xgboost import XGBClassifier

    from mlbundle.schema import reports_for

    manifest_data["target"]["class_labels"] = [[1, "1"]]
    with pytest.raises(InvalidSchemaError, match="unique string forms"):
        reports_for(
            mlbundle.Manifest.from_bytes(
                canonical_dumps(manifest_data)
            ).verified_claims()
        )

    path = tmp_path / "never.mlbundle"
    model = XGBClassifier(n_estimators=1, n_jobs=1).fit([[0], [1]], [0, 1])
    with pytest.raises(mlbundle.UnsupportedModelError, match="class_labels"):
        mlbundle.export(model, path, class_labels=[[1, "1"]])
    assert not path.exists()


@pytest.mark.parametrize("missing", [True, False])
def test_schema_reference_presence(missing, tmp_path):
    from conftest import build_zip

    path = tmp_path / "schema.mlbundle"
    mlbundle.export(DecisionTreeClassifier().fit([[0], [1]], [0, 1]), path)
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    del members[MemberName.MANIFEST]
    if missing:
        del members[MemberName.SCHEMA]
    else:
        parts["references"]["schema"] = None
        del parts["integrity"]["schema.json"]
    with pytest.raises((mlbundle.InvalidManifestError, mlbundle.InvalidContainerError)):
        write_container(parts, members)
    bundle = build_zip([("manifest.json", canonical_dumps(parts)), *members.items()])
    error = mlbundle.InvalidContainerError if missing else mlbundle.InvalidManifestError
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(error, match="re-exported" if missing else "manifest"):
            reader(bundle)
