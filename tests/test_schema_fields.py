"""Closed field vocabulary: native inference shapes and malicious alterations."""

import copy

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
    view = mlbundle.export(model, path, training_data=frame, target_name=None)
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
    parts["target"]["name"] = "age"
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
        f"{index:07d}" + "".join(rng.choices(string.ascii_letters, k=1000))
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
