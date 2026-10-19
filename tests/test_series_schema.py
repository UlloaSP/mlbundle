"""Native mlschema series output and its closed reader contract."""

import copy
from typing import get_args

import pandas as pd
import pytest
from mlschema import infer_schema
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle._schema_fields import OneHotOption, SchemaField
from mlbundle.canonical import canonical_dumps
from mlbundle.container import read_container
from mlbundle.writing import write_container


def test_field_vocabulary_matches_mlschema():
    from mlschema.strategies.app import builtin_kinds
    from mlschema.strategies.domain.onehot_category_field import OneHotCategoryOption

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


@pytest.fixture
def series_bundle(tmp_path):
    frame = pd.DataFrame({"pair": [(1, 2), (3, 4)]})
    path = tmp_path / "series.mlbundle"
    view = mlbundle.export(
        DecisionTreeClassifier().fit([[1], [3]], [0, 1]), path, training_data=frame
    )
    original = infer_schema(frame)[0]
    field = view.schema["fields"][0]
    assert field == {**original, "mappedTo": 0}
    assert view.schema["schema_version"] == "2"
    assert mlbundle.inspect(path).schema == mlbundle.load(path).schema == view.schema
    return path, field


def test_series_export(series_bundle):
    assert [column["id"] for column in series_bundle[1]["columns"]] == [
        "field1",
        "field2",
    ]


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f["columns"].reverse(),
        lambda f: f["columns"].pop(),
        lambda f: f["columns"].append(copy.deepcopy(f["columns"][0])),
        lambda f: f["columns"][1].update(id="field1"),
        lambda f: f["columns"][0].update(id="unknown"),
        lambda f: f["columns"][0].pop("id"),
        lambda f: f["columns"][0].update(mappedTo=0),
        lambda f: f["columns"][0].update(kind="series"),
        lambda f: f["columns"][0].update(step=0),
        lambda f: f["columns"][0].update(required="false"),
        lambda f: f["columns"][0].update(extra="private"),
    ],
)
def test_series_rejections(change, series_bundle):
    path, field = series_bundle
    manifest, members = read_container(path)
    schema = mlbundle.inspect(path).schema
    change(field)
    schema["fields"] = [field]
    members[mlbundle.MemberName.SCHEMA] = canonical_dumps(schema)
    del members[mlbundle.MemberName.MANIFEST]
    data = write_container(manifest, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(mlbundle.InvalidSchemaError):
            reader(data)
