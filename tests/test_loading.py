"""Validating load: success, lying claims and hostile native payloads together."""

import io
import json
import sys
import zipfile

import pandas as pd
import pytest
import skops.io as sio
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mlbundle
from mlbundle import (
    ClaimMismatchError,
    InvalidModelError,
    MemberName,
    UnsupportedModelError,
)
from mlbundle._sklearn import Adapter
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.container import read_container
from mlbundle.writing import write_container


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_load_sources_and_native_predictions(data, tmp_path):
    x, y = data
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(DecisionTreeClassifier(random_state=7).fit(x, y), path)
    for source in [path, str(path), path.read_bytes(), io.BytesIO(path.read_bytes())]:
        loaded = mlbundle.load(source)
        assert isinstance(loaded.native_model, DecisionTreeClassifier)
        assert (loaded.native_model.predict(x) == y).all()
        assert loaded.structure["family"] == "decision_tree"


def _rewrite(path, change):
    manifest, members = read_container(path)
    value = manifest.model_dump(mode="json", by_alias=True)
    del members[MemberName.MANIFEST]
    change(value, members)
    return write_container(value, members)


@pytest.mark.parametrize(
    "change,claim",
    [
        (lambda m, _: m["model"].update(estimator_class="Ridge"), "estimator_class"),
        (
            lambda m, _: (
                m["model"].update(kind="regressor", n_classes=None),
                m["target"].update(class_labels=None),
            ),
            "kind",
        ),
        (lambda m, _: m["target"].update(class_labels=[[2, 1, 0]]), "class_labels"),
        (
            lambda m, _: m["target"].update(class_labels=[[0, 1, "renamed"]]),
            "class_labels",
        ),
        (lambda m, _: m["target"].update(class_labels=[[0, "1", 2]]), "class_labels"),
        (
            lambda m, _: (
                m["model"].update(n_features=3),
                m["features"].update(names=list("abc"), dtypes=[None] * 3),
            ),
            "n_features",
        ),
        (lambda m, _: m["features"].update(names=list("dcba")), "feature_names"),
        (lambda m, _: m["features"].update(origin="dataframe"), "feature_names"),
        (
            lambda m, _: m["features"].update(model_dtypes=["float64"] * 4),
            "feature_dtypes",
        ),
        (
            lambda m, _: m["model"]["hyperparameters"].update(max_depth=1),
            "hyperparameters",
        ),
        (
            lambda m, b: (
                m["references"].update(structure=None),
                b.pop(MemberName.STRUCTURE),
            ),
            "structure",
        ),
    ],
)
def test_lying_manifest(change, claim, data, tmp_path):
    x, y = data
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(
        DecisionTreeClassifier().fit(pd.DataFrame(x, columns=list("abcd")), y), path
    )
    with pytest.raises(ClaimMismatchError) as error:
        mlbundle.load(_rewrite(path, change))
    assert error.value.claim == claim


@pytest.mark.parametrize(
    "model,alter",
    [
        (
            DecisionTreeClassifier(),
            lambda s: s["tree"]["threshold"].__setitem__(0, 99.0),
        ),
        (
            DecisionTreeRegressor(),
            lambda s: s["tree"]["value"][-1][0].__setitem__(0, 99.0),
        ),
        (Ridge(), lambda s: s["coef"].__setitem__(0, 99.0)),
        (RandomForestClassifier(n_estimators=3), lambda s: s["trees"].pop()),
    ],
)
def test_lying_structure(model, alter, data, tmp_path):
    x, y = data
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model.fit(x, y), path)

    def change(_, members):
        structure = canonical_loads(members[MemberName.STRUCTURE])
        alter(structure)
        members[MemberName.STRUCTURE] = canonical_dumps(structure)

    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(_rewrite(path, change))


def test_added_structure_and_swapped_model(data, tmp_path):
    x, y = data
    path = tmp_path / "pipeline.mlbundle"
    model = Pipeline(
        [("scale", StandardScaler()), ("tree", DecisionTreeClassifier())]
    ).fit(x, y)
    mlbundle.export(model, path)

    def added(m, b):
        m["references"]["structure"] = "structure.json"
        b[MemberName.STRUCTURE] = b"{}"

    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(_rewrite(path, added))
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(DecisionTreeClassifier(max_depth=1).fit(x, y), path)
    different = Adapter().serialize(DecisionTreeClassifier(max_depth=2).fit(x, y))
    with pytest.raises(ClaimMismatchError, match="hyperparameters"):
        mlbundle.load(
            _rewrite(path, lambda _, b: b.update({MemberName.SKOPS: different}))
        )


@pytest.mark.parametrize(
    "module,name", [("os", "system"), ("builtins", "eval"), ("subprocess", "Popen")]
)
def test_hostile_payload(module, name, data, tmp_path):
    x, y = data
    path, sentinel = tmp_path / "tree.mlbundle", tmp_path / "sentinel"
    mlbundle.export(DecisionTreeClassifier().fit(x, y), path)
    action = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    argument = f'echo attack > "{sentinel}"' if name == "system" else action
    if name == "Popen":
        argument = json.dumps([sys.executable, "-c", action])
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        z.writestr(
            "schema.json",
            json.dumps(
                {
                    "__module__": module,
                    "__class__": name,
                    "__loader__": "ConstructorFromReduceNode",
                    "content": {
                        "__module__": "builtins",
                        "__class__": "tuple",
                        "__loader__": "TupleNode",
                        "content": [
                            {
                                "__module__": "builtins",
                                "__class__": "str",
                                "__loader__": "JsonNode",
                                "content": argument,
                                "is_json": name == "Popen",
                            }
                        ],
                    },
                    "protocol": 2,
                }
            ),
        )
    with pytest.raises(UnsupportedModelError, match=name):
        mlbundle.load(
            _rewrite(
                path, lambda _, b: b.update({MemberName.SKOPS: payload.getvalue()})
            )
        )
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "payload,error",
    [
        (b"not a zip", InvalidModelError),
        (sio.dumps({"not": "estimator"}), UnsupportedModelError),
    ],
)
def test_invalid_or_non_estimator_payload(payload, error, data, tmp_path):
    x, y = data
    path = tmp_path / "tree.mlbundle"
    mlbundle.export(DecisionTreeClassifier().fit(x, y), path)
    with pytest.raises(error):
        mlbundle.load(
            _rewrite(path, lambda _, b: b.update({MemberName.SKOPS: payload}))
        )


@pytest.mark.parametrize(
    "member,value,error",
    [
        (MemberName.STRUCTURE, None, ClaimMismatchError),
        (MemberName.SCHEMA, [], InvalidModelError),
        (MemberName.SAMPLES, [1], InvalidModelError),
        (MemberName.SCHEMA, {"fields": [], "reports": []}, None),
        (MemberName.SAMPLES, [{"feature_1": 0}], None),
    ],
)
def test_optional_members(member, value, error, tmp_path):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [("scale", StandardScaler()), ("tree", DecisionTreeClassifier())]
    ).fit([[0], [1]], [0, 1])
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model, path)
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    parts["references"][member.value[:-5]] = member.value
    del members[MemberName.MANIFEST]
    members[member] = canonical_dumps(value)
    bundle = write_container(parts, members)
    if error is not None:
        with pytest.raises(error):
            mlbundle.load(bundle)
    else:
        assert getattr(mlbundle.load(bundle), member.value[:-5]) == value
