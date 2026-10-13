"""Local schema inference, adapter alignment and hostile declarations together."""

import pandas as pd
import pytest
from mlschema import infer_schema
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import mlbundle
from mlbundle import InvalidSchemaError, MemberName
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.container import read_container
from mlbundle.writing import write_container
from onnx_helpers import classifier, regressor


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [0, 1, 0, 1], "b": [1.0, 0.0, 0.9, 0.1]})


@pytest.mark.parametrize("library", ["sklearn", "xgboost", "onnx"])
@pytest.mark.parametrize("kind", ["classifier", "regressor"])
@pytest.mark.parametrize("native_names", [False, True])
def test_adapter_schema_roundtrip(library, kind, native_names, frame, tmp_path):
    y = [0, 1, 0, 1]
    if library == "onnx":
        model = (
            classifier(("cold", "warm"))
            if kind == "classifier"
            else regressor(columns=native_names)
        )
    else:
        cls = {
            ("sklearn", "classifier"): DecisionTreeClassifier,
            ("sklearn", "regressor"): DecisionTreeRegressor,
            ("xgboost", "classifier"): XGBClassifier,
            ("xgboost", "regressor"): XGBRegressor,
        }[library, kind]
        kwargs = {"n_estimators": 2, "n_jobs": 1} if library == "xgboost" else {}
        model = cls(**kwargs).fit(frame if native_names else frame.to_numpy(), y)
    path = tmp_path / "schema.mlbundle"
    original = frame.copy(deep=True)
    options = (
        {"class_labels": [["cold", "warm"]]}
        if (library, kind) == ("xgboost", "classifier")
        else {}
    )
    view = mlbundle.export(
        model, path, training_data=frame, feature_names=["a", "b"], **options
    )
    loaded = mlbundle.load(path)
    assert view.schema == loaded.schema == mlbundle.inspect(path).schema
    assert view.schema["schema_version"] == "2"
    assert view.schema["source"] == "training_data"
    assert view.schema["onehot_separator"] == "__"
    assert view.samples is None and loaded.samples is None
    assert MemberName.SCHEMA in view.members and MemberName.SAMPLES not in view.members
    features = view.manifest.features
    assert features.names == ["a", "b"] and features.dtypes == ["int64", "float64"]
    named = native_names and (library != "onnx" or kind == "regressor")
    assert features.origin == ("model" if named else "dataframe")
    fields = infer_schema(frame, onehot_separator="__")
    if not named:
        for index, field in enumerate(fields):
            field["mappedTo"] = index
        values = frame.to_numpy()
        actual = (
            loaded.predict_proba(values)
            if kind == "classifier"
            else loaded.predict(values)
        )
        assert len(actual) == len(frame)
    assert view.schema["fields"] == fields
    assert features.model_dtypes == loaded.manifest.features.model_dtypes
    report = view.schema["reports"][0]
    assert report["kind"] == kind and report["mappedTo"] == 0
    if kind == "classifier":
        assert report["labels"] == [str(label) for label in loaded.class_labels[0]]
        assert report["showClassProbabilities"] is True
    else:
        assert set(report) == {"kind", "label", "mappedTo"}
    pd.testing.assert_frame_equal(frame, original)


def test_mixed_schema_and_series(tmp_path):
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0],
            "category": pd.Categorical(["low", "high"]),
            "text": ["private_a", "private_b"],
            "active": [True, False],
            "date": pd.date_range("2026-01-01", periods=2),
            "pair": [(1.0, 2.0), (2.0, 3.0)],
        }
    )
    model = Pipeline(
        [
            (
                "select",
                ColumnTransformer([("n", "passthrough", ["numeric", "active"])]),
            ),
            ("tree", DecisionTreeClassifier()),
        ]
    ).fit(frame, [False, True])
    view = mlbundle.export(model, tmp_path / "mixed.mlbundle", training_data=frame)
    assert view.schema["fields"] == infer_schema(frame, onehot_separator="__")
    assert [field["kind"] for field in view.schema["fields"]] == [
        "number",
        "category",
        "text",
        "boolean",
        "date",
        "series",
    ]
    assert view.schema["reports"][0]["labels"] == ["False", "True"]
    assert view.manifest.features.dtypes == [str(dtype) for dtype in frame.dtypes]


def test_onehot_nonadjacent_columns_and_multioutput(tmp_path):
    frame = pd.DataFrame({"color__red": [1, 0], "age": [20, 30], "color__blue": [0, 1]})
    model = DecisionTreeClassifier().fit(frame, [[False, 0], [True, 1]])
    view = mlbundle.export(model, tmp_path / "onehot.mlbundle", training_data=frame)
    assert view.schema["fields"] == infer_schema(frame, onehot_separator="__")
    field = view.schema["fields"][0]
    assert field["kind"] == "onehot-category" and field["label"] == "color"
    assert [option["mappedTo"] for option in field["options"]] == [
        "color__red",
        "color__blue",
    ]
    assert [report["mappedTo"] for report in view.schema["reports"]] == [0, 1]
    assert [report["label"] for report in view.schema["reports"]] == [
        "Predicted class 0",
        "Predicted class 1",
    ]
    assert [report["labels"] for report in view.schema["reports"]] == [
        [str(label) for label in output] for output in view.manifest.target.class_labels
    ]
    reg = DecisionTreeRegressor().fit(frame, [[0.0, 1.0], [1.0, 0.0]])
    reports = mlbundle.export(
        reg, tmp_path / "reg.mlbundle", training_data=frame
    ).schema["reports"]
    assert [report["mappedTo"] for report in reports] == [0, 1]
    assert [report["label"] for report in reports] == [
        "Predicted value 0",
        "Predicted value 1",
    ]


@pytest.mark.parametrize(
    "change,options,match",
    [
        (lambda df: [], {}, "DataFrame"),
        (lambda df: df.iloc[:0], {}, "nonempty"),
        (lambda df: df.iloc[:, :0], {}, "nonempty"),
        (lambda df: df[["a"]], {}, "count"),
        (lambda df: df.assign(extra=1), {}, "count"),
        (lambda df: df[["b", "a"]], {}, "order"),
        (lambda df: df.rename(columns={"a": "wrong"}), {}, "names"),
        (lambda df: df.set_axis(["a", "a"], axis=1), {}, "unique"),
        (lambda df: df.set_axis([0, "b"], axis=1), {}, "strings"),
        (lambda df: df.set_axis(["", "b"], axis=1), {}, "strings"),
        (lambda df: df.set_axis(["x" * 1025, "b"], axis=1), {}, "bounded"),
        (lambda df: df, {"target_names": ["absent"]}, "target_names"),
        (lambda df: df, {"feature_names": ["b", "a"]}, "feature_names"),
    ],
)
def test_training_consistency_errors(change, options, match, frame, tmp_path):
    model = DecisionTreeClassifier().fit(frame, [0, 1, 0, 1])
    path = tmp_path / "never.mlbundle"
    with pytest.raises(InvalidSchemaError, match=match):
        mlbundle.export(model, path, training_data=change(frame), **options)
    assert not path.exists()


@pytest.mark.parametrize("library", ["xgboost", "onnx"])
def test_other_native_name_order_errors(library, frame, tmp_path):
    model = (
        XGBRegressor(n_estimators=1, n_jobs=1).fit(frame, [0, 1, 0, 1])
        if library == "xgboost"
        else regressor(columns=True)
    )
    with pytest.raises(InvalidSchemaError, match="order"):
        mlbundle.export(
            model, tmp_path / "never.mlbundle", training_data=frame[["b", "a"]]
        )


def test_frame_origin_target_exclusion_and_missing_extra(frame, tmp_path, monkeypatch):
    from mlbundle import training

    model = DecisionTreeClassifier().fit(frame.to_numpy(), [0, 1, 0, 1])
    path = tmp_path / "frame.mlbundle"
    data = frame.assign(target=["secret"] * 4)
    view = mlbundle.export(model, path, training_data=data, target_names=["target"])
    assert view.manifest.features.origin == "dataframe"
    assert [field["mappedTo"] for field in view.schema["fields"]] == [0, 1]
    assert view.manifest.target.names == ["target"]
    declared = mlbundle.export(
        model, tmp_path / "declared.mlbundle", feature_names=["a", "b"]
    )
    assert declared.manifest.features.origin == "declared"
    assert declared.schema["source"] == "default"
    assert declared.manifest.features.dtypes == [None, None]

    original_import = training.import_module
    for dependency in ("pandas", "mlschema"):

        def missing(name, dependency=dependency):
            if name == dependency:
                raise ModuleNotFoundError(name=name)
            return original_import(name)

        monkeypatch.setattr(training, "import_module", missing)
        for data in (frame, None):
            with pytest.raises(
                mlbundle.MissingOptionalDependencyError, match=r"mlbundle\[schema\]"
            ):
                mlbundle.export(
                    model, tmp_path / "missing.mlbundle", training_data=data
                )
    assert not (tmp_path / "missing.mlbundle").exists()


def test_inference_failure_and_schema_budget(frame, tmp_path, monkeypatch):
    import mlschema

    model = DecisionTreeClassifier().fit(frame, [0, 1, 0, 1])
    with pytest.raises(mlbundle.LimitExceededError):
        mlbundle.export(
            model,
            tmp_path / "limit.mlbundle",
            training_data=frame,
            limits=mlbundle.ArchiveLimits(max_schema_bytes=1),
        )

    def fail(*args, **kwargs):
        raise RuntimeError("private values must not appear")

    monkeypatch.setattr(mlschema, "infer_schema", fail)
    with pytest.raises(InvalidSchemaError, match="Local mlschema inference failed"):
        mlbundle.export(model, tmp_path / "failure.mlbundle", training_data=frame)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["reports"][0].update(labels=["liar", "labels"]),
        lambda s: s["reports"][0].update(kind="regressor"),
        lambda s: s["reports"][0].update(output=1),
        lambda s: s["reports"][0].update(mappedTo=1),
        lambda s: s["reports"][0].update(mappedTo=True),
        lambda s: s["reports"].append(s["reports"][0].copy()),
        lambda s: s["reports"][0].update(labels=[0, 1]),
        lambda s: s["reports"][0].update(label="Predicted class 0"),
        lambda s: s["reports"][0].update(showClassProbabilities=1),
        lambda s: s["fields"][0].update(mappedTo="renamed", label="renamed"),
        lambda s: s["fields"][0].update(mappedTo=0),
        lambda s: s["fields"].append(s["fields"][0].copy()),
        lambda s: s["fields"].pop(),
        lambda s: s["fields"].reverse(),
        lambda s: s["fields"][0].update(required="true"),
        lambda s: s["fields"][0].update(defaultValue="private"),
        lambda s: s.update(schema_version="1"),
        lambda s: s.update(schema_version="99"),
        lambda s: s.update(source="unknown"),
        lambda s: s.pop("source"),
        lambda s: s.update(onehot_separator=""),
        lambda s: s.update(onehot_separator="x" * 17),
        lambda s: s.update(onehot_separator=1),
        lambda s: s.pop("onehot_separator"),
        lambda s: s.update(extra="unknown"),
        lambda s: s.update(fields="invalid"),
        lambda s: s.pop("reports"),
    ],
)
def test_lying_schema(change, frame, tmp_path):
    path = tmp_path / "schema.mlbundle"
    mlbundle.export(
        DecisionTreeClassifier().fit(frame, [0, 1, 0, 1]), path, training_data=frame
    )
    manifest, members = read_container(path)
    schema = canonical_loads(members[MemberName.SCHEMA])
    change(schema)
    members[MemberName.SCHEMA] = canonical_dumps(schema)
    del members[MemberName.MANIFEST]
    bundle = write_container(manifest, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidSchemaError):
            reader(bundle)
