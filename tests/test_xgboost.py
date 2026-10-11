"""Native XGBoost round trips, hostile payloads and claim verification together."""

import json
import struct
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import load_iris

import mlbundle
from mlbundle import (
    ClaimMismatchError,
    InvalidModelError,
    MemberName,
    UnsupportedModelError,
)
from mlbundle._xgboost import Adapter
from mlbundle._xgboost.native import REGRESSORS
from mlbundle._xgboost.ubjson import Buffer, decode
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.container import read_container
from mlbundle.declarations import export_declarations
from mlbundle.writing import write_container


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def classifier(x, y, **params):
    return xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, **params).fit(x, y)


@pytest.mark.parametrize(
    "kind,frame,names",
    [
        ("binary", False, None),
        ("multiclass", True, None),
        ("regression", False, list("abcd")),
        ("multiclass", False, list("abcd")),
    ],
)
def test_roundtrip(kind, frame, names, data, tmp_path):
    x, y = data
    inputs = pd.DataFrame(x, columns=list("abcd")) if frame else x
    labels = [["setosa", "versicolor", "virginica"]] if kind == "multiclass" else None
    model = (
        xgb.XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1).fit(inputs, y)
        if kind == "regression"
        else classifier(inputs, y % 2 if kind == "binary" else y)
    )
    before = bytes(model.get_booster().save_raw(raw_format="ubj"))
    path = tmp_path / "model.mlbundle"
    view = mlbundle.export(model, path, feature_names=names, class_labels=labels)
    loaded = mlbundle.load(path)
    assert type(loaded.native_model) is type(model)
    assert view.manifest == loaded.manifest
    assert bytes(model.get_booster().save_raw(raw_format="ubj")) == before
    prepared = Adapter().prepare_export(model, export_declarations(labels))
    assert canonical_dumps(
        loaded.manifest.verified_claims().model_dump(mode="json")
    ) == canonical_dumps(Adapter().extract_claims(prepared).model_dump(mode="json"))
    assert canonical_dumps(loaded.structure) == canonical_dumps(
        Adapter().extract_structure(prepared)
    )
    method = "predict" if kind == "regression" else "predict_proba"
    for values in [inputs, x, x.tolist()]:
        assert np.array_equal(
            getattr(loaded, method)(values), getattr(model, method)(values)
        )
    assert np.array_equal(loaded.native_model.predict(inputs), model.predict(inputs))
    assert loaded.class_labels == (None if kind == "regression" else labels or [[0, 1]])
    assert loaded.feature_origin == (
        "model" if frame else "dataframe" if names else "generated"
    )
    if frame:
        assert loaded.manifest.features.model_dtypes == ["float"] * 4


def test_wrapper_facts_and_in_memory_native_persistence(data, tmp_path):
    x, y = data
    for labels in [np.array(["a", "b", "c"])[y], np.array([10, 20, 30])[y]]:
        with pytest.raises(ValueError, match="Invalid classes inferred"):
            classifier(x, labels)
    frame = pd.DataFrame(x, columns=list("abcd"))
    for cls in [xgb.XGBClassifier, xgb.XGBRegressor]:
        model = cls(
            n_estimators=3, max_depth=2, learning_rate=0.2, random_state=7, n_jobs=1
        ).fit(frame, y)
        booster = model.get_booster()
        booster.set_attr(probe='["red","green","blue"]', best_iteration="1")
        restored = cls()
        restored.load_model(bytearray(booster.save_raw(raw_format="ubj")))
        assert booster.attributes() == restored.get_booster().attributes()
        assert booster.feature_names == restored.get_booster().feature_names
        assert booster.feature_types == restored.get_booster().feature_types
        assert np.array_equal(model.feature_names_in_, restored.feature_names_in_)
        assert (
            model.n_features_in_ == restored.n_features_in_
            and model.best_iteration == restored.best_iteration
        )
        assert model.get_params() != restored.get_params()
        before = json.loads(booster.save_config())["learner"]
        after = json.loads(restored.get_booster().save_config())["learner"]
        assert before != after
        assert before["learner_model_param"] == after["learner_model_param"]
        assert before["objective"] == after["objective"]
        assert np.array_equal(model.predict(frame), restored.predict(frame))
        if cls is xgb.XGBClassifier:
            assert model.n_classes_ == restored.n_classes_ == 3
            assert model.classes_.tolist() == restored.classes_.tolist() == [0, 1, 2]
        path = tmp_path / f"{cls.__name__}.ubj"
        model.save_model(path)
        disk = cls()
        disk.load_model(path)
        assert np.array_equal(model.predict(frame), disk.predict(frame))
        raw = xgb.Booster()
        raw.load_model(bytearray(booster.save_raw(raw_format="ubj")))
        assert raw.attributes() == booster.attributes()
        assert np.array_equal(
            raw.predict(xgb.DMatrix(frame)), booster.predict(xgb.DMatrix(frame))
        )


@pytest.mark.parametrize(
    "labels",
    [
        [[0, 0, 1]],
        [[0, 1]],
        [[0, 1, 2, 3]],
        [["a", 1, "c"]],
        [[None, None, None]],
        [[float("nan"), 1, 2]],
        [[{}, {}, {}]],
    ],
)
def test_label_rejections(labels, data, tmp_path):
    x, y = data
    with pytest.raises(UnsupportedModelError):
        mlbundle.export(
            classifier(x, y), tmp_path / "bad.mlbundle", class_labels=labels
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "labels", [[[10, 20, 30]], [[1.0, 2.0, 3.0]], [["", "b", "c"]]]
)
def test_original_label_types(labels, data, tmp_path):
    x, y = data
    path = tmp_path / "labels.mlbundle"
    mlbundle.export(classifier(x, y), path, class_labels=labels)
    assert mlbundle.load(path).class_labels == labels


def test_unsupported_models(data, tmp_path):
    x, y = data

    def custom(labels, predictions):
        return predictions - labels, np.ones_like(predictions)

    models = [
        xgb.XGBClassifier(),
        xgb.XGBRegressor(),
        xgb.XGBRanker(n_estimators=2, n_jobs=1).fit(x, y, qid=np.zeros(len(y))),
        xgb.XGBRegressor(n_estimators=2, n_jobs=1, objective=custom).fit(x, y),
        classifier(x, y).get_booster(),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        models.extend(
            [
                xgb.XGBRFClassifier(n_estimators=2, n_jobs=1).fit(x, y),
                xgb.XGBRFRegressor(n_estimators=2, n_jobs=1).fit(x, y),
            ]
        )
    for model in models:
        with pytest.raises(UnsupportedModelError):
            mlbundle.export(model, tmp_path / "bad.mlbundle")
    regressor = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, y)
    with pytest.raises(UnsupportedModelError, match="Regressors"):
        mlbundle.export(regressor, tmp_path / "bad.mlbundle", class_labels=[[0, 1, 2]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        linear = xgb.XGBRegressor(booster="gblinear", n_estimators=2, n_jobs=1).fit(
            x, y
        )
        with pytest.raises(UnsupportedModelError, match="gbtree"):
            mlbundle.export(linear, tmp_path / "bad.mlbundle")
    multi = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, np.column_stack([y, y]))
    with pytest.raises(UnsupportedModelError, match="single target"):
        mlbundle.export(multi, tmp_path / "bad.mlbundle")


@pytest.mark.parametrize("objective", sorted(REGRESSORS))
def test_supported_regression_objectives(objective, data, tmp_path):
    x, y = data
    target = y / 3 + 0.1 if objective == "reg:logistic" else y + 1.0
    model = xgb.XGBRegressor(objective=objective, n_estimators=2, n_jobs=1).fit(
        x, target
    )
    path = tmp_path / "regression.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(mlbundle.load(path).predict(x), model.predict(x))


def test_prediction_settings_and_dropout(data, tmp_path):
    x, y = data
    x = x.copy()
    x[0, 0] = -99
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = classifier(
            x,
            y,
            booster="dart",
            rate_drop=0.3,
            missing=-99,
            importance_type="weight",
            random_state=7,
        )
    model.get_booster().set_attr(best_iteration="1")
    path = tmp_path / "dropout.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.native_model.missing == -99
    assert loaded.native_model.importance_type == "weight"
    assert loaded.native_model.best_iteration == 1
    assert np.array_equal(loaded.predict_proba(x), model.predict_proba(x))
    assert np.array_equal(
        loaded.structure["feature_importances"], model.feature_importances_
    )


def rewrite(path, change):
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    del members[MemberName.MANIFEST]
    change(parts, members)
    return write_container(parts, members)


@pytest.mark.parametrize(
    "change,claim",
    [
        (
            lambda m, b: m["target"].update(class_labels=[["c", "b", "a"]]),
            "class_labels",
        ),
        (
            lambda m, b: m["target"].update(class_labels=[["x", "b", "c"]]),
            "class_labels",
        ),
        (
            lambda m, b: (
                m["model"].update(n_classes=[2]),
                m["target"].update(class_labels=[["a", "b"]]),
            ),
            "n_classes",
        ),
        (
            lambda m, b: (
                m["model"].update(kind="regressor", n_classes=None),
                m["target"].update(class_labels=None),
            ),
            "kind",
        ),
        (
            lambda m, b: m["model"].update(estimator_class="XGBRegressor"),
            "estimator_class",
        ),
        (lambda m, b: m["features"].update(names=list("dcba")), "feature_names"),
        (
            lambda m, b: m["model"]["hyperparameters"]["learner_model_param"].update(
                base_score="[99E0]"
            ),
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
    path = tmp_path / "model.mlbundle"
    mlbundle.export(
        classifier(pd.DataFrame(x, columns=list("abcd")), y),
        path,
        class_labels=[["a", "b", "c"]],
    )
    with pytest.raises(ClaimMismatchError) as error:
        mlbundle.load(rewrite(path, change))
    assert error.value.claim == claim


@pytest.mark.parametrize(
    "alter",
    [
        lambda s: s["trees"][0].update(split_condition=99.0),
        lambda s: s["trees"][0]["children"][0].update(leaf=99.0),
        lambda s: s["trees"].pop(),
    ],
)
def test_lying_structure(alter, data, tmp_path):
    x, y = data
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(x, y), path)

    def change(m, b):
        value = canonical_loads(b[MemberName.STRUCTURE])
        alter(value)
        b[MemberName.STRUCTURE] = canonical_dumps(value)

    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, change))


def test_swapped_model(data, tmp_path):
    x, y = data
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(x, y), path)
    different = Adapter().prepare_export(
        classifier(x, (y + 1) % 3), export_declarations(None)
    )
    with pytest.raises(ClaimMismatchError):
        mlbundle.load(
            rewrite(
                path,
                lambda m, b: b.update(
                    {MemberName.UBJSON: Adapter().serialize(different)}
                ),
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"random bytes",
        b"{L\x00\x00\x00\x00\x00\x00\x00\x07learner{",
        b"{L\x00\x00\x00\x00\x00\x00\x00\x07learner{}}",
        b"{}",
        b"PK\x03\x04other library",
    ],
)
def test_hostile_payloads(payload, data, tmp_path):
    x, y = data
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(x, y), path)
    with pytest.raises(mlbundle.MLBundleError):
        mlbundle.load(
            rewrite(
                path,
                lambda m, b, payload=payload: b.update({MemberName.UBJSON: payload}),
            )
        )


def test_native_payload_rejections(data, tmp_path, monkeypatch):
    x, y = data
    model = classifier(x, y)
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model, path)
    native = bytes(model.get_booster().save_raw(raw_format="ubj"))
    for payload in [
        native[: len(native) // 2],
        bytes(model.get_booster().save_raw(raw_format="json")),
        native,
    ]:
        with pytest.raises(InvalidModelError):
            Adapter().deserialize(payload)
        with pytest.raises(InvalidModelError):
            mlbundle.load(
                rewrite(
                    path,
                    lambda m, b, payload=payload: b.update(
                        {MemberName.UBJSON: payload}
                    ),
                )
            )
    prepared = Adapter().prepare_export(model, export_declarations(None))
    native = Adapter().serialize(prepared)
    monkeypatch.setattr(
        xgb.Booster,
        "load_model",
        lambda *args: (_ for _ in ()).throw(
            xgb.core.XGBoostError("untrusted huge message")
        ),
    )
    with pytest.raises(InvalidModelError, match="UBJSON payload") as error:
        Adapter().deserialize(native)
    assert "untrusted" not in str(error.value)


def test_lazy_import_and_missing_extra(data, tmp_path, monkeypatch):
    x, y = data
    path = tmp_path / "model.mlbundle"
    model = classifier(x, y)
    mlbundle.export(model, path)
    code = "import sys,mlbundle; mlbundle.inspect(sys.argv[1]); assert not {'sklearn','skops','numpy','pandas','xgboost','onnx','onnxruntime'} & sys.modules.keys(); print('lazy import and inspect: OK')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lazy import and inspect: OK" in result.stdout
    original = mlbundle.registry.import_module

    def missing(name):
        if name == "mlbundle._xgboost":
            raise ModuleNotFoundError(name="xgboost")
        return original(name)

    monkeypatch.setattr(mlbundle.registry, "import_module", missing)
    for action in [lambda: mlbundle.export(model, path), lambda: mlbundle.load(path)]:
        with pytest.raises(
            mlbundle.MissingOptionalDependencyError, match="mlbundle\\[xgboost\\]"
        ):
            action()


def test_prediction_validation(data, monkeypatch):
    x, y = data
    model = classifier(x, y)
    predictor = Adapter().predictor(model)
    with pytest.raises(ValueError, match="width"):
        predictor.predict_proba([[0]])
    with pytest.raises(ValueError, match="regressor"):
        predictor.predict(x)
    for output in [[[float("nan")] * 3], [[0.5] * 3], [[1.1, -0.1, 0]], [[1.0]]]:
        monkeypatch.setattr(
            model, "predict_proba", lambda values, output=output: output
        )
        with pytest.raises(InvalidModelError):
            predictor.predict_proba([x[0]])
    regressor = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, y)
    predictor = Adapter().predictor(regressor)
    with pytest.raises(ValueError, match="classifier"):
        predictor.predict_proba(x)
    monkeypatch.setattr(
        regressor, "predict", lambda values: [float("inf")] * len(values)
    )
    with pytest.raises(InvalidModelError):
        predictor.predict(x)


def ubjson(value):
    if isinstance(value, Buffer):
        return (
            b"[$"
            + value.marker.encode()
            + b"#L"
            + struct.pack(">q", value.count)
            + bytes(value.data)
        )
    if isinstance(value, dict):
        return (
            b"{"
            + b"".join(
                b"L"
                + struct.pack(">q", len(key.encode()))
                + key.encode()
                + ubjson(item)
                for key, item in value.items()
            )
            + b"}"
        )
    if isinstance(value, list):
        return b"[" + b"".join(ubjson(item) for item in value) + b"]"
    if isinstance(value, str):
        return b"SL" + struct.pack(">q", len(value.encode())) + value.encode()
    return ubjson_scalar(value)


def ubjson_scalar(value):
    if type(value) is bool:
        return b"T" if value else b"F"
    if type(value) is int:
        return b"L" + struct.pack(">q", value)
    if type(value) is float:
        return b"D" + struct.pack(">d", value)
    assert value is None
    return b"Z"


@pytest.mark.parametrize(
    "change",
    [
        lambda learner: learner["learner_model_param"].update(num_feature="100001"),
        lambda learner: learner["learner_model_param"].update(num_class="100001"),
        lambda learner: learner["gradient_booster"]["model"][
            "gbtree_model_param"
        ].update(num_trees="100001"),
        lambda learner: learner["gradient_booster"]["model"]["trees"][0][
            "tree_param"
        ].update(num_nodes="999999"),
        lambda learner: learner["gradient_booster"]["model"]["trees"][0].update(
            left_children=[999999] * 5
        ),
        lambda learner: learner["objective"].update(name="rank:pairwise"),
        lambda learner: learner["gradient_booster"]["model"].update(tree_info=[-1] * 9),
        lambda learner: learner["gradient_booster"]["model"].update(
            tree_info=[900000000] * 9
        ),
        lambda learner: learner["gradient_booster"]["model"].update(tree_info=[0]),
        lambda learner: learner["gradient_booster"]["model"].update(
            iteration_indptr=[0, 999999999]
        ),
    ],
)
def test_preflight_rejects_before_native_parser(change, data, monkeypatch):
    x, y = data
    value = decode(bytes(classifier(x, y).get_booster().save_raw(raw_format="ubj")))
    change(value["learner"])
    calls = []
    monkeypatch.setattr(xgb.Booster, "load_model", lambda *args: calls.append(True))
    with pytest.raises(mlbundle.MLBundleError):
        Adapter().deserialize(ubjson(value))
    assert not calls


def test_binding_validation_and_limits(data, tmp_path):
    x, y = data
    model = Adapter().prepare_export(classifier(x, y), export_declarations(None))
    binding = canonical_loads(model.get_booster().attr("mlbundle").encode())
    for change in [
        lambda v: v.update(estimator_class="os.system"),
        lambda v: v.update(class_labels=None),
        lambda v: v["prediction"].update(extra="bad"),
        lambda v: v["prediction"].update(enable_categorical="yes"),
    ]:
        value = json.loads(json.dumps(binding))
        change(value)
        model.get_booster().set_attr(mlbundle=canonical_dumps(value).decode())
        with pytest.raises(mlbundle.MLBundleError):
            Adapter().deserialize(bytes(model.get_booster().save_raw(raw_format="ubj")))
    for limits in [
        mlbundle.ArchiveLimits(max_member_bytes=1),
        mlbundle.ArchiveLimits(max_structure_bytes=1),
    ]:
        with pytest.raises(mlbundle.LimitExceededError):
            mlbundle.export(classifier(x, y), tmp_path / "bad.mlbundle", limits=limits)
    assert not list(tmp_path.iterdir())


def test_readme_example(tmp_path):
    rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(
        rows, [0, 1, 0, 1]
    )
    path = tmp_path / "xgboost.mlbundle"
    mlbundle.export(
        model,
        path,
        feature_names=["temperature", "humidity"],
        class_labels=[["cold", "warm"]],
    )
    bundle = mlbundle.load(path)
    assert isinstance(bundle.native_model, xgb.XGBClassifier)
    assert bundle.native_model.predict(rows).tolist() == model.predict(rows).tolist()
    assert bundle.predict_proba(rows) == model.predict_proba(rows).tolist()
    assert bundle.class_labels == [["cold", "warm"]]


def test_lying_regressor_kind(data, tmp_path):
    x, y = data
    path = tmp_path / "regressor.mlbundle"
    mlbundle.export(xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, y), path)

    def change(manifest, members):
        manifest["model"].update(kind="classifier", n_classes=[2])
        manifest["target"].update(class_labels=[[False, True]])

    with pytest.raises(ClaimMismatchError, match="kind"):
        mlbundle.load(rewrite(path, change))
