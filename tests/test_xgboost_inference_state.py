"""Iteration/group/dropout/category state and closed attributes affect validation."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import mlbundle
from mlbundle import ClaimMismatchError, InvalidModelError, MemberName
from mlbundle._xgboost import Adapter
from mlbundle._xgboost.ubjson import decode
from xgboost_helpers import classifier, rewrite, ubjson


def test_best_iteration_is_claimed(xgb_data, tmp_path):
    x, y = xgb_data
    model = classifier(x, y)
    path = tmp_path / "iteration.mlbundle"
    mlbundle.export(model, path)

    def change(manifest, members):
        native = decode(members[MemberName.UBJSON])
        native["learner"]["attributes"]["best_iteration"] = "0"
        members[MemberName.UBJSON] = ubjson(native)

    with pytest.raises(ClaimMismatchError, match="hyperparameters"):
        mlbundle.load(rewrite(path, change))


def test_tree_to_class_assignment_is_verified(xgb_data, tmp_path):
    x, y = xgb_data
    path = tmp_path / "groups.mlbundle"
    mlbundle.export(classifier(x, y), path)

    def change(manifest, members):
        native = decode(members[MemberName.UBJSON])
        model = native["learner"]["gradient_booster"]["model"]
        model["tree_info"] = [(group + 1) % 3 for group in model["tree_info"]]
        members[MemberName.UBJSON] = ubjson(native)

    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, change))


def test_dropout_weights_are_verified(xgb_data, tmp_path):
    x, y = xgb_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = classifier(x, y, booster="dart", rate_drop=0.5)
    path = tmp_path / "dropout.mlbundle"
    mlbundle.export(model, path)

    def change(manifest, members):
        native = decode(members[MemberName.UBJSON])
        native["learner"]["gradient_booster"]["model"]["weight_drop"][0] *= 0.5
        members[MemberName.UBJSON] = ubjson(native)

    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, change))


@pytest.mark.parametrize("where", ["attribute", "wrapper", "native"])
def test_unknown_xgboost_state_rejected(where, xgb_data, tmp_path, monkeypatch):
    x, y = xgb_data
    model = classifier(x, y)
    if where == "wrapper":
        model.review_extra_ = [1]
    elif where == "attribute":
        model.get_booster().set_attr(review_extra="unverified")
    else:
        native = decode(bytes(model.get_booster().save_raw(raw_format="ubj")))
        native["learner"]["gradient_booster"]["model"]["review_extra"] = [1]
        monkeypatch.setattr(
            xgb.Booster,
            "load_model",
            lambda *a, **kw: pytest.fail("Native parsing must not run"),
        )
        with pytest.raises(InvalidModelError, match="native state"):
            Adapter().deserialize(ubjson(native))
        return
    with pytest.raises(InvalidModelError, match="Unknown XGBoost"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")
    native = bytes(model.get_booster().save_raw(raw_format="ubj"))
    if where == "attribute":
        with pytest.raises(InvalidModelError, match="attribute"):
            Adapter().deserialize(native)


@pytest.mark.parametrize("value", ["-1", "1000000000", "x", "3"])
def test_best_iteration_is_bounded(value, xgb_data, monkeypatch):
    x, y = xgb_data
    native = decode(bytes(classifier(x, y).get_booster().save_raw(raw_format="ubj")))
    native["learner"]["attributes"]["best_iteration"] = value
    monkeypatch.setattr(
        xgb.Booster,
        "load_model",
        lambda *a, **kw: pytest.fail("Native parsing must not run"),
    )
    with pytest.raises(InvalidModelError, match="best iteration"):
        Adapter().deserialize(ubjson(native))


def test_iteration_pointers_match_parallel_tree_counts(xgb_data, monkeypatch):
    x, y = xgb_data
    native = decode(bytes(classifier(x, y).get_booster().save_raw(raw_format="ubj")))
    native["learner"]["gradient_booster"]["model"]["iteration_indptr"] = [0, 6, 9]
    monkeypatch.setattr(
        xgb.Booster,
        "load_model",
        lambda *a, **kw: pytest.fail("Native parsing must not run"),
    )
    with pytest.raises(InvalidModelError, match="iteration dimensions"):
        Adapter().deserialize(ubjson(native))


@pytest.mark.parametrize("labels", [["a", "b", "a", "b"], [10, 20, 10, 20]])
def test_categorical_models_roundtrip(labels, tmp_path):
    frame = pd.DataFrame({"a": pd.Categorical(labels), "b": [0, 1, 2, 3]})
    model = xgb.XGBClassifier(
        n_estimators=2, n_jobs=1, enable_categorical=True, min_child_weight=0
    ).fit(frame, [0, 1, 0, 1])
    path = tmp_path / "categorical.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert np.array_equal(loaded.predict_proba(frame), model.predict_proba(frame))
    assert loaded.structure["native_model"]["cats"]["enc"]


def test_category_index_cannot_drive_large_bitset_allocation(monkeypatch):
    frame = pd.DataFrame({"a": pd.Categorical(["a", "b", "a", "b"])})
    model = xgb.XGBClassifier(
        n_estimators=1, n_jobs=1, enable_categorical=True, min_child_weight=0
    ).fit(frame, [0, 1, 0, 1])
    native = decode(bytes(model.get_booster().save_raw(raw_format="ubj")))
    native["learner"]["gradient_booster"]["model"]["trees"][0]["categories"] = [
        2**31 - 1
    ]
    monkeypatch.setattr(
        xgb.Booster,
        "load_model",
        lambda *a, **kw: pytest.fail("Native parsing must not run"),
    )
    with pytest.raises(InvalidModelError, match="categorical node"):
        Adapter().deserialize(ubjson(native))


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("objective", "multi:softmax"),
        ("booster", "gblinear"),
        ("feature_types", ["c"] * 4),
    ],
)
def test_wrapper_prediction_settings_cannot_disagree_with_native_state(
    attribute, value, xgb_data, tmp_path
):
    x, y = xgb_data
    model = classifier(x, y)
    setattr(model, attribute, value)
    with pytest.raises(InvalidModelError, match="wrapper and native"):
        mlbundle.export(model, tmp_path / "bad.mlbundle")


def test_objective_class_count_checked_before_native_parser(xgb_data, monkeypatch):
    x, y = xgb_data
    native = decode(bytes(classifier(x, y).get_booster().save_raw(raw_format="ubj")))
    native["learner"]["objective"]["softmax_multiclass_param"]["num_class"] = (
        "1000000000"
    )
    monkeypatch.setattr(
        xgb.Booster,
        "load_model",
        lambda *a, **kw: pytest.fail("Native parsing must not run"),
    )
    with pytest.raises(InvalidModelError, match="class counts"):
        Adapter().deserialize(ubjson(native))


def test_category_string_offsets_require_stored_values(monkeypatch):
    frame = pd.DataFrame({"a": pd.Categorical(["a", "b", "a", "b"])})
    model = xgb.XGBClassifier(n_estimators=1, n_jobs=1, enable_categorical=True).fit(
        frame, [0, 1, 0, 1]
    )
    native = decode(bytes(model.get_booster().save_raw(raw_format="ubj")))
    native["learner"]["gradient_booster"]["model"]["cats"]["enc"][0]["offsets"] = [
        0,
        1000000000,
    ]
    monkeypatch.setattr(
        xgb.Booster,
        "load_model",
        lambda *a, **kw: pytest.fail("Native parsing must not run"),
    )
    with pytest.raises(InvalidModelError, match="string offsets"):
        Adapter().deserialize(ubjson(native))
