"""XGBoost payload contracts and rejection paths."""

import pytest
import xgboost as xgb

import mlbundle
from mlbundle import InvalidModelError, MemberName
from mlbundle._xgboost import Adapter
from mlbundle._xgboost.ubjson import decode
from mlbundle.declarations import export_declarations
from xgboost_helpers import classifier, rewrite, ubjson


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
def test_hostile_payloads(payload, xgb_data, tmp_path):
    x, y = xgb_data
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(x, y), path)
    with pytest.raises(mlbundle.MLBundleError):
        mlbundle.load(
            rewrite(
                path,
                lambda m, b, payload=payload: b.update({MemberName.UBJSON: payload}),
            )
        )


def test_native_payload_rejections(xgb_data, tmp_path, monkeypatch):
    x, y = xgb_data
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
def test_preflight_rejects_before_native_parser(change, xgb_data, monkeypatch):
    x, y = xgb_data
    value = decode(bytes(classifier(x, y).get_booster().save_raw(raw_format="ubj")))
    change(value["learner"])
    calls = []
    monkeypatch.setattr(xgb.Booster, "load_model", lambda *args: calls.append(True))
    with pytest.raises(mlbundle.MLBundleError):
        Adapter().deserialize(ubjson(value))
    assert not calls


def test_other_library_model_bytes(xgb_data, tmp_path):
    from sklearn.tree import DecisionTreeClassifier

    from mlbundle._sklearn import Adapter as SklearnAdapter

    x, y = xgb_data
    path = tmp_path / "xgboost.mlbundle"
    mlbundle.export(classifier(x, y), path)
    other = SklearnAdapter().serialize(DecisionTreeClassifier().fit(x, y))
    with pytest.raises(InvalidModelError, match="UBJSON"):
        mlbundle.load(rewrite(path, lambda m, b: b.update({MemberName.UBJSON: other})))
