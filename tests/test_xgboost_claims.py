"""XGBoost claims contracts and rejection paths."""

import pandas as pd
import pytest
import xgboost as xgb

import mlbundle
from mlbundle import ClaimMismatchError, MemberName
from mlbundle._xgboost import Adapter
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.declarations import export_declarations
from xgboost_helpers import classifier, rewrite


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
def test_lying_manifest(change, claim, xgb_data, tmp_path):
    x, y = xgb_data
    path = tmp_path / "model.mlbundle"
    mlbundle.export(
        classifier(pd.DataFrame(x, columns=list("abcd")), y),
        path,
        class_labels=[["a", "b", "c"]],
    )
    with pytest.raises(ClaimMismatchError) as error:
        mlbundle.load(rewrite(path, change, schema=True))
    assert error.value.claim == claim


@pytest.mark.parametrize(
    "alter",
    [
        lambda s: s["trees"][0].update(split_condition=99.0),
        lambda s: s["trees"][0]["children"][0].update(leaf=99.0),
        lambda s: s["trees"].pop(),
    ],
)
def test_lying_structure(alter, xgb_data, tmp_path):
    x, y = xgb_data
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(x, y), path)

    def change(m, b):
        value = canonical_loads(b[MemberName.STRUCTURE])
        alter(value)
        b[MemberName.STRUCTURE] = canonical_dumps(value)

    with pytest.raises(ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, change, schema=True))


def test_swapped_model(xgb_data, tmp_path):
    x, y = xgb_data
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


def test_lying_regressor_kind(xgb_data, tmp_path):
    x, y = xgb_data
    path = tmp_path / "regressor.mlbundle"
    mlbundle.export(xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(x, y), path)

    def change(manifest, members):
        manifest["model"].update(kind="classifier", n_classes=[2])
        manifest["target"].update(class_labels=[[False, True]])

    with pytest.raises(ClaimMismatchError, match="kind"):
        mlbundle.load(rewrite(path, change, schema=True))
