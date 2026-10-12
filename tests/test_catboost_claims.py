"""Recomputed CatBoost claims, complete structure, and fitted-state commitment."""

import pytest

import mlbundle
from catboost_helpers import classifier, regressor, rewrite, vector
from mlbundle.canonical import canonical_dumps, canonical_loads


@pytest.mark.parametrize(
    "change,claim",
    [
        (lambda m, b: m["target"].update(class_labels=[[10, 20, 30]]), "class_labels"),
        (lambda m, b: m["features"].update(names=list("cba")), "feature_names"),
        (lambda m, b: m["features"].update(origin="declared"), "feature_names"),
        (
            lambda m, b: m["features"].update(model_dtypes=["int64"] * 3),
            "feature_dtypes",
        ),
        (
            lambda m, b: m["model"].update(estimator_class="CatBoostRegressor"),
            "estimator_class",
        ),
        (
            lambda m, b: (
                m["model"].update(kind="regressor", n_classes=None),
                m["target"].update(class_labels=None),
            ),
            "kind",
        ),
        (
            lambda m, b: (
                m["model"].update(n_classes=[2]),
                m["target"].update(class_labels=[[0, 1]]),
            ),
            "n_classes",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"].update(tree_count=1),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"].update(best_iteration=0),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"].update(
                scale_and_bias=[2.0, 1.0]
            ),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"].update(
                probability_threshold=0.1
            ),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"]["parameters"].update(depth=9),
            "hyperparameters",
        ),
        (lambda m, b: m["model"].update(state_sha256="0" * 64), "state_sha256"),
        (
            lambda m, b: (
                m["references"].update(structure=None),
                b.pop(mlbundle.MemberName.STRUCTURE),
            ),
            "structure",
        ),
    ],
)
def test_lying_manifest(change, claim, tmp_path):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    with pytest.raises(mlbundle.ClaimMismatchError) as error:
        mlbundle.load(rewrite(path, change))
    assert error.value.claim == claim


@pytest.mark.parametrize(
    "alter",
    [
        lambda s: s["native_model"]["tree_splits"].pop(),
        lambda s: s["native_model"]["tree_depths"].pop(),
        lambda s: s["native_model"]["leaf_values"].__setitem__(0, 99.0),
        lambda s: s["native_model"]["leaf_weights"].__setitem__(0, 99.0),
        lambda s: s["native_model"]["float_features"][0].update(nan_value_treatment=2),
        lambda s: s["native_model"].update(scale=2.0),
        lambda s: s["feature_importances"].__setitem__(0, 99.0),
    ],
)
def test_lying_structure(alter, tmp_path):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)

    def change(manifest, members):
        value = canonical_loads(members[mlbundle.MemberName.STRUCTURE])
        alter(value)
        members[mlbundle.MemberName.STRUCTURE] = canonical_dumps(value)

    with pytest.raises(mlbundle.ClaimMismatchError, match="structure"):
        mlbundle.load(rewrite(path, change))


def test_native_leaf_state_changed_with_fresh_integrity(tmp_path):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)

    def change(manifest, members):
        members[mlbundle.MemberName.CBM] = vector(
            members[mlbundle.MemberName.CBM], 8, 99.0, "d"
        )

    with pytest.raises(mlbundle.ClaimMismatchError, match="state_sha256"):
        mlbundle.load(rewrite(path, change))


def test_regressor_kind_is_native(tmp_path):
    path = tmp_path / "regressor.mlbundle"
    mlbundle.export(regressor(), path)

    def change(manifest, members):
        manifest["model"].update(kind="classifier", n_classes=[2])
        manifest["target"].update(class_labels=[[0, 1]])

    with pytest.raises(mlbundle.ClaimMismatchError):
        mlbundle.load(rewrite(path, change))
