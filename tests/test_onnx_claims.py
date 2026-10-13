"""Lying ONNX manifests and native swaps reject recomputed model claims."""

import pytest

import mlbundle
from mlbundle import ClaimMismatchError, MemberName
from mlbundle.canonical import canonical_dumps
from onnx_helpers import classifier, regressor, rewrite, softmax


@pytest.mark.parametrize(
    "change,claim",
    [
        (
            lambda m, b: (
                m["model"].update(kind="regressor", n_classes=None),
                m["target"].update(class_labels=None),
            ),
            "kind",
        ),
        (lambda m, b: m["target"].update(class_labels=[[20, 10]]), "class_labels"),
        (
            lambda m, b: (
                m["model"].update(n_classes=[1]),
                m["target"].update(class_labels=[[10]]),
            ),
            "n_classes",
        ),
        (
            lambda m, b: (
                m["model"].update(n_features=1),
                m["features"].update(
                    names=["feature_0"], dtypes=[None], model_dtypes=["float32"]
                ),
            ),
            "n_features",
        ),
        (lambda m, b: m["environment"]["onnx"]["opsets"][0].update(version=20), "onnx"),
        (lambda m, b: m["environment"]["onnx"].update(ir_version=9), "onnx"),
        (lambda m, b: m["environment"]["onnx"].update(producer_name="liar"), "onnx"),
        (lambda m, b: m["environment"]["onnx"].update(producer_version="2"), "onnx"),
        (
            lambda m, b: m["model"]["hyperparameters"]["inputs"][0].update(name="liar"),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"]["inputs"][0].update(
                dtype="float64"
            ),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"]["inputs"][0].update(
                shape=[None, 1]
            ),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"]["outputs"][0].update(
                name="other"
            ),
            "hyperparameters",
        ),
        (
            lambda m, b: m["model"]["hyperparameters"]["metadata_props"].update(
                training="lie"
            ),
            "hyperparameters",
        ),
        (
            lambda m, b: m["features"].update(model_dtypes=["float64"] * 2),
            "feature_dtypes",
        ),
        (
            lambda m, b: m["model"].update(estimator_class="os.system"),
            "estimator_class",
        ),
        (
            lambda m, b: (
                m["references"].update(structure="structure.json"),
                b.update({MemberName.STRUCTURE: canonical_dumps({"family": "linear"})}),
            ),
            "structure",
        ),
    ],
)
def test_lying_manifest(change, claim, tmp_path):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    with pytest.raises(ClaimMismatchError) as error:
        mlbundle.load(rewrite(path, change, schema=True))
    assert error.value.claim == claim


def test_lying_feature_names(tmp_path):
    path = tmp_path / "columns.mlbundle"
    mlbundle.export(regressor(columns=True), path)
    with pytest.raises(ClaimMismatchError, match="feature_names"):
        mlbundle.load(
            rewrite(
                path, lambda m, b: m["features"].update(names=["b", "a"]), schema=True
            )
        )


@pytest.mark.parametrize(
    "replacement,claim",
    [
        (classifier((1, 2)), "class_labels"),
        (softmax(), "hyperparameters"),
        (regressor(), "kind"),
    ],
)
def test_swapped_model(replacement, claim, tmp_path):
    path = tmp_path / "original.mlbundle"
    mlbundle.export(classifier(), path)
    with pytest.raises(ClaimMismatchError) as error:
        mlbundle.load(
            rewrite(
                path,
                lambda m, b: b.update(
                    {MemberName.ONNX: replacement.SerializeToString()}
                ),
            )
        )
    assert error.value.claim == claim


def test_label_free_declarations_are_count_checked_not_authenticated(tmp_path):
    path = tmp_path / "model.mlbundle"
    mlbundle.export(softmax(), path, class_labels=[["cold", "warm"]])
    result = mlbundle.load(
        rewrite(
            path,
            lambda m, b: m["target"].update(class_labels=[["red", "blue"]]),
            schema=True,
        )
    )
    assert result.class_labels == [["red", "blue"]]
    with pytest.raises(ClaimMismatchError, match="n_classes"):
        mlbundle.load(
            rewrite(
                path,
                lambda m, b: (
                    m["model"].update(n_classes=[1]),
                    m["target"].update(class_labels=[["red"]]),
                ),
                schema=True,
            )
        )
