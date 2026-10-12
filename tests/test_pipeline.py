"""Composite public round trips, declarations, inspection and prediction errors."""

import builtins
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn import config_context
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from conftest import build_zip
from mlbundle._sklearn.trusted import TRUSTED_TYPES
from mlbundle.canonical import canonical_dumps
from pipeline_helpers import data, estimator, parts, pipeline

pytestmark = [
    pytest.mark.filterwarnings("ignore:X does not have valid feature names"),
    pytest.mark.filterwarnings("ignore:X has feature names"),
]


@pytest.mark.parametrize("library", ["xgboost", "catboost"])
@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_composite_roundtrip(library, regression, prefix, tmp_path):
    model, frame = pipeline(library, regression, prefix)
    before = vars(model).copy()
    final_before = vars(model.steps[-1][1]).copy()
    destination = tmp_path / "composite.mlbundle"
    view = mlbundle.export(model, destination)
    bundle = mlbundle.load(destination)
    assert vars(model) == before
    assert vars(model.steps[-1][1]).keys() == final_before.keys()
    assert bundle.manifest == view.manifest
    assert bundle.library == "sklearn" and bundle.class_name == "Pipeline"
    assert type(bundle.native_model) is Pipeline
    recipe = bundle.manifest.model.pipeline
    assert recipe.estimator.library == library
    assert (recipe.transformers is not None) == prefix
    assert recipe.transformed_features.n_features == 3
    assert bundle.structure["family"] == library
    assert bundle.structure["pipeline"]["features"]["n_features"] == 3
    for values in [frame, frame.to_numpy(), frame.to_numpy().tolist()]:
        predict = "predict" if regression else "predict_proba"
        expected = getattr(model, predict)(values)
        assert np.array_equal(getattr(bundle, predict)(values), expected)
        assert np.array_equal(getattr(bundle.native_model, predict)(values), expected)
    with pytest.raises(ValueError, match="requires"):
        getattr(bundle, "predict_proba" if regression else "predict")(frame)
    with pytest.raises(FileExistsError):
        mlbundle.export(model, destination)
    mlbundle.export(model, destination, overwrite=True)
    assert not any(name.startswith(("xgboost.", "catboost.")) for name in TRUSTED_TYPES)


def test_schema_samples_lowered_limits_and_labels(tmp_path):
    model, frame = pipeline()
    destination = tmp_path / "with-data.mlbundle"
    mlbundle.export(
        model,
        destination,
        training_data=frame,
        sample_rows=2,
        class_labels=[["no", "yes"]],
    )
    bundle = mlbundle.load(destination)
    assert len(bundle.members) == 6
    assert bundle.samples == frame.head(2).to_dict("records")
    assert bundle.class_labels == [["no", "yes"]]
    assert list(bundle.native_model.classes_) == [0, 1]
    assert "model.skops" in bundle.manifest.integrity
    for limits in [
        mlbundle.ArchiveLimits(max_members=5),
        mlbundle.ArchiveLimits(max_member_bytes=1),
        mlbundle.ArchiveLimits(max_structure_bytes=1),
    ]:
        with pytest.raises(mlbundle.LimitExceededError):
            mlbundle.export(
                model,
                tmp_path / "never.mlbundle",
                training_data=frame,
                sample_rows=2,
                limits=limits,
            )
    assert not (tmp_path / "never.mlbundle").exists()
    model, _ = pipeline("catboost")
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="class_labels is unsupported"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle", class_labels=[[0, 1]])


def test_inspect_import_isolation_and_cli(tmp_path, monkeypatch):
    model, _ = pipeline()
    destination = tmp_path / "composite.mlbundle"
    mlbundle.export(model, destination)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "sklearn",
            "skops",
            "xgboost",
            "catboost",
            "numpy",
            "pandas",
        }
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded)
        assert mlbundle.inspect(destination).manifest.model.pipeline.step_names == [
            "scale",
            "est",
        ]
    executable = Path(sys.executable).with_name(
        "mlbundle.exe" if sys.platform == "win32" else "mlbundle"
    )
    for command in ("inspect", "verify"):
        result = subprocess.run(
            [executable, command, str(destination)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not result.stderr
        assert '"Pipeline"' in result.stdout and '"XGBClassifier"' in result.stdout
        assert (
            "Steps:" in result.stdout
            and "Boundary: input=3, transformed=3" in result.stdout
        )


@pytest.mark.parametrize("sentinel", [None, "passthrough"])
def test_identity_prefix_and_array_feature_declarations(sentinel, tmp_path):
    frame, y = data()
    model = Pipeline([("identity", sentinel), ("est", estimator())]).fit(
        frame.to_numpy(), y
    )
    path = tmp_path / "identity.mlbundle"
    mlbundle.export(model, path, feature_names=list("abc"))
    bundle = mlbundle.load(path)
    assert bundle.manifest.model.pipeline.transformed_features.names == list("abc")
    assert np.array_equal(
        bundle.predict_proba(frame.to_numpy()), model.predict_proba(frame.to_numpy())
    )


def test_per_step_pandas_output_and_global_prediction_context(tmp_path):
    frame, y = data()
    model = Pipeline(
        [
            ("scale", StandardScaler().set_output(transform="pandas")),
            ("est", estimator()),
        ]
    ).fit(frame, y)
    path = tmp_path / "pandas.mlbundle"
    mlbundle.export(model, path)
    bundle = mlbundle.load(path)
    assert bundle.manifest.model.pipeline.estimator.claims.feature_names == list("abc")
    with config_context(transform_output="pandas", enable_metadata_routing=True):
        assert np.array_equal(bundle.predict_proba(frame), model.predict_proba(frame))
        assert mlbundle.load(path).feature_names == tuple("abc")


def test_prediction_failures_are_bounded(tmp_path, monkeypatch):
    model, frame = pipeline()
    path = tmp_path / "prediction.mlbundle"
    mlbundle.export(model, path)
    bundle = mlbundle.load(path)
    for values in [[[1]], [["secret", 1, 2]], [[np.inf, 1, 2]], frame[list("bac")]]:
        with pytest.raises(mlbundle.InvalidModelError) as error:
            bundle.predict_proba(values)
        assert "secret" not in str(error.value)
    prefix = bundle.predictor.prefix
    monkeypatch.setattr(prefix, "transform", lambda _: np.zeros((1, 3)))
    with pytest.raises(mlbundle.InvalidModelError, match="preserve prediction rows"):
        bundle.predict_proba(frame)

    def fail(_):
        raise RuntimeError("secret")

    monkeypatch.setattr(prefix, "transform", fail)
    with pytest.raises(mlbundle.InvalidModelError, match="transformation failed"):
        bundle.predict_proba(frame)


def test_old_standalone_and_all_sklearn_representation(tmp_path):
    frame, y = data()
    for model in [
        estimator().fit(frame, y),
        Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier())]).fit(
            frame, y
        ),
    ]:
        path = tmp_path / "old.mlbundle"
        mlbundle.export(model, path, overwrite=True)
        bundle = mlbundle.load(path)
        assert bundle.manifest.model.pipeline is None
        record, members = parts(path)
        record["model"].pop("pipeline")
        legacy = build_zip(
            [("manifest.json", canonical_dumps(record)), *members.items()]
        )
        assert mlbundle.load(legacy).manifest.model.pipeline is None
        assert np.array_equal(bundle.predict_proba(frame), model.predict_proba(frame))
        if type(model) is Pipeline:
            assert (
                bundle.structure is None
                and bundle.manifest.model.serialization == "skops"
            )


def test_full_export_roundtrip_rejects_component_loss(tmp_path, monkeypatch):
    from mlbundle._sklearn import transformers

    model, _ = pipeline()
    original = transformers.deserialize

    def changed(data):
        restored = original(data)
        restored.steps[0][1].mean_[0] += 1
        return restored

    monkeypatch.setattr(transformers, "deserialize", changed)
    with pytest.raises(mlbundle.ClaimMismatchError, match=r"pipeline\.transformers"):
        mlbundle.export(model, tmp_path / "never.mlbundle")
    assert not list(tmp_path.iterdir())
