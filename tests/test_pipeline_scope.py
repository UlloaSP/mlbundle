"""Supported prefix scope and every forbidden native placement/outer option."""

import numpy as np
import pytest
from sklearn import config_context
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import mlbundle
from mlbundle._sklearn import transformers
from pipeline_helpers import data, estimator, pipeline


@pytest.mark.parametrize("library", ["xgboost", "catboost"])
@pytest.mark.parametrize(
    "placement",
    [
        "nonfinal",
        "column",
        "union",
        "stacking",
        "voting",
        "calibration",
        "search",
        "target",
        "nested",
    ],
)
def test_native_placement(library, placement, tmp_path):
    leaf = estimator(library)
    variants = {
        "nonfinal": Pipeline([("booster", leaf), ("scale", StandardScaler())]),
        "column": Pipeline(
            [
                ("columns", ColumnTransformer([("leaf", leaf, [0])])),
                ("final", estimator(library)),
            ]
        ),
        "union": FeatureUnion([("leaf", leaf)]),
        "stacking": StackingClassifier([("leaf", leaf)]),
        "voting": VotingClassifier([("leaf", leaf)]),
        "calibration": CalibratedClassifierCV(leaf),
        "search": GridSearchCV(leaf, {}),
        "target": TransformedTargetRegressor(regressor=leaf),
        "nested": Pipeline([("outer", Pipeline([("leaf", leaf)]))]),
    }
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="last step of the outermost Pipeline"
    ) as error:
        mlbundle.export(variants[placement], tmp_path / "never.mlbundle")
    assert any(name.startswith(library + ".") for name in error.value.untrusted_types)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("memory", "cache", "memory must be None"),
        ("verbose", 1, "verbose must be a boolean"),
        ("transform_input", [], "transform_input and metadata routing"),
        ("_metadata_request", {}, "transform_input and metadata routing"),
        ("unknown", 1, "Unknown outer Pipeline state"),
    ],
)
def test_outer_state(name, value, message, tmp_path):
    model, _ = pipeline()
    setattr(model, name, value)
    with pytest.raises(mlbundle.MLBundleError, match=message):
        mlbundle.export(model, tmp_path / "never.mlbundle")


@pytest.mark.parametrize(
    "names",
    [
        ["same", "same"],
        ["memory", "est"],
        ["x__y", "est"],
        ["", "est"],
        ["x" * 257, "est"],
        [0, "est"],
    ],
)
def test_step_names(names, tmp_path):
    model, _ = pipeline()
    model.steps = list(zip(names, [step for _, step in model.steps], strict=True))
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="unique bounded sklearn names"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")


def test_exact_outer_leaf_and_step_count(tmp_path):
    class CustomPipeline(Pipeline):
        pass

    model, _ = pipeline()
    for invalid in [
        CustomPipeline(model.steps),
        Pipeline([("p" + str(i), None) for i in range(128)] + [model.steps[-1]]),
    ]:
        with pytest.raises(
            mlbundle.UnsupportedModelError, match="exact outer Pipeline"
        ):
            mlbundle.export(invalid, tmp_path / "never.mlbundle")

    class CustomEstimator(type(model.steps[-1][1])):
        pass

    with pytest.raises(mlbundle.UnsupportedModelError, match="Only XGBClassifier"):
        mlbundle.export(
            Pipeline([("leaf", CustomEstimator())]), tmp_path / "never.mlbundle"
        )


def test_metadata_output_and_unfitted_prefix(tmp_path):
    model, _ = pipeline()
    with (
        config_context(transform_output="pandas"),
        pytest.raises(
            mlbundle.UnsupportedModelError, match="default global transform_output"
        ),
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")
    model.steps[0][1]._sklearn_output_config = {"transform": "polars"}
    with pytest.raises(mlbundle.UnsupportedModelError, match="default or pandas"):
        mlbundle.export(model, tmp_path / "never.mlbundle")
    model.steps[0] = ("scale", StandardScaler())
    with pytest.raises(
        mlbundle.UnsupportedModelError, match="transformers must be fitted"
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")
    model.steps[0][1]._metadata_request = {}
    with pytest.raises(mlbundle.UnsupportedModelError, match="metadata routing"):
        mlbundle.export(model, tmp_path / "never.mlbundle")


def test_transformer_route_rejections_and_success():
    frame, y = data()
    prefix = Pipeline([("scale", StandardScaler())]).fit(frame)
    restored = transformers.deserialize(transformers.serialize(prefix))
    assert np.array_equal(restored.transform(frame), prefix.transform(frame))
    for invalid, message in [
        (StandardScaler(), "nonempty exact Pipeline"),
        (Pipeline([]), "nonempty exact Pipeline"),
        (Pipeline([("leaf", LogisticRegression().fit(frame, y))]), "non-transformer"),
    ]:
        with pytest.raises(mlbundle.UnsupportedModelError, match=message):
            transformers.serialize(invalid)


def test_unverifiable_width_and_custom_function(tmp_path):
    from sklearn.kernel_approximation import RBFSampler

    frame, y = data()
    model = Pipeline([("r", RBFSampler(n_components=3)), ("est", estimator())]).fit(
        frame, y
    )
    with pytest.raises(
        mlbundle.UnsupportedModelError,
        match="output width cannot be verified without data",
    ):
        mlbundle.export(model, tmp_path / "never.mlbundle")
    model = Pipeline(
        [("custom", FunctionTransformer(lambda x: x)), ("est", estimator())]
    ).fit(frame, y)
    with pytest.raises(mlbundle.UnsupportedModelError) as error:
        mlbundle.export(model, tmp_path / "never.mlbundle")
    assert any("lambda" in name for name in error.value.untrusted_types)
