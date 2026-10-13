"""Closed fitted width transitions, successful prediction and malformed state."""

import copy
import warnings

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

import mlbundle
from mlbundle._sklearn.transform_width import output_width
from mlbundle._sklearn.trusted import NUMPY_FUNCTIONS
from pipeline_helpers import data, estimator


def transform_cases():
    return [
        StandardScaler(),
        StandardScaler(with_mean=False, with_std=False),
        MinMaxScaler(),
        MaxAbsScaler(),
        RobustScaler(),
        Normalizer(),
        PCA(2),
        VarianceThreshold(),
        SelectKBest(k=2),
        SelectPercentile(percentile=75),
        SelectFpr(alpha=1),
        SelectFdr(alpha=1),
        SelectFwe(alpha=1),
        GenericUnivariateSelect(mode="k_best", param=2),
        SimpleImputer(),
        SimpleImputer(add_indicator=True),
        SimpleImputer(strategy="constant", keep_empty_features=True),
        PolynomialFeatures(2),
        PolynomialFeatures((2, 3)),
        PolynomialFeatures(2, interaction_only=True),
        *[
            KBinsDiscretizer(n_bins=2, encode=encoding)
            for encoding in ("ordinal", "onehot", "onehot-dense")
        ],
        OneHotEncoder(),
        OneHotEncoder(min_frequency=2, drop="if_binary"),
        OrdinalEncoder(),
        OrdinalEncoder(min_frequency=2),
        *[FunctionTransformer(func) for func in (None, *NUMPY_FUNCTIONS)],
        Pipeline([("sc", StandardScaler()), ("pca", PCA(2))]),
    ]


@pytest.mark.parametrize("transform", transform_cases(), ids=lambda x: type(x).__name__)
def test_width_roundtrip(transform, tmp_path):
    frame, y = data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = Pipeline([("transform", transform), ("est", estimator())]).fit(frame, y)
        transformed = transform.transform(frame)
        assert output_width(transform, 3) == transformed.shape[1]
        path = tmp_path / "width.mlbundle"
        mlbundle.export(model, path)
        bundle = mlbundle.load(path)
        assert (
            bundle.manifest.model.pipeline.transformed_features.n_features
            == transformed.shape[1]
        )
        assert np.array_equal(bundle.predict_proba(frame), model.predict_proba(frame))


@pytest.mark.parametrize("add_indicator", [False, True])
@pytest.mark.parametrize("keep_empty", [False, True])
def test_imputer_empty_columns(add_indicator, keep_empty, tmp_path):
    frame, y = data()
    frame["b"] = np.nan
    model = Pipeline(
        [
            (
                "imp",
                SimpleImputer(
                    add_indicator=add_indicator, keep_empty_features=keep_empty
                ),
            ),
            ("est", estimator()),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(frame, y)
        path = tmp_path / "empty.mlbundle"
        mlbundle.export(model, path)
        bundle = mlbundle.load(path)
        assert (
            bundle.manifest.model.pipeline.transformed_features.n_features
            == 2 + int(keep_empty) + int(add_indicator)
        )
        assert np.array_equal(bundle.predict_proba(frame), model.predict_proba(frame))


@pytest.mark.parametrize("remainder", ["drop", "passthrough", StandardScaler()])
@pytest.mark.parametrize("columns", [["a", "b"], [0, 1], [True, True, False]])
def test_column_routing_and_remainder(remainder, columns, tmp_path):
    frame, y = data()
    transform = ColumnTransformer(
        [("scale", StandardScaler(), columns), ("empty", StandardScaler(), [])],
        remainder=remainder,
    )
    model = Pipeline([("ct", transform), ("est", estimator())]).fit(frame, y)
    path = tmp_path / "column.mlbundle"
    mlbundle.export(model, path)
    assert np.array_equal(
        mlbundle.load(path).predict_proba(frame), model.predict_proba(frame)
    )


def test_column_sparse_strings_and_names(tmp_path):
    frame, y = data()
    frame["b"] = np.where(y, "high", "low")
    transform = ColumnTransformer(
        [
            ("enc", OneHotEncoder(handle_unknown="ignore"), ["b"]),
            (
                "num",
                Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]),
                ["a", "c"],
            ),
        ],
        sparse_threshold=1,
    )
    model = Pipeline([("ct", transform), ("est", estimator())]).fit(frame, y)
    path = tmp_path / "sparse.mlbundle"
    mlbundle.export(model, path)
    bundle = mlbundle.load(path)
    assert transform.sparse_output_
    assert bundle.manifest.model.pipeline.estimator.claims.feature_names is None
    assert bundle.manifest.model.pipeline.transformed_features.names == [
        "enc__b_high",
        "enc__b_low",
        "num__a",
        "num__c",
    ]
    assert np.array_equal(bundle.predict_proba(frame), model.predict_proba(frame))


@pytest.mark.parametrize(
    "factory,attribute,value",
    [
        (StandardScaler, "scale_", np.ones(2)),
        (StandardScaler, "n_features_in_", True),
        (StandardScaler, "n_samples_seen_", np.ones(2)),
        (MinMaxScaler, "data_min_", np.ones(2)),
        (lambda: PCA(2), "components_", np.ones((2, 2))),
        (lambda: PCA(2), "n_components_", 1.5),
        (VarianceThreshold, "variances_", np.ones(2)),
        (lambda: SelectKBest(k=2), "scores_", np.ones(2)),
        (SimpleImputer, "statistics_", np.ones(2)),
        (lambda: SimpleImputer(add_indicator=True), "indicator_", None),
        (OneHotEncoder, "categories_", [np.array([1])]),
        (OneHotEncoder, "_n_features_outs", [1, 1, 1]),
        (OrdinalEncoder, "_missing_indices", {0: 99}),
        (lambda: KBinsDiscretizer(n_bins=2), "n_bins_", np.array([2, 2])),
        (lambda: KBinsDiscretizer(n_bins=2), "bin_edges_", [np.array([0, 0, 1])] * 3),
        (PolynomialFeatures, "n_output_features_", 1),
        (PolynomialFeatures, "_max_degree", 5),
    ],
)
def test_malformed_fitted_dimensions(factory, attribute, value, tmp_path):
    frame, y = data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        transform = factory().fit(frame, y)
        model = Pipeline(
            [("tr", transform), ("est", estimator().fit(transform.transform(frame), y))]
        )
        setattr(transform, attribute, value)
        with pytest.raises(mlbundle.MLBundleError):
            mlbundle.export(model, tmp_path / "never.mlbundle")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_columns", [[2]]),
        ("_transformer_to_input_indices", {"sc": [True], "remainder": [1, 2]}),
        ("_transformer_to_input_indices", {"sc": [0], "remainder": [2]}),
        ("output_indices_", {"sc": slice(0, 2), "remainder": slice(2, 4)}),
        ("sparse_output_", 1),
        ("remainder", "drop"),
    ],
)
def test_column_cache_disagreement(attribute, value, tmp_path):
    frame, y = data()
    transform = ColumnTransformer(
        [("sc", StandardScaler(), [0])], remainder="passthrough"
    ).fit(frame)
    model = Pipeline(
        [("ct", transform), ("est", estimator().fit(transform.transform(frame), y))]
    )
    setattr(transform, attribute, value)
    with pytest.raises(mlbundle.InvalidModelError, match="ColumnTransformer"):
        mlbundle.export(model, tmp_path / "never.mlbundle")


def test_grouping_drop_indicator_and_width_mismatches(tmp_path):
    frame, y = data()
    for transform in [
        OneHotEncoder(min_frequency=3, drop="first"),
        SimpleImputer(add_indicator=True),
    ]:
        training = frame.copy()
        if type(transform) is OneHotEncoder:
            training[:] = np.repeat([0, 1, 2], [12, 10, 2])[:, None]
        transform.fit(training)
        leaf = estimator().fit(transform.transform(training), y)
        if type(transform) is OneHotEncoder:
            transform._default_to_infrequent_mappings[0][0] = 99
        else:
            transform.indicator_.features_ = np.array([99])
        with pytest.raises(mlbundle.InvalidModelError, match="dimensions disagree"):
            mlbundle.export(
                Pipeline([("tr", transform), ("est", leaf)]),
                tmp_path / "never.mlbundle",
            )
    scaler = StandardScaler().fit(frame)
    leaf = estimator().fit(frame[["a", "b"]], y)
    with pytest.raises(
        mlbundle.InvalidModelError, match="output width disagrees with final estimator"
    ):
        mlbundle.export(
            Pipeline([("sc", scaler), ("est", leaf)]), tmp_path / "never.mlbundle"
        )
    nested = None
    for _ in range(18):
        nested = Pipeline([("p", nested)])
    with pytest.raises(mlbundle.UnsupportedModelError, match="nesting exceeds"):
        output_width(nested, 3)
    changed = copy.deepcopy(scaler)
    changed.n_features_in_ = 2
    with pytest.raises(mlbundle.InvalidModelError, match="input widths disagree"):
        output_width(changed, 3)
