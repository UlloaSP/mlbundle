"""Closed, data-free fitted width proof; no samples or dummy predictions."""

from math import comb
from typing import Any

import numpy as np
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
from sklearn.impute import MissingIndicator, SimpleImputer
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
from sklearn.utils._mask import _get_mask

from mlbundle.errors import UnsupportedModelError

from ._width_checks import dimensions, indices, input_width, vector, width
from .column_width import column_width
from .encoder_width import encoder_width
from .trusted import NUMPY_FUNCTIONS

SELECTORS = (
    SelectKBest,
    SelectPercentile,
    SelectFpr,
    SelectFdr,
    SelectFwe,
    GenericUnivariateSelect,
)


def identity(model: Any) -> bool:
    return model is None or (type(model) is str and model == "passthrough")


def output_width(model: Any, count: int, *, depth: int = 0) -> int:
    width(count)
    if depth > 16:
        raise UnsupportedModelError(reason="Transformer nesting exceeds the limit.")
    if identity(model):
        return count
    if type(model) is Pipeline:
        for _, step in model.steps:
            count = output_width(step, count, depth=depth + 1)
        return count
    input_width(model, count)
    if type(model) is not FunctionTransformer or model.validate:
        dimensions(width(getattr(model, "n_features_in_", None)) == count)
    if type(model) is ColumnTransformer:
        return column_width(model, count, depth, output_width)
    return leaf_width(model, count)


def leaf_width(model: Any, count: int) -> int:
    rule = WIDTH_RULES.get(type(model))
    if rule is None:
        raise UnsupportedModelError(
            (f"{type(model).__module__}.{type(model).__name__}",),
            reason="Transformer output width cannot be verified without data.",
        )
    return rule(model, count)


def scaler_width(model: Any, count: int) -> int:
    if type(model) is StandardScaler:
        for name, required in (
            ("mean_", model.with_mean or model.with_std),
            ("var_", model.with_std),
            ("scale_", model.with_std),
        ):
            value = vector(model, name, count, required=bool(required))
            dimensions(required or value is None)
        samples = np.asarray(model.n_samples_seen_)
        dimensions(samples.ndim == 0 or samples.shape == (count,))
        return count
    fields = {
        MinMaxScaler: ("min_", "scale_", "data_min_", "data_max_", "data_range_"),
        MaxAbsScaler: ("scale_", "max_abs_"),
        RobustScaler: (),
        Normalizer: (),
    }
    if type(model) in fields:
        for name in fields[type(model)]:
            vector(model, name, count)
        if type(model) is RobustScaler:
            for name, required in (
                ("center_", model.with_centering),
                ("scale_", model.with_scaling),
            ):
                value = vector(model, name, count, required=bool(required))
                dimensions(required or value is None)
        return count
    return count


def function_width(model: Any, count: int) -> int:
    if any(
        func is not None and all(func is not native for native in NUMPY_FUNCTIONS)
        for func in (model.func, model.inverse_func)
    ):
        raise UnsupportedModelError(
            reason="Transformer output width cannot be verified without data."
        )
    if model.kw_args not in (None, {}) or model.inv_kw_args not in (None, {}):
        raise UnsupportedModelError(
            reason="Transformer output width cannot be verified without data."
        )
    return count


def pca_width(model: Any, count: int) -> int:
    result = width(model.n_components_)
    dimensions(
        result <= count and np.asarray(model.components_).shape == (result, count)
    )
    for name in (
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
    ):
        vector(model, name, result)
    vector(model, "mean_", count)
    return result


def selector_width(model: Any, count: int) -> int:
    if type(model) is VarianceThreshold:
        vector(model, "variances_", count)
    else:
        vector(model, "scores_", count)
        vector(model, "pvalues_", count, required=False)
    support = np.asarray(model.get_support())
    dimensions(support.shape == (count,) and support.dtype.kind == "b")
    return width(np.count_nonzero(support))


def imputer_width(model: SimpleImputer, count: int) -> int:
    statistics = vector(model, "statistics_", count)
    dimensions(
        type(model.keep_empty_features) is bool and type(model.add_indicator) is bool
    )
    result = (
        count
        if model.keep_empty_features
        else int(
            np.count_nonzero(np.logical_not(np.asarray(_get_mask(statistics, np.nan))))
        )
    )
    indicator = model.indicator_
    dimensions((indicator is not None) == model.add_indicator)
    if indicator is not None:
        dimensions(
            type(indicator) is MissingIndicator
            and indicator._precomputed is True
            and indicator.features == "missing-only"
            and indicator.error_on_new is False
        )
        dimensions(width(indicator._n_features) == count)
        selected = indices(indicator.features_, count)
        dimensions(selected == sorted(selected))
        result += len(selected)
    return width(result)


def bins_width(model: KBinsDiscretizer, count: int) -> int:
    bins = vector(model, "n_bins_", count)
    dimensions(
        bins is not None and bins.dtype.kind in "iu" and len(model.bin_edges_) == count
    )
    counts = [width(item) for item in np.asarray(bins)]
    for edges, size in zip(model.bin_edges_, counts, strict=True):
        values = np.asarray(edges)
        dimensions(
            values.shape == (size + 1,)
            and values.dtype.kind == "f"
            and not np.isnan(values).any()
            and bool(np.all(np.diff(values) > 0))
        )
    if model.encode == "ordinal":
        dimensions(not hasattr(model, "_encoder"))
        return count
    dimensions(
        model.encode in ("onehot", "onehot-dense")
        and type(model._encoder) is OneHotEncoder
    )
    input_width(model._encoder, count)
    dimensions(
        all(
            np.array_equal(cats, np.arange(size))
            for cats, size in zip(model._encoder.categories_, counts, strict=True)
        )
    )
    result = encoder_width(model._encoder, count)
    dimensions(
        result == sum(counts)
        and model._encoder.sparse_output == (model.encode == "onehot")
    )
    return width(result)


def polynomial_width(model: PolynomialFeatures, count: int) -> int:
    minimum, maximum = model._min_degree, model._max_degree
    dimensions(
        type(minimum) is int
        and type(maximum) is int
        and 0 <= minimum <= maximum <= 100_000
    )
    degrees = model.degree if isinstance(model.degree, tuple) else (0, model.degree)
    dimensions(
        tuple(degrees) == (minimum, maximum)
        and type(model.include_bias) is bool
        and type(model.interaction_only) is bool
    )

    # Check the bounded count before any combinations or powers_ expansion.
    def total(start: int) -> int:
        result = int(model.include_bias)
        for degree in range(
            max(1, start),
            min(maximum, count) + 1 if model.interaction_only else maximum + 1,
        ):
            result += (
                comb(count, degree)
                if model.interaction_only
                else comb(count + degree - 1, degree)
            )
            width(result)
        return width(result)

    result = total(minimum)
    dimensions(
        width(model.n_output_features_) == result
        and width(model._n_out_full) == total(0)
    )
    return result


WIDTH_RULES = {
    **dict.fromkeys(
        (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer),
        scaler_width,
    ),
    FunctionTransformer: function_width,
    PCA: pca_width,
    **dict.fromkeys((VarianceThreshold, *SELECTORS), selector_width),
    OneHotEncoder: encoder_width,
    OrdinalEncoder: encoder_width,
    SimpleImputer: imputer_width,
    KBinsDiscretizer: bins_width,
    PolynomialFeatures: polynomial_width,
}
