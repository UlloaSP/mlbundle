"""Supplemental skops trust owned by mlbundle, never by serialized declarations."""

import numpy as np
from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)

from mlbundle.errors import UnsupportedModelError

SCALAR_TYPES = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)
SCALAR_NAMES = tuple(f"numpy.{value.__name__}" for value in SCALAR_TYPES)
SCORE_FUNCTIONS = (
    f_classif,
    f_regression,
    chi2,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
NUMPY_FUNCTIONS = (
    np.log,
    np.log1p,
    np.log2,
    np.log10,
    np.exp,
    np.expm1,
    np.sqrt,
    np.square,
    np.absolute,
)
FUNCTIONS = {
    **{f"{value.__module__}.{value.__name__}": value for value in SCORE_FUNCTIONS},
    **{f"numpy.{value.__name__}": value for value in NUMPY_FUNCTIONS},
}


def reference_identifier(value: object) -> str | None:
    """Check native identity, rather than trusting a callable's advertised name."""
    for name, native in zip(SCALAR_NAMES, SCALAR_TYPES, strict=True):
        if value is native:
            return name
    for name, native in FUNCTIONS.items():
        if value is native:
            return name
    return None


def dtype_identifier(value: np.dtype) -> str:
    # Categorical SimpleImputer stores object; no supported fit needs string,
    # datetime, void, compound or platform-dependent extended-precision dtypes.
    if (
        value.fields is not None
        or value.subdtype is not None
        or value.metadata is not None
        or value.newbyteorder("=")
        not in (*map(np.dtype, SCALAR_TYPES), np.dtype(object))
    ):
        raise UnsupportedModelError(
            ("numpy.dtype",), reason="Unsupported scalar dtype."
        )
    return value.str


OPTIMIZER_TYPES = (
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
)
# Each supplemental name is restricted to its owning loader before skops audit.
TRUSTED_TYPES = ("sklearn.tree._tree.Tree", *OPTIMIZER_TYPES, "numpy.dtype", *FUNCTIONS)
