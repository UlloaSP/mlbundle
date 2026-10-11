"""Local-only DataFrame inference; optional libraries enter only on export."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, cast

from .canonical import canonical_dumps, canonical_value
from .claims import ModelClaims
from .contract import MemberName
from .errors import (
    InvalidSamplesError,
    InvalidSchemaError,
    MissingOptionalDependencyError,
    MLBundleError,
)
from .limits import ArchiveLimits
from .samples import sample_bytes
from .schema import ONEHOT_SEPARATOR, reports_for, validate_schema

if TYPE_CHECKING:
    from pandas import DataFrame


@dataclass(frozen=True)
class TrainingDeclaration:
    names: list[str]
    dtypes: list[str]
    schema: bytes
    samples: bytes | None


def _feature_frame(
    data: DataFrame,
    claims: ModelClaims,
    feature_names: Sequence[str] | None,
    target_name: str | None,
) -> DataFrame:
    if data.empty:
        raise InvalidSchemaError("training_data must be a nonempty pandas DataFrame.")
    columns = list(data.columns)
    if any(type(name) is not str or not 1 <= len(name) <= 1024 for name in columns):
        raise InvalidSchemaError("Training columns must be bounded nonempty strings.")
    if len(set(columns)) != len(columns):
        raise InvalidSchemaError("Training column names must be unique.")
    if target_name is not None and target_name not in columns:
        raise InvalidSchemaError("target_name must identify a training column.")
    names = [name for name in columns if name != target_name]
    if len(names) != claims.n_features:
        raise InvalidSchemaError("Training column count disagrees with the model.")
    if claims.feature_names is not None and names != claims.feature_names:
        raise InvalidSchemaError("Training names and order disagree with the model.")
    if feature_names is not None and list(feature_names) != names:
        raise InvalidSchemaError("feature_names must equal training names in order.")
    return data.loc[:, names]


def prepare_training(
    data: object | None,
    claims: ModelClaims,
    *,
    feature_names: Sequence[str] | None,
    target_name: str | None,
    sample_rows: int | None,
    limits: ArchiveLimits,
) -> TrainingDeclaration | None:
    if data is None:
        if sample_rows is not None:
            raise InvalidSamplesError("sample_rows requires training_data.")
        return None
    try:
        pd = import_module("pandas")
        mlschema = import_module("mlschema")
    except ImportError:
        raise MissingOptionalDependencyError("schema") from None
    if not isinstance(data, pd.DataFrame):
        raise InvalidSchemaError("training_data must be a nonempty pandas DataFrame.")
    frame = _feature_frame(cast("DataFrame", data), claims, feature_names, target_name)
    names = list(frame.columns)
    cap = limits.member_bytes(MemberName.SCHEMA)
    try:
        fields = mlschema.infer_schema(frame, onehot_separator=ONEHOT_SEPARATOR)
        schema = canonical_value(
            {"schema_version": "1", "fields": fields, "reports": reports_for(claims)},
            max_bytes=cap,
        )
    except MLBundleError:
        raise
    except Exception:
        raise InvalidSchemaError("Local mlschema inference failed.") from None
    validate_schema(schema, names, claims, max_bytes=cap)
    samples = (
        None
        if sample_rows is None
        else sample_bytes(frame, sample_rows, limits.member_bytes(MemberName.SAMPLES))
    )
    return TrainingDeclaration(
        names=names,
        dtypes=[str(dtype) for dtype in frame.dtypes],
        schema=canonical_dumps(schema, max_bytes=cap),
        samples=samples,
    )
