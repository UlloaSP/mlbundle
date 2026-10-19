"""Local-only DataFrame inference; optional libraries enter only on export."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError

from .canonical import canonical_dumps, canonical_value
from .claims import ModelClaims
from .contract import FeatureOrigin, MemberName
from .errors import (
    InvalidManifestError,
    InvalidSamplesError,
    InvalidSchemaError,
    MissingOptionalDependencyError,
    MLBundleError,
)
from .limits import ArchiveLimits
from .manifest import FeatureInfo
from .samples import sample_bytes
from .schema import reports_for, validate_schema, validate_separator

if TYPE_CHECKING:
    from pandas import DataFrame


@dataclass(frozen=True)
class TrainingDeclaration:
    names: list[str]
    dtypes: list[str | None]
    schema: bytes
    samples: bytes | None
    origin: FeatureOrigin


def _feature_frame(
    data: DataFrame,
    claims: ModelClaims,
    feature_names: Sequence[str] | None,
    target_names: list[str] | None,
) -> DataFrame:
    if data.empty:
        raise InvalidSchemaError("training_data must be a nonempty pandas DataFrame.")
    columns = list(data.columns)
    if any(type(name) is not str or not 1 <= len(name) <= 1024 for name in columns):
        raise InvalidSchemaError("Training columns must be bounded nonempty strings.")
    if len(set(columns)) != len(columns):
        raise InvalidSchemaError("Training column names must be unique.")
    targets = set(target_names or [])
    if not targets <= set(columns):
        raise InvalidSchemaError(
            "Every target_names entry must identify a training column."
        )
    names = [name for name in columns if name not in targets]
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
    target_names: list[str] | None,
    sample_rows: int | None,
    onehot_separator: str,
    limits: ArchiveLimits,
) -> TrainingDeclaration:
    separator = validate_separator(onehot_separator)
    if data is None and sample_rows is not None:
        raise InvalidSamplesError("sample_rows requires training_data.")
    try:
        pd = import_module("pandas")
        mlschema = import_module("mlschema")
    except ImportError:
        raise MissingOptionalDependencyError("schema") from None
    dtypes: list[str | None]
    if data is not None:
        if not isinstance(data, pd.DataFrame):
            raise InvalidSchemaError(
                "training_data must be a nonempty pandas DataFrame."
            )
        frame = _feature_frame(
            cast("DataFrame", data), claims, feature_names, target_names
        )
        names = list(frame.columns)
        origin = (
            FeatureOrigin.MODEL
            if claims.feature_names is not None
            else FeatureOrigin.DATAFRAME
        )
        dtypes = [str(dtype) for dtype in frame.dtypes]
    else:
        names = claims.feature_names
        origin = FeatureOrigin.MODEL
        if names is None:
            origin = (
                FeatureOrigin.DECLARED
                if feature_names is not None
                else FeatureOrigin.GENERATED
            )
            names = (
                list(feature_names)
                if feature_names is not None
                else [f"feature_{index}" for index in range(claims.n_features)]
            )
        try:
            FeatureInfo(names=names, origin=origin, dtypes=[None] * claims.n_features)
        except ValidationError:
            raise InvalidManifestError from None
        columns = (
            range(claims.n_features) if origin == FeatureOrigin.GENERATED else names
        )
        frame = pd.DataFrame([[1.0] * claims.n_features], columns=columns)
        dtypes = [None] * claims.n_features
    cap = limits.member_bytes(MemberName.SCHEMA)
    try:
        fields = mlschema.infer_schema(frame, onehot_separator=separator)
        if origin in (FeatureOrigin.DATAFRAME, FeatureOrigin.DECLARED):
            fields = mlschema.with_positional_mappings(fields, names)
        schema = canonical_value(
            {
                "schema_version": "2",
                "source": "default" if data is None else "training_data",
                "onehot_separator": separator,
                "fields": fields,
                "reports": reports_for(claims),
            },
            max_bytes=cap,
        )
    except MLBundleError:
        raise
    except Exception:
        raise InvalidSchemaError("Local mlschema inference failed.") from None
    validate_schema(
        schema, names, claims, origin=origin, target_names=target_names, max_bytes=cap
    )
    samples = (
        None
        if sample_rows is None
        else sample_bytes(frame, sample_rows, limits.member_bytes(MemberName.SAMPLES))
    )
    return TrainingDeclaration(
        names=names,
        dtypes=dtypes,
        schema=canonical_dumps(schema, max_bytes=cap),
        samples=samples,
        origin=origin,
    )
