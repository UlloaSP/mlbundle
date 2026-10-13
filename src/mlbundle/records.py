"""Single-record input and output contracts without optional library imports."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue

from ._display import display
from .adapters import Predictor
from .contract import FeatureOrigin, ModelKind
from .errors import PredictionInputError
from .manifest import Manifest

_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_ASCII_WHITESPACE = " \t\n\r\v\f"
_MISSING_LIMIT = 5


@dataclass(frozen=True)
class OutputPrediction:
    """One output's report labels and validated numeric prediction."""

    output: int
    kind: ModelKind
    labels: tuple[str, ...] | None
    probabilities: tuple[float, ...] | None
    value: float | None
    predicted_label: str | None


def _value(value: object, key: str | int, *, required: bool) -> float:
    prefix = f"Feature {display(key, limit=128)}: "
    if value is None:
        if required:
            raise PredictionInputError(prefix + "a required feature cannot be None.")
        return math.nan
    if isinstance(value, str):
        value = value.strip(_ASCII_WHITESPACE)
        if len(value) > 64:
            raise PredictionInputError(prefix + "string exceeds 64 characters.")
        if value.lower() in ("true", "false"):
            return float(value.lower() == "true")
        if _DECIMAL.fullmatch(value) is None:
            raise PredictionInputError(
                prefix + "expected a decimal or true/false string."
            )
    elif not isinstance(value, (int, float)):
        raise PredictionInputError(
            prefix + "expected a number, boolean or numeric string."
        )
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise PredictionInputError(prefix + "number must be finite.") from None
    if not math.isfinite(number):
        raise PredictionInputError(prefix + "number must be finite.")
    return number


def _schema(schema: dict[str, JsonValue] | None) -> dict:
    if schema is None:
        raise PredictionInputError("Record input requires a validated schema.")
    return cast(dict, schema)


def row_from_record(
    manifest: Manifest, schema: dict[str, JsonValue] | None, record: object
) -> list[float]:
    """Look up only expected keys; extra keys are ignored without iterating record."""
    if not isinstance(record, Mapping):
        raise PredictionInputError("Prediction record must be a mapping.")
    named = manifest.features.origin == FeatureOrigin.MODEL
    required = {}
    for field in _schema(schema)["fields"]:
        mappings = (
            [option["mappedTo"] for option in field["options"]]
            if field["kind"] == "onehot-category"
            else [field["mappedTo"]]
        )
        for mapping in mappings:
            required[mapping] = field["required"]
    row = []
    missing = []
    missing_count = 0
    for position, name in enumerate(manifest.features.names):
        key = name if named else position
        alternate = key if named else str(position)
        present = key in record
        if not named and alternate in record:
            if present:
                raise PredictionInputError(
                    f"Feature {position}: both integer and string keys are present."
                )
            present = True
            key = alternate
        if not present:
            missing_count += 1
            if len(missing) < _MISSING_LIMIT:
                missing.append(display(alternate, limit=128))
            continue
        row.append(
            _value(record[key], key, required=required[name if named else position])
        )
    if missing_count:
        more = (
            f"; {missing_count - len(missing)} more"
            if missing_count > len(missing)
            else ""
        )
        raise PredictionInputError(f"Missing features: {', '.join(missing)}{more}.")
    return row


def predict_record(
    manifest: Manifest,
    schema: dict[str, JsonValue] | None,
    predictor: Predictor,
    record: object,
) -> tuple[OutputPrediction, ...]:
    row = row_from_record(manifest, schema, record)
    # Owned predictors may preserve names needed by their native transformers.
    record_input = getattr(predictor, "record_input", None)
    data = [row] if record_input is None else record_input(row, manifest.features.names)
    kind = manifest.model.kind
    if kind == ModelKind.CLASSIFIER:
        raw = predictor.predict_proba(data)
        decisions = predictor.predict_class_indices(data)
        outputs = [raw] if manifest.model.n_outputs == 1 else raw
        return tuple(
            OutputPrediction(
                output=report["mappedTo"],
                kind=kind,
                labels=tuple(report["labels"]),
                probabilities=tuple(cast(list[list[float]], matrix)[0]),
                value=None,
                predicted_label=None
                if decisions is None
                else report["labels"][decisions[0][report["mappedTo"]]],
            )
            for report, matrix in zip(_schema(schema)["reports"], outputs, strict=True)
        )
    values = predictor.predict(data)
    result = [values[0]] if manifest.model.n_outputs == 1 else values[0]
    return tuple(
        OutputPrediction(
            output=output,
            kind=kind,
            labels=None,
            probabilities=None,
            value=value,
            predicted_label=None,
        )
        for output, value in enumerate(cast(list[float], result))
    )
