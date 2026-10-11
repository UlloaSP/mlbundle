"""Native, in-memory UBJSON adapter with mlbundle-owned wrapper identities."""

from typing import Any, cast

import numpy as np
import xgboost as xgb
from pydantic import JsonValue

from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.claims import ClassLabels, ModelClaims, verify_claims, verify_structure
from mlbundle.contract import Library, MemberName, ModelKind
from mlbundle.declarations import (
    ExportDeclarations,
    export_declarations,
    resolve_labels,
)
from mlbundle.errors import InvalidModelError, MLBundleError, UnsupportedModelError
from mlbundle.limits import HYPERPARAMETER_BYTES

from .native import (
    ATTRIBUTE,
    configuration,
    dimensions,
    parameters,
    tree_model,
    validate_payload,
)
from .prediction import Predictor
from .structure import extract_structure

CLASSES = {"XGBClassifier": xgb.XGBClassifier, "XGBRegressor": xgb.XGBRegressor}


def _settings(model: Any) -> dict[str, Any]:
    missing = model.missing
    if not isinstance(missing, (int, float, np.number)) or np.isinf(missing):
        raise UnsupportedModelError(reason="Unsupported XGBoost missing-value marker.")
    importance = model.importance_type or "gain"
    if importance not in ("gain", "weight", "cover", "total_gain", "total_cover"):
        raise UnsupportedModelError(reason="Unsupported XGBoost importance type.")
    return {
        "missing": canonical_loads(canonical_dumps(float(missing))),
        "importance_type": importance,
        "enable_categorical": bool(model.enable_categorical),
    }


def _native_settings(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "missing",
        "importance_type",
        "enable_categorical",
    }:
        raise InvalidModelError("Invalid XGBoost prediction settings.")
    missing = value["missing"]
    if missing == {"$float": "nan"}:
        missing = float("nan")
    if type(value["enable_categorical"]) is not bool:
        raise InvalidModelError("Invalid XGBoost categorical setting.")
    return {**value, "missing": missing}


class Adapter:
    library = Library.XGBOOST
    model_member = MemberName.UBJSON

    def recognizes(self, model: object) -> bool:
        return isinstance(model, (xgb.XGBModel, xgb.Booster))

    def _validate(self, model: Any) -> tuple[dict[str, Any], int | None]:
        if type(model) not in CLASSES.values():
            raise UnsupportedModelError(
                reason="Only XGBClassifier and XGBRegressor are supported."
            )
        if not model.__sklearn_is_fitted__():
            raise UnsupportedModelError(reason="XGBoost estimator must be fitted.")
        if callable(model.objective):
            raise UnsupportedModelError(
                reason="Custom XGBoost objectives are unsupported."
            )
        learner = configuration(model.get_booster())
        kind, width, classes = dimensions(learner)
        tree_model(learner)
        if (type(model) is xgb.XGBClassifier) != (kind == ModelKind.CLASSIFIER):
            raise UnsupportedModelError(
                reason="XGBoost wrapper kind and native objective disagree."
            )
        if model.n_features_in_ != width or (
            classes is not None and model.n_classes_ != classes
        ):
            raise InvalidModelError("XGBoost wrapper and native dimensions disagree.")
        _settings(model)
        return learner, classes

    def prepare_export(self, model: Any, declarations: ExportDeclarations) -> object:
        try:
            self._validate(model)
            claims = self.extract_claims(model)
            if (
                claims.kind == ModelKind.REGRESSOR
                and declarations.class_labels is not None
            ):
                raise UnsupportedModelError(
                    reason="Regressors cannot accept class_labels."
                )
            # Explicit declarations replace encoded class indices on the copy only.
            if declarations.class_labels is not None:
                claims = resolve_labels(
                    claims.model_copy(update={"class_labels": None}), declarations
                )
            settings = _settings(model)
            restored = type(model)(**_native_settings(settings))
            restored.load_model(model.get_booster().save_raw(raw_format="ubj"))
            restored.enable_categorical = settings["enable_categorical"]
            binding = {
                "estimator_class": type(model).__name__,
                "class_labels": claims.class_labels,
                "prediction": settings,
            }
            restored.get_booster().set_attr(
                **{
                    ATTRIBUTE: canonical_dumps(
                        binding, max_bytes=HYPERPARAMETER_BYTES
                    ).decode()
                }
            )
            verify_claims(claims, self.extract_claims(restored))
            verify_structure(
                self.extract_structure(model), self.extract_structure(restored)
            )
            return restored
        except MLBundleError:
            raise
        except Exception:
            raise UnsupportedModelError(
                reason="XGBoost export preparation failed."
            ) from None

    def serialize(self, model: Any) -> bytes:
        try:
            self._validate(model)
            data = bytes(model.get_booster().save_raw(raw_format="ubj"))
            validate_payload(data)
            return data
        except MLBundleError:
            raise
        except Exception:
            raise UnsupportedModelError(
                reason="XGBoost UBJSON serialization failed."
            ) from None

    def deserialize(self, data: bytes) -> object:
        try:
            validate_payload(data)
            booster = xgb.Booster()
            booster.load_model(bytearray(data))
            binding = _binding(booster)
            if not isinstance(binding, dict) or set(binding) != {
                "estimator_class",
                "class_labels",
                "prediction",
            }:
                raise InvalidModelError("Invalid XGBoost mlbundle binding.")
            cls = CLASSES.get(binding["estimator_class"])
            if cls is None:
                raise InvalidModelError("Unsupported XGBoost wrapper identity.")
            settings = _native_settings(binding["prediction"])
            model = cls(**settings)
            model.load_model(bytearray(data))
            model.enable_categorical = settings["enable_categorical"]
            self.extract_claims(model)
            return model
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid XGBoost UBJSON payload.") from None

    def extract_claims(self, model: Any) -> ModelClaims:
        try:
            learner, classes = self._validate(model)
            booster = model.get_booster()
            binding = booster.attr(ATTRIBUTE)
            labels = None if classes is None else [list(range(classes))]
            if binding is not None:
                record = _binding(booster)
                if record["estimator_class"] != type(model).__name__ or canonical_dumps(
                    record["prediction"]
                ) != canonical_dumps(_settings(model)):
                    raise InvalidModelError(
                        "XGBoost binding and loaded wrapper disagree."
                    )
                labels = record["class_labels"]
                declared = export_declarations(labels)
                if classes is not None and declared.class_labels is None:
                    raise InvalidModelError("Missing XGBoost bound class labels.")
            return ModelClaims(
                estimator_class=type(model).__name__,
                kind=ModelKind.CLASSIFIER
                if classes is not None
                else ModelKind.REGRESSOR,
                hyperparameters=parameters(learner, _settings(model)),
                n_features=model.n_features_in_,
                n_classes=None if classes is None else [classes],
                n_outputs=1,
                feature_names=booster.feature_names,
                feature_dtypes=booster.feature_types,
                class_labels=cast(ClassLabels | None, labels),
            )
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid XGBoost model claims.") from None

    def extract_structure(self, model: Any) -> dict[str, JsonValue]:
        try:
            self._validate(model)
            return extract_structure(model)
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid XGBoost model structure.") from None

    def predictor(self, model: object) -> Predictor:
        return Predictor(model, self.extract_claims(model))


def _binding(booster: Any) -> dict[str, Any]:
    text = booster.attr(ATTRIBUTE)
    if text is None:
        raise InvalidModelError("Missing XGBoost mlbundle binding.")
    value = canonical_loads(text.encode(), max_bytes=HYPERPARAMETER_BYTES)
    if not isinstance(value, dict):
        raise InvalidModelError("Invalid XGBoost mlbundle binding.")
    return cast(dict[str, Any], value)
