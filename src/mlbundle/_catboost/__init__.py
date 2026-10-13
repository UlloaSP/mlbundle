"""Native CBM adapter; exact CatBoost wrappers and no reader-side file I/O."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import catboost as cb
from pydantic import JsonValue

from mlbundle.canonical import canonical_value
from mlbundle.claims import ClassLabels, ModelClaims
from mlbundle.contract import Library, MemberName, ModelKind
from mlbundle.declarations import ExportDeclarations
from mlbundle.errors import InvalidModelError, MLBundleError, UnsupportedModelError

from .native import decode
from .prediction import Predictor
from .scope import CLASSIFIERS, validate
from .state import commitment, feature_importances, json_value, snapshot


class Adapter:
    library = Library.CATBOOST
    model_member = MemberName.CBM

    def recognizes(self, model: object) -> bool:
        return isinstance(model, cb.CatBoost)

    def prepare_export(self, model: Any, declarations: ExportDeclarations) -> object:
        try:
            validate(model)
            if declarations.class_labels is not None:
                raise UnsupportedModelError(
                    reason="CatBoost already carries class labels; class_labels is unsupported."
                )
            self.extract_claims(model)
            return model
        except MLBundleError:
            raise
        except Exception:
            raise UnsupportedModelError(
                reason="CatBoost export preparation failed."
            ) from None

    def serialize(self, model: Any) -> bytes:
        try:
            validate(model)
            # Public save_model accepts paths only; export owns this temporary file.
            with TemporaryDirectory(prefix="mlbundle-catboost-") as directory:
                path = Path(directory) / "model.cbm"
                model.save_model(str(path), format="cbm")
                data = path.read_bytes()
            decode(data)
            return data
        except MLBundleError:
            raise
        except Exception:
            raise UnsupportedModelError(
                reason="CatBoost CBM serialization failed."
            ) from None

    def deserialize(self, data: bytes) -> object:
        try:
            state = decode(data)
            cls = (
                cb.CatBoostClassifier
                if state.loss in CLASSIFIERS
                else cb.CatBoostRegressor
            )
            # blob loading does not restore _init_params. Poisson's default predict
            # selects Exponent from that wrapper field, so bind the validated loss.
            model: Any = cls(loss_function=state.loss)
            model.load_model(blob=data)
            model._n_features_in = state.width
            actual = snapshot(model)
            if commitment(state) != commitment(actual):
                raise InvalidModelError(
                    "CatBoost state does not survive native construction."
                )
            validate(model)
            return model
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid CatBoost CBM payload.") from None

    def extract_claims(self, model: Any) -> ModelClaims:
        try:
            validate(model)
            state = snapshot(model)
            classifier = type(model) is cb.CatBoostClassifier
            if (
                classifier != (state.loss in CLASSIFIERS)
                or model.n_features_in_ not in (0, state.width)
                or model.tree_count_ != len(state.native["tree_depths"])
            ):
                raise InvalidModelError(
                    "CatBoost wrapper and native dimensions or kind disagree."
                )
            labels = (
                cast(ClassLabels, canonical_value([model.classes_]))
                if classifier
                else None
            )
            if classifier and (
                labels is None
                or len(labels[0])
                != (state.dimension if state.loss == "MultiClass" else 2)
            ):
                raise InvalidModelError(
                    "CatBoost classes and native dimensions disagree."
                )
            loss = model.get_params().get("loss_function")
            if loss is not None and loss.split(":", 1)[0] != state.loss:
                raise InvalidModelError("CatBoost wrapper and native loss disagree.")
            return ModelClaims(
                estimator_class=type(model).__name__,
                kind=ModelKind.CLASSIFIER if classifier else ModelKind.REGRESSOR,
                hyperparameters={
                    "parameters": canonical_value(model.get_all_params()),
                    "tree_count": model.tree_count_,
                    "best_iteration": model.get_best_iteration(),
                    "scale_and_bias": canonical_value(model.get_scale_and_bias()),
                    "probability_threshold": model.get_probability_threshold()
                    if classifier
                    else None,
                },
                state_sha256=commitment(state),
                n_features=state.width,
                n_classes=None if labels is None else [len(labels[0])],
                n_outputs=1,
                feature_names=model.feature_names_,
                feature_dtypes=None,
                class_labels=labels,
            )
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid CatBoost model claims.") from None

    def extract_structure(self, model: Any) -> dict[str, JsonValue]:
        try:
            validate(model)
            state = snapshot(model)
            return cast(
                dict[str, JsonValue],
                {
                    "family": "catboost",
                    "native_model": json_value(state.native),
                    "feature_importances": feature_importances(model, state.width),
                },
            )
        except MLBundleError:
            raise
        except Exception:
            raise InvalidModelError("Invalid CatBoost model structure.") from None

    def predictor(self, model: object, claims: ModelClaims | None = None) -> Predictor:
        return Predictor(
            model, self.extract_claims(model) if claims is None else claims
        )
