"""One composite extraction for export and load."""

from typing import Any

from pydantic import JsonValue
from sklearn.pipeline import Pipeline

from mlbundle._pipeline_contract import (
    NativeComponentInfo,
    PipelineInfo,
    TransformedFeatures,
)
from mlbundle._sklearn import transformers
from mlbundle._sklearn.pipeline_scope import validate_components
from mlbundle._sklearn.transform_names import input_boundary, output_names
from mlbundle._sklearn.transform_width import output_width
from mlbundle.canonical import canonical_dumps
from mlbundle.claims import ModelClaims
from mlbundle.contract import MODEL_MEMBERS, Library
from mlbundle.errors import ClaimMismatchError, InvalidModelError, MLBundleError


def outer_claims(
    prefix: Pipeline | None, leaf: ModelClaims, verbose: bool
) -> ModelClaims:
    count, names = input_boundary(prefix, leaf)
    return ModelClaims(
        estimator_class="Pipeline",
        kind=leaf.kind,
        hyperparameters={"memory": None, "verbose": verbose, "transform_input": None},
        n_features=count,
        n_classes=leaf.n_classes,
        n_outputs=leaf.n_outputs,
        feature_names=names,
        feature_dtypes=leaf.feature_dtypes if prefix is None else None,
        class_labels=leaf.class_labels,
    )


def extract(
    prefix: Pipeline | None,
    leaf: ModelClaims,
    library: Library,
    step_names: list[str],
    verbose: bool,
    *,
    names: list[str],
    transformer_adapter: transformers.Adapter,
) -> PipelineInfo:
    try:
        if library not in (Library.XGBOOST, Library.CATBOOST):
            raise InvalidModelError("Invalid composite Pipeline state.")
        outer = outer_claims(prefix, leaf, verbose)
        count = (
            outer.n_features
            if prefix is None
            else output_width(prefix, outer.n_features)
        )
        if count != leaf.n_features:
            raise InvalidModelError(
                "Transformer output width disagrees with final estimator."
            )
        transformed = names if prefix is None else output_names(prefix, names)
        origin = "input" if prefix is None else "transformer"
        if transformed is None:
            transformed = leaf.feature_names
            origin = "model"
        if transformed is None:
            transformed = [f"transformed_feature_{i}" for i in range(1, count + 1)]
            origin = "generated"
        if len(transformed) != count:
            raise InvalidModelError("Transformed names and width disagree.")
        if prefix is not None:
            validate_components(prefix)
        return PipelineInfo(
            step_names=step_names,
            transformers=None if prefix is None else transformer_adapter.claims(prefix),
            estimator=NativeComponentInfo(
                library=library,
                serialization=MODEL_MEMBERS[library][0],
                member=MODEL_MEMBERS[library][1],
                claims=leaf,
            ),
            transformed_features=TransformedFeatures(
                n_features=count, names=transformed, origin=origin
            ),
        )
    except MLBundleError:
        raise
    except Exception:
        raise InvalidModelError("Invalid composite Pipeline state.") from None


def verify(expected: PipelineInfo, actual: PipelineInfo) -> None:
    for field in ("step_names", "transformers", "transformed_features"):
        if canonical_dumps(expected.model_dump(mode="json")[field]) != canonical_dumps(
            actual.model_dump(mode="json")[field]
        ):
            raise ClaimMismatchError("pipeline." + field)
    left = expected.estimator.model_dump(mode="json")
    right = actual.estimator.model_dump(mode="json")
    for field, value in left["claims"].items():
        if canonical_dumps(value) != canonical_dumps(right["claims"][field]):
            raise ClaimMismatchError("pipeline.estimator." + field)
    if canonical_dumps(left) != canonical_dumps(right):
        raise ClaimMismatchError("pipeline.estimator")


def structure(
    native: dict[str, JsonValue] | None, recipe: PipelineInfo
) -> dict[str, JsonValue]:
    if native is None:
        raise InvalidModelError("Invalid composite Pipeline state.")
    return {
        **native,
        "pipeline": {
            "step_name": recipe.step_names[-1],
            "features": recipe.transformed_features.model_dump(mode="json"),
        },
    }


def rebuild(
    prefix: Pipeline | None, leaf: Any, step_names: list[str], verbose: bool
) -> Pipeline:
    if prefix is not None and [name for name, _ in prefix.steps] != step_names[:-1]:
        raise ClaimMismatchError("pipeline.transformers")
    return Pipeline(
        [*([] if prefix is None else prefix.steps), (step_names[-1], leaf)],
        memory=None,
        verbose=verbose,
        transform_input=None,
    )
