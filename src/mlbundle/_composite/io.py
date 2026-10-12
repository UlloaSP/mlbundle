"""Composite orchestration; each byte member belongs to its existing loader."""

from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue
from sklearn.pipeline import Pipeline

from mlbundle._native_io import NativeExport, prepare_native
from mlbundle._pipeline_contract import PipelineInfo
from mlbundle._sklearn import transformers
from mlbundle._sklearn.pipeline_scope import validate_components
from mlbundle.claims import ModelClaims, verify_claims, verify_structure
from mlbundle.contract import Library, MemberName, SerializationFormat
from mlbundle.declarations import ExportDeclarations
from mlbundle.errors import LimitExceededError, UnsupportedModelError
from mlbundle.limits import ArchiveLimits
from mlbundle.registry import get_adapter

from .claims import extract, outer_claims, rebuild, structure, verify
from .prediction import Predictor


@dataclass(frozen=True)
class CompositeExport:
    leaf: NativeExport
    prefix: Pipeline | None
    restored_prefix: Pipeline | None
    step_names: list[str]
    verbose: bool
    claims: ModelClaims
    members: dict[MemberName, bytes]
    library: Library = Library.SKLEARN
    serialization: SerializationFormat = SerializationFormat.PIPELINE

    @property
    def structure(self) -> dict[str, JsonValue] | None:
        return self.leaf.structure

    def finish(self, names: list[str]) -> tuple[PipelineInfo, dict[str, JsonValue]]:
        expected = extract(
            self.prefix,
            self.leaf.claims,
            self.leaf.library,
            self.step_names,
            self.verbose,
            names=names,
        )
        actual_leaf = self.leaf.adapter.extract_claims(self.leaf.restored)
        actual_model = rebuild(
            self.restored_prefix, self.leaf.restored, self.step_names, self.verbose
        )
        validate_components(actual_model)
        actual_claims = outer_claims(self.restored_prefix, actual_leaf, self.verbose)
        verify_claims(self.claims, actual_claims)
        actual = extract(
            self.restored_prefix,
            actual_leaf,
            self.leaf.library,
            self.step_names,
            self.verbose,
            names=names,
        )
        verify(expected, actual)
        original = structure(self.leaf.structure, expected)
        encoded = verify_structure(
            original,
            structure(self.leaf.adapter.extract_structure(self.leaf.restored), actual),
        )
        self.members[MemberName.STRUCTURE] = encoded
        return expected, original


def prepare(
    model: Any, declarations: ExportDeclarations, limits: ArchiveLimits
) -> CompositeExport:
    validate_components(model, exporting=True)
    leaf_model = model.steps[-1][1]
    roots = {base.__module__.split(".")[0] for base in type(leaf_model).__mro__}
    library = Library.XGBOOST if "xgboost" in roots else Library.CATBOOST
    adapter = get_adapter(library)
    if not adapter.recognizes(leaf_model):
        raise UnsupportedModelError(
            reason="Composite final estimator must be a supported XGBoost or CatBoost wrapper."
        )
    leaf = prepare_native(leaf_model, adapter, declarations, limits)
    prefix = restored = None
    members = dict(leaf.members)
    if len(model.steps) > 1:
        prefix = Pipeline(
            list(model.steps[:-1]),
            memory=None,
            verbose=model.verbose,
            transform_input=None,
        )
        data = transformers.serialize(prefix)
        if len(data) > limits.member_bytes(MemberName.SKOPS):
            raise LimitExceededError("Model member size limit exceeded.")
        members[MemberName.SKOPS] = data
        restored = transformers.deserialize(data)
    claims = outer_claims(prefix, leaf.claims, model.verbose)
    return CompositeExport(
        leaf,
        prefix,
        restored,
        [name for name, _ in model.steps],
        model.verbose,
        claims,
        members,
    )


def load_components(
    recipe: PipelineInfo,
    members: dict[MemberName, bytes],
    names: list[str],
    verbose: bool,
) -> tuple[Pipeline, Predictor, ModelClaims, dict[str, JsonValue]]:
    adapter = get_adapter(recipe.estimator.library)
    leaf = adapter.deserialize(members[adapter.model_member])
    prefix = (
        None
        if recipe.transformers is None
        else transformers.deserialize(members[MemberName.SKOPS])
    )
    native_claims = adapter.extract_claims(leaf)
    model = rebuild(prefix, leaf, recipe.step_names, verbose)
    validate_components(model)
    actual = extract(
        prefix, native_claims, adapter.library, recipe.step_names, verbose, names=names
    )
    verify(recipe, actual)
    claims = outer_claims(prefix, native_claims, verbose)
    extracted = structure(adapter.extract_structure(leaf), actual)
    settings = native_claims.hyperparameters.get("prediction")
    categorical = (
        adapter.library == Library.XGBOOST
        and isinstance(settings, dict)
        and settings.get("enable_categorical") is True
    )
    return (
        model,
        Predictor(prefix, adapter.predictor(leaf), claims, categorical=categorical),
        claims,
        extracted,
    )
