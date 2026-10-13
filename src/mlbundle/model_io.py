"""Shared export and validating load; adapters own all native-model behavior."""

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import cast

from pydantic import JsonValue

from ._container_rules import MEMBER_ORDER, BundleSource, effective_limits
from ._export_manifest import manifest_parts
from ._native_io import prepare_native
from .claims import Label, verify_claims, verify_structure_bytes
from .container import read_validated_container
from .contract import Library, MemberName
from .declarations import (
    ExportDeclarations,
    export_declarations,
    export_target_names,
    resolve_labels,
)
from .errors import ClaimMismatchError
from .inspection import metadata_view
from .limits import ArchiveLimits
from .registry import find_adapter, get_adapter, is_composite
from .results import LoadedBundle, MetadataView
from .schema import read_declarations, validate_separator
from .training import prepare_training
from .writing import write_validated_container


def export(
    model: object,
    destination: str | PathLike[str],
    *,
    training_data: object | None = None,
    feature_names: Sequence[str] | None = None,
    class_labels: Sequence[Sequence[Label]] | None = None,
    target_names: Sequence[str] | None = None,
    sample_rows: int | None = None,
    onehot_separator: str = "__",
    metadata: Mapping[str, JsonValue] | None = None,
    overwrite: bool = False,
    limits: ArchiveLimits | None = None,
) -> MetadataView:
    """Publish a native bundle only after its claims survive serialization/reloading.

    Model-exposed feature names take precedence over supplied names. Otherwise
    supplied names are declarations; omitted names become feature_0 … feature_{N-1}.
    Class labels are output-major declarations handled by the native adapter;
    omitted labels use native labels or generated encoded indices.
    Export requires mlbundle[schema]. With no training_data, mlschema infers a
    default schema from one row of 1.0 values; training dtypes remain null.
    training_data must be a nonempty pandas DataFrame.
    Its columns must match native names and order; feature_names must agree.
    target_names declares one unique name per output, in output order.
    With a frame, all target columns must exist and are excluded;
    otherwise the frame contains features only. Dtypes and locally inferred
    schema fields are declarations. Category values and nullability leave the
    training machine in the schema, but rows stay local by default.
    sample_rows explicitly includes the first N feature rows: samples are real
    user data. N must be a positive integer within the frame; target values and
    the index are excluded. Missing values become null, timestamps ISO strings,
    categorical/NumPy scalars JSON scalars and infinities canonical float tags.
    """
    selected = effective_limits(limits)
    separator = validate_separator(onehot_separator)
    declarations = export_declarations(class_labels)
    composite = None
    if is_composite(model):
        from ._composite.io import prepare

        prepared = composite = prepare(model, declarations, selected)
    else:
        prepared = prepare_native(model, find_adapter(model), declarations, selected)
    claims = prepared.claims
    targets = export_target_names(target_names, claims.n_outputs)
    training = prepare_training(
        training_data,
        claims,
        feature_names=feature_names,
        target_names=targets,
        sample_rows=sample_rows,
        onehot_separator=separator,
        limits=selected,
    )
    members = prepared.members
    structure = prepared.structure
    members[MemberName.SCHEMA] = training.schema
    if training.samples is not None:
        members[MemberName.SAMPLES] = training.samples
    parts = manifest_parts(
        claims,
        prepared.library,
        prepared.serialization,
        structure=structure,
        training=training,
        target_names=targets,
        metadata=metadata,
    )
    if composite is not None:
        recipe, structure = composite.finish(
            cast(list[str], cast(dict, parts["features"])["names"])
        )
        cast(dict, parts["model"])["pipeline"] = recipe.model_dump(mode="json")
    _, validated = write_validated_container(
        parts, members, destination=destination, overwrite=overwrite, limits=selected
    )
    return metadata_view(validated, selected)


def load(source: BundleSource, *, limits: ArchiveLimits | None = None) -> LoadedBundle:
    """Verify format, integrity and recomputed native claims before returning a model."""
    selected = effective_limits(limits)
    container = read_validated_container(source, limits=selected)
    manifest, members = container.manifest, container.members
    schema, samples = read_declarations(manifest, container.json, selected)
    if manifest.model.pipeline is not None:
        get_adapter(Library.SKLEARN)
        from ._composite.io import load_components

        model, predictor, actual, extracted = load_components(
            manifest.model.pipeline,
            members,
            manifest.features.names,
            cast(bool, manifest.model.hyperparameters["verbose"]),
        )
        verify_claims(manifest.verified_claims(), actual)
    else:
        adapter = get_adapter(manifest.model.library)
        model = adapter.deserialize(members[adapter.model_member])
        actual = adapter.extract_claims(model)
        expected = manifest.verified_claims()
        # Check native dimensions before using any label declaration from the bundle.
        verify_claims(
            expected.model_copy(update={"class_labels": actual.class_labels}), actual
        )
        actual = resolve_labels(
            actual,
            ExportDeclarations(
                class_labels=manifest.target.class_labels
                if actual.class_labels is None
                else None
            ),
        )
        verify_claims(expected, actual)
        predictor = adapter.predictor(model)
        extracted = adapter.extract_structure(model)
    decoded = container.json.get(MemberName.STRUCTURE)
    structure = None if decoded is None else decoded.value
    if (MemberName.STRUCTURE in members) != (extracted is not None):
        raise ClaimMismatchError("structure")
    verify_structure_bytes(b"null" if decoded is None else decoded, extracted)
    return LoadedBundle(
        manifest=manifest,
        members=tuple(name for name in MEMBER_ORDER if name in members),
        native_model=model,
        predictor=predictor,
        schema=schema,
        structure=cast(dict[str, JsonValue] | None, structure),
        samples=samples,
    )
