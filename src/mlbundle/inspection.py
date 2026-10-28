"""Public metadata-only inspection, without resolving any model adapter."""

from ._container_rules import MEMBER_ORDER, BundleSource, effective_limits
from .container import ValidatedContainer, read_validated_container
from .limits import ArchiveLimits
from .results import MetadataView
from .schema import read_declarations


def inspect(
    source: BundleSource, *, limits: ArchiveLimits | None = None
) -> MetadataView:
    """Validate the container, manifest, JSON bounds, hashes and lock without loading.

    Inspection cannot attest that model-derived claims or structure match the
    model. Only a full validating model load can provide that guarantee.
    """
    selected = effective_limits(limits)
    return metadata_view(read_validated_container(source, limits=selected), selected)


def metadata_view(container: ValidatedContainer, limits: ArchiveLimits) -> MetadataView:
    manifest, members = container.manifest, container.members
    schema, samples = read_declarations(manifest, container.json, limits)
    return MetadataView(
        manifest,
        tuple(name for name in MEMBER_ORDER if name in members),
        container.lock,
        schema=schema,
        samples=samples,
    )
