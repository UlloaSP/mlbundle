"""Public metadata-only inspection, without resolving any model adapter."""

from ._container_rules import MEMBER_ORDER, BundleSource, effective_limits
from .container import read_container
from .limits import ArchiveLimits
from .results import MetadataView
from .schema import read_declarations


def inspect(
    source: BundleSource, *, limits: ArchiveLimits | None = None
) -> MetadataView:
    """Validate the container, manifest, JSON bounds and hashes without loading.

    Inspection cannot attest that model-derived claims or structure match the
    model. Only a full validating model load can provide that guarantee.
    """
    selected = effective_limits(limits)
    manifest, members = read_container(source, limits=selected)
    schema, samples = read_declarations(manifest, members, selected)
    return MetadataView(
        manifest,
        tuple(name for name in MEMBER_ORDER if name in members),
        schema=schema,
        samples=samples,
    )
