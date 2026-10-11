"""Public metadata-only inspection, without resolving any model adapter."""

from ._container_rules import MEMBER_ORDER, BundleSource
from .container import read_container
from .limits import ArchiveLimits
from .results import MetadataView


def inspect(
    source: BundleSource, *, limits: ArchiveLimits | None = None
) -> MetadataView:
    """Validate the container, manifest, JSON bounds and hashes without loading.

    Inspection cannot attest that model-derived claims or structure match the
    model. Only a full validating model load can provide that guarantee.
    """
    manifest, members = read_container(source, limits=limits)
    return MetadataView(
        manifest, tuple(name for name in MEMBER_ORDER if name in members)
    )
