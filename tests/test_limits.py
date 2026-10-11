"""Safety ceilings can be lowered, never raised."""

import pytest
from pydantic import ValidationError

from mlbundle import ArchiveLimits, MemberName


def test_limits():
    limits = ArchiveLimits()
    assert limits.max_members == 5
    assert limits.member_bytes(MemberName.SKOPS) == 512 * 1024**2
    assert limits.member_bytes(MemberName.STRUCTURE) == 256 * 1024**2
    assert (
        ArchiveLimits(max_member_bytes=1024).member_bytes(MemberName.MANIFEST) == 1024
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"max_members": 6},
        {"max_compression_ratio": 201},
        {"max_total_bytes": 0},
        {"max_structure_bytes": "10"},
        {"unexpected": 1},
    ],
)
def test_invalid_limits(fields):
    with pytest.raises(ValidationError):
        ArchiveLimits(**fields)
