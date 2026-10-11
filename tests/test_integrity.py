"""Member digests and size checks."""

import pytest
from pydantic import ValidationError

from mlbundle import IntegrityMismatchError, IntegrityRecord, MemberName
from mlbundle.integrity import hash_member, verify_integrity


def test_digest_and_success():
    expected = hash_member(b"abc")
    assert expected.size == 3
    assert (
        expected.sha256
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    verify_integrity(MemberName.SKOPS, b"abc", expected)
    assert hash_member(b"").size == 0


@pytest.mark.parametrize("data,size", [(b"abd", 3), (b"abc", 4)])
def test_mismatch(data, size):
    expected = IntegrityRecord(sha256=hash_member(b"abc").sha256, size=size)
    with pytest.raises(IntegrityMismatchError, match=r"model\.skops"):
        verify_integrity(MemberName.SKOPS, data, expected)


@pytest.mark.parametrize(
    "field,value",
    [("sha256", "A" * 64), ("sha256", "a" * 65), ("size", -1), ("size", "3")],
)
def test_invalid_record(field, value):
    data = hash_member(b"abc").model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        IntegrityRecord.model_validate(data)
