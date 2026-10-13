"""Integrity of stored member bytes; this does not authenticate an author."""

import hashlib
import hmac
from typing import Annotated

from pydantic import Field, StringConstraints

from ._base import ContractModel
from .contract import MemberName
from .errors import IntegrityMismatchError
from .limits import MIB


class IntegrityRecord(ContractModel):
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    size: int = Field(ge=0, le=512 * MIB)


def hash_member(data: bytes) -> IntegrityRecord:
    return IntegrityRecord(sha256=hashlib.sha256(data).hexdigest(), size=len(data))


def verify_integrity(
    member: MemberName, data: bytes, expected: IntegrityRecord
) -> None:
    if len(data) != expected.size or not hmac.compare_digest(
        hashlib.sha256(data).hexdigest(), expected.sha256
    ):
        raise IntegrityMismatchError(member.value)
