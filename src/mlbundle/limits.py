"""Format safety ceilings. Readers and writers may only lower them."""

from pydantic import Field

from ._base import ContractModel
from .contract import MemberName

MIB = 1024 * 1024
MANIFEST_BYTES = 8 * MIB
STRUCTURE_BYTES = 256 * MIB
LOCK_BYTES = MIB
METADATA_BYTES = 64 * 1024
HYPERPARAMETER_BYTES = MIB
JSON_DEPTH = 32
MAX_FEATURES = 100_000
MAX_CLASSES = 100_000
MAX_OUTPUTS = 1024


class ArchiveLimits(ContractModel):
    max_members: int = Field(default=7, ge=2, le=7)
    max_archive_bytes: int = Field(default=512 * MIB, gt=0, le=512 * MIB)
    max_compressed_bytes: int = Field(default=512 * MIB, gt=0, le=512 * MIB)
    max_member_bytes: int = Field(default=512 * MIB, gt=0, le=512 * MIB)
    max_total_bytes: int = Field(default=1024 * MIB, gt=0, le=1024 * MIB)
    max_compression_ratio: int = Field(default=200, gt=0, le=200)
    max_manifest_bytes: int = Field(default=MANIFEST_BYTES, gt=0, le=MANIFEST_BYTES)
    max_structure_bytes: int = Field(default=STRUCTURE_BYTES, gt=0, le=STRUCTURE_BYTES)
    max_schema_bytes: int = Field(default=16 * MIB, gt=0, le=16 * MIB)
    max_samples_bytes: int = Field(default=16 * MIB, gt=0, le=16 * MIB)
    max_lock_bytes: int = Field(default=LOCK_BYTES, gt=0, le=LOCK_BYTES)

    def member_bytes(self, member: MemberName) -> int:
        cap = {
            MemberName.MANIFEST: self.max_manifest_bytes,
            MemberName.STRUCTURE: self.max_structure_bytes,
            MemberName.SCHEMA: self.max_schema_bytes,
            MemberName.SAMPLES: self.max_samples_bytes,
            MemberName.LOCK: self.max_lock_bytes,
        }.get(member, self.max_member_bytes)
        return min(cap, self.max_member_bytes)
