"""Shared native preparation and full round-trip verification."""

from dataclasses import dataclass

from pydantic import JsonValue

from .adapters import Adapter
from .claims import ModelClaims, verify_claims, verify_structure
from .contract import MODEL_MEMBERS, Library, MemberName, SerializationFormat
from .declarations import ExportDeclarations, resolve_labels
from .errors import LimitExceededError
from .limits import ArchiveLimits


@dataclass(frozen=True)
class NativeExport:
    adapter: Adapter
    model: object
    restored: object
    claims: ModelClaims
    structure: dict[str, JsonValue] | None
    members: dict[MemberName, bytes]

    @property
    def library(self) -> Library:
        return self.adapter.library

    @property
    def serialization(self) -> SerializationFormat:
        return MODEL_MEMBERS[self.library][0]


def prepare_native(
    model: object,
    adapter: Adapter,
    declarations: ExportDeclarations,
    limits: ArchiveLimits,
) -> NativeExport:
    model = adapter.prepare_export(model, declarations)
    data = adapter.serialize(model)
    if len(data) > limits.member_bytes(adapter.model_member):
        raise LimitExceededError("Model member size limit exceeded.")
    claims = resolve_labels(adapter.extract_claims(model), declarations)
    structure = adapter.extract_structure(model)
    restored = adapter.deserialize(data)
    verify_claims(
        claims, resolve_labels(adapter.extract_claims(restored), declarations)
    )
    structure_bytes = verify_structure(structure, adapter.extract_structure(restored))
    members = {adapter.model_member: data}
    if structure is not None:
        members[MemberName.STRUCTURE] = structure_bytes
    return NativeExport(adapter, model, restored, claims, structure, members)
