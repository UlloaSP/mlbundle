"""Container round trips and hostile ZIP inputs, including bounded inflation."""

import io
import stat
import struct

import pytest
from pydantic import ValidationError

from conftest import (
    UNSAFE_NAMES,
    build_zip,
    deflate_boundary,
    descriptor_zip,
    oversized_descriptor,
    patch,
    reverse_directory,
)
from mlbundle import (
    ArchiveLimits,
    IntegrityMismatchError,
    InvalidContainerError,
    InvalidJSONError,
    InvalidManifestError,
    LimitExceededError,
    MemberName,
    UnsupportedFormatVersionError,
)
from mlbundle.canonical import canonical_dumps
from mlbundle.container import read_container
from mlbundle.integrity import hash_member


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_unsafe_names(parts, name):
    with pytest.raises(InvalidContainerError) as error:
        read_container(build_zip([parts[0], (name, b"data")]))
    assert len(str(error.value)) < 160
    assert name not in str(error.value)


@pytest.mark.parametrize("name", ["model.skops", "MODEL.SKOPS"])
def test_duplicates(parts, name):
    with pytest.raises(InvalidContainerError):
        read_container(build_zip([*parts, (name, b"duplicate")]))


@pytest.mark.parametrize(
    "attributes",
    [
        {"external_attr": stat.S_IFLNK << 16},
        {"external_attr": stat.S_IFIFO << 16},
        {"external_attr": stat.S_IFSOCK << 16},
        {"external_attr": stat.S_IFCHR << 16},
        {"external_attr": stat.S_IFBLK << 16},
        {"external_attr": stat.S_IFDIR << 16},
        {"external_attr": 0x10},
        {"comment": b"comment"},
        {"archive_comment": b"comment"},
        {"extra": b"\x55\x54\x00\x00"},
        {"extra": b"\x01"},
        {"extra": b"\x01\x00\x08\x00" + b"\x00" * 8},
    ],
)
def test_nonregular_comments_and_extras(parts, attributes):
    with pytest.raises(InvalidContainerError):
        read_container(build_zip(parts, **attributes))


@pytest.mark.parametrize(
    "central,local,value,fmt,both",
    [
        (8, 6, 1, "H", True),
        (8, 6, 64, "H", True),
        (10, 8, 12, "H", True),
        (34, 0, 1, "H", False),
        (16, 14, 0, "I", True),
        (24, 22, 1, "I", False),
        (24, 22, 0xFFFFFFFF, "I", True),
        (20, 18, 1, "I", False),
        (6, 4, 99, "H", True),
        (12, 10, 1, "H", False),
    ],
)
def test_bad_headers(parts, central, local, value, fmt, *, both):
    with pytest.raises(InvalidContainerError):
        read_container(patch(build_zip(parts), central, local, value, fmt, local=both))


@pytest.mark.parametrize(
    "field,value,fmt",
    [(4, 1, "H"), (6, 1, "H"), (8, 1, "H"), (16, 1, "I"), (20, 1, "H")],
)
def test_bad_end_records(parts, field, value, fmt):
    raw = bytearray(build_zip(parts))
    struct.pack_into("<" + fmt, raw, len(raw) - 22 + field, value)
    with pytest.raises(InvalidContainerError):
        read_container(bytes(raw))


@pytest.mark.parametrize(
    "transform",
    [
        lambda b: b[:-1],
        lambda b: b[:20],
        lambda b: b"not a zip",
        lambda b: b"prefix" + b,
        lambda b: b + b"suffix",
        lambda b: b + b"\x00" * 22,
        lambda b: b.replace(b"native model bytes", b"broken model bytes"),
        lambda b: b.replace(b"model.skops", b"model.skop\x00"),
    ],
)
def test_corruption_and_polyglots(parts, transform):
    with pytest.raises(InvalidContainerError):
        read_container(transform(build_zip(parts)))


@pytest.mark.parametrize(
    "change",
    [
        "missing_manifest",
        "no_model",
        "two_models",
        "wrong_model",
        "unreferenced",
        "absent_reference",
    ],
)
def test_member_set(parts, manifest_data, change):
    if change == "absent_reference":
        manifest_data["references"]["structure"] = "structure.json"
        manifest_data["integrity"]["structure.json"] = hash_member(b"{}").model_dump()
        parts[0] = ("manifest.json", canonical_dumps(manifest_data))
    else:
        parts = {
            "missing_manifest": parts[1:],
            "no_model": parts[:1],
            "two_models": [*parts, ("model.onnx", b"model")],
            "wrong_model": [parts[0], ("model.ubj", parts[1][1])],
            "unreferenced": [*parts, ("schema.json", b"{}")],
        }[change]
    with pytest.raises(InvalidContainerError):
        read_container(build_zip(parts))


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("size", 1)])
def test_integrity_mismatch(parts, manifest_data, field, value):
    manifest_data["integrity"]["model.skops"][field] = value
    parts[0] = ("manifest.json", canonical_dumps(manifest_data))
    with pytest.raises(IntegrityMismatchError, match=r"model\.skops"):
        read_container(build_zip(parts))


@pytest.mark.parametrize(
    "data,error",
    [
        (
            b'{"format_name":"mlbundle","format_version":"99","alien":[]}',
            UnsupportedFormatVersionError,
        ),
        (b'{"format_name":"other","format_version":"99"}', InvalidManifestError),
        (b'{"a":1,"a":2}', InvalidManifestError),
        (b"\xff", InvalidManifestError),
        (b"NaN", InvalidManifestError),
        (b"[" * 33 + b"0" + b"]" * 33, LimitExceededError),
    ],
)
def test_invalid_manifest(parts, data, error):
    with pytest.raises(error):
        read_container(build_zip([("manifest.json", data), *parts[1:]]))


def test_unknown_field_and_noncanonical_json(parts, manifest_data):
    parts[0] = ("manifest.json", b" \n" + parts[0][1] + b" \n")
    assert read_container(build_zip(parts))[1][MemberName.MANIFEST] == parts[0][1]
    manifest_data["unexpected"] = True
    parts[0] = ("manifest.json", canonical_dumps(manifest_data))
    with pytest.raises(InvalidManifestError):
        read_container(build_zip(parts))


@pytest.mark.parametrize(
    "limit,value",
    [
        ("max_archive_bytes", 32),
        ("max_compressed_bytes", 16),
        ("max_member_bytes", 16),
        ("max_manifest_bytes", 16),
        ("max_total_bytes", 16),
    ],
)
def test_size_limits(parts, limit, value):
    limits = ArchiveLimits(**{limit: value})
    with pytest.raises(LimitExceededError):
        read_container(build_zip(parts), limits=limits)


@pytest.mark.parametrize("name", ["schema", "structure", "samples"])
def test_optional_caps_count_and_json(parts, manifest_data, name):
    member = MemberName(name + ".json")
    manifest_data["references"][name] = member.value
    for value, error, limits in [
        (b"{}", LimitExceededError, ArchiveLimits(**{f"max_{name}_bytes": 1})),
        (b"{}", LimitExceededError, ArchiveLimits(max_members=2)),
        (b'{"a":1,"a":2}', InvalidJSONError, ArchiveLimits()),
    ]:
        manifest_data["integrity"][member.value] = hash_member(value).model_dump()
        bundle = build_zip(
            [
                ("manifest.json", canonical_dumps(manifest_data)),
                *[part for part in parts[1:] if part[0] != member],
                (member, value),
            ]
        )
        with pytest.raises(error):
            read_container(bundle, limits=limits)


def test_honest_and_lying_bombs(parts):
    bundle = build_zip(
        [parts[0], ("model.skops", b"0" * 4096), parts[2]], compression=8
    )
    with pytest.raises(LimitExceededError):
        read_container(bundle, limits=ArchiveLimits(max_compression_ratio=10))
    with pytest.raises(InvalidContainerError, match="Actual member size"):
        read_container(patch(bundle, 24, 22, 18))
    raw = bytearray(bundle)
    raw[raw.index(b"PK\x03\x04", 4) + 41] = 7
    with pytest.raises(InvalidContainerError):
        read_container(bytes(raw))
    for truncate in (False, True):
        with pytest.raises(InvalidContainerError):
            read_container(deflate_boundary(bundle, truncate=truncate))


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_data_descriptors(parts, wide, signed):
    bundle = descriptor_zip(parts, wide=wide, signed=signed)
    assert read_container(bundle)[1][MemberName.SKOPS] == parts[1][1]
    with pytest.raises(InvalidContainerError):
        read_container(descriptor_zip(parts, wide=wide, signed=signed, corrupt=True))


def test_descriptor_length_checked_before_copy(parts):
    with pytest.raises(InvalidContainerError):
        read_container(
            oversized_descriptor(parts),
            limits=ArchiveLimits(max_member_bytes=2048, max_total_bytes=2048),
        )


def test_zip64_records(parts, monkeypatch):
    monkeypatch.setattr("zipfile.ZIP64_LIMIT", 1)
    bundle = build_zip(parts)
    assert read_container(bundle)[1][MemberName.SKOPS] == parts[1][1]
    for signature, field, value in [
        (b"PK\x06\x07", 4, 1),
        (b"PK\x06\x07", 16, 2),
        (b"PK\x06\x06", 16, 1),
        (b"PK\x06\x06", 4, 43),
    ]:
        raw = bytearray(bundle)
        struct.pack_into("<I", raw, raw.index(signature) + field, value)
        with pytest.raises(InvalidContainerError):
            read_container(bytes(raw))


def test_total_compressed_size(parts):
    limits = ArchiveLimits(max_compressed_bytes=len(parts[0][1]) + 1)
    with pytest.raises(LimitExceededError):
        read_container(build_zip(parts), limits=limits)
    actual = read_container(reverse_directory(build_zip(parts)))[1]
    assert actual[MemberName.SKOPS] == parts[1][1]


def test_bounded_sources(parts, tmp_path, monkeypatch):
    bundle = build_zip(parts)
    target = tmp_path / "huge.mlbundle"
    target.write_bytes(bundle)

    def forbidden(*args, **kwargs):
        raise AssertionError("path opened before size check")

    monkeypatch.setattr("pathlib.Path.open", forbidden)
    limits = ArchiveLimits(max_archive_bytes=20)
    with pytest.raises(LimitExceededError):
        read_container(target, limits=limits)
    stream = io.BytesIO(bundle)
    with pytest.raises(LimitExceededError):
        read_container(stream, limits=limits)
    assert stream.tell() == 21 and not stream.closed
    with pytest.raises(TypeError):
        read_container(io.StringIO("text"))
    with pytest.raises(ValidationError):
        read_container(
            bundle, limits=ArchiveLimits().model_copy(update={"max_members": 99})
        )
