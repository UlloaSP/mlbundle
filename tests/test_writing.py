"""Container round trips and hostile ZIP inputs, including bounded inflation."""

import errno
import hashlib
import io
import random
import stat
import zipfile

import pytest

from mlbundle import (
    ArchiveLimits,
    InvalidContainerError,
    InvalidJSONError,
    InvalidManifestError,
    LimitExceededError,
    Manifest,
    MemberName,
)
from mlbundle.canonical import canonical_dumps
from mlbundle.container import read_container
from mlbundle.integrity import hash_member
from mlbundle.writing import write_container


@pytest.mark.parametrize(
    "library,serialization,member",
    [
        ("sklearn", "skops", MemberName.SKOPS),
        ("xgboost", "ubjson", MemberName.UBJSON),
        ("onnx", "onnx", MemberName.ONNX),
    ],
)
@pytest.mark.parametrize("optional", [False, True])
def test_roundtrip_and_determinism(
    manifest_data, library, serialization, member, optional
):
    manifest_data["model"].update(library=library, serialization=serialization)
    if library == "onnx":
        manifest_data["environment"]["onnx"] = {
            "opsets": [{"domain": "", "version": 21}],
            "ir_version": 10,
            "producer_name": "",
            "producer_version": "",
        }
    members = {member: b"dummy native bytes"}
    if optional:
        for name in (MemberName.SCHEMA, MemberName.STRUCTURE, MemberName.SAMPLES):
            members[name] = b"[]" if name == MemberName.SAMPLES else b"{}"
            manifest_data["references"][name.value[:-5]] = name.value
    manifest_data.pop("integrity")
    bundle = write_container(manifest_data, members)
    manifest, actual = read_container(bundle)
    assert actual == {MemberName.MANIFEST: manifest.to_bytes(), **members}
    assert manifest.integrity == {
        name: hash_member(data) for name, data in members.items()
    }
    assert write_container(manifest, dict(reversed(list(members.items())))) == bundle
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert all(
            e.date_time == (1980, 1, 1, 0, 0, 0)
            and not e.extra
            and not e.comment
            and e.compress_type == 8
            and e.create_system == 3
            and e.external_attr == (stat.S_IFREG | 0o600) << 16
            for e in archive.infolist()
        )


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_write_and_failure(manifest_data, tmp_path, monkeypatch, overwrite):
    target = tmp_path / "model.mlbundle"
    args = (
        Manifest.from_bytes(canonical_dumps(manifest_data)),
        {MemberName.SKOPS: b"model"},
    )
    first = write_container(*args, destination=target)
    assert read_container(str(target)) == read_container(io.BytesIO(first))
    if not overwrite:
        with pytest.raises(FileExistsError):
            write_container(*args, destination=target)
    assert write_container(*args, destination=target, overwrite=True) == first

    def fail(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr("pathlib.Path.replace" if overwrite else "os.link", fail)
    if not overwrite:
        target.unlink()
    with pytest.raises(OSError, match="simulated"):
        write_container(*args, destination=target, overwrite=overwrite)
    assert target.read_bytes() == first if overwrite else not target.exists()
    assert list(tmp_path.glob(".mlbundle-*")) == []


@pytest.mark.parametrize(
    "members,error",
    [
        ({MemberName.MANIFEST: b"{}", MemberName.SKOPS: b"x"}, InvalidContainerError),
        ({"../model.skops": b"x"}, InvalidContainerError),
        ({"mödél.skops": b"x"}, InvalidContainerError),
        ({}, InvalidContainerError),
        ({MemberName.SKOPS: "text"}, TypeError),
    ],
)
def test_invalid_writer_inputs(manifest_data, members, error):
    with pytest.raises(error):
        write_container(manifest_data, members)


@pytest.mark.parametrize(
    "limits,data",
    [
        (ArchiveLimits(max_compression_ratio=2), b"0" * 4096),
        (ArchiveLimits(max_member_bytes=2048), b"x" * 4096),
        (ArchiveLimits(max_total_bytes=19), b"native model bytes"),
        (ArchiveLimits(max_compressed_bytes=1), b"x"),
        (ArchiveLimits(max_archive_bytes=1), b"x"),
        (ArchiveLimits(max_manifest_bytes=1), b"x"),
    ],
)
def test_writer_limits_leave_destination_untouched(
    manifest_data, tmp_path, limits, data
):
    target = tmp_path / "bundle.mlbundle"
    target.write_bytes(b"original")
    with pytest.raises(LimitExceededError):
        write_container(
            manifest_data,
            {MemberName.SKOPS: data},
            destination=target,
            overwrite=True,
            limits=limits,
        )
    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob(".mlbundle-*")) == []


def test_no_overwrite_publication_race(manifest_data, tmp_path, monkeypatch):
    target = tmp_path / "bundle.mlbundle"

    def race(temporary, destination):
        destination.write_bytes(b"concurrent writer")
        raise FileExistsError(destination)

    monkeypatch.setattr("os.link", race)
    with pytest.raises(FileExistsError):
        write_container(manifest_data, {MemberName.SKOPS: b"model"}, destination=target)
    assert target.read_bytes() == b"concurrent writer"
    assert list(tmp_path.glob(".mlbundle-*")) == []


def test_chunked_model_and_empty_model(manifest_data):
    for data in (b"", random.Random(0).randbytes(200_000)):
        bundle = write_container(manifest_data, {MemberName.SKOPS: data})
        assert read_container(bundle)[1][MemberName.SKOPS] == data


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            b"native model bytes",
            "8e4bea2f61364aa5247df42588f6ad89065bcf87dc65a96d0cc185b602b27fe8",
        ),
        (
            random.Random(0).randbytes(200_000),
            "f32df46fe39f515d09403af12a2e64bbee1fe4c579ad4229c14f187d397eb473",
        ),
        (
            b"hello world " * 10_000,
            "2cfab1749bdca7b97cfd6eba990e16054f4b7404ac8a8a2c3a818c95d04150c8",
        ),
        (
            bytes(range(256)) * 500,
            "a2d903caee80044799e2640cd5672e26c2fe13d0a0eddf1a5a9d061f12b9c942",
        ),
        (
            b"0" * 1000,
            "5e92b884593597b822b7a72326b06084fe4c85ee028c164869115895f52dbc41",
        ),
    ],
    ids=["native", "noise", "text", "uniform", "runs"],
)
def test_fixed_archive_bytes_on_both_platforms(manifest_data, data, expected):
    manifest_data["writer_version"] = "0.1.0a1"
    manifest_data["environment"]["packages"]["mlbundle"] = "0.1.0a1"
    bundle = write_container(manifest_data, {MemberName.SKOPS: data})
    assert hashlib.sha256(bundle).hexdigest() == expected


@pytest.mark.parametrize("name", ["schema", "structure", "samples"])
def test_writer_optional_member_errors(manifest_data, name):
    member = MemberName(name + ".json")
    manifest_data["references"][name] = member.value
    for data, limits, error in [
        (b"{}", ArchiveLimits(**{f"max_{name}_bytes": 1}), LimitExceededError),
        (b"{}", ArchiveLimits(max_members=2), LimitExceededError),
        (b'{"a":1,"a":2}', ArchiveLimits(), InvalidJSONError),
    ]:
        with pytest.raises(error):
            write_container(
                manifest_data, {MemberName.SKOPS: b"model", member: data}, limits=limits
            )
    manifest_data["references"][name] = None
    with pytest.raises(InvalidManifestError):
        write_container(manifest_data, {MemberName.SKOPS: b"model", member: b"{}"})


def test_remaining_compression_budget(manifest_data, monkeypatch):
    from mlbundle import writing

    original = writing._compress
    caps = []

    def observed(data, cap):
        caps.append(cap)
        return original(data, cap)

    monkeypatch.setattr(writing, "_compress", observed)
    with pytest.raises(LimitExceededError):
        write_container(
            manifest_data,
            {MemberName.SKOPS: random.Random(0).randbytes(2000)},
            limits=ArchiveLimits(max_archive_bytes=2048, max_compressed_bytes=1000),
        )
    assert len(caps) == 2 and 0 < caps[1] < caps[0] == 1000


@pytest.mark.parametrize(
    "race,fail_copy", [(False, False), (True, False), (False, True)]
)
def test_no_hard_links_fallback(manifest_data, tmp_path, monkeypatch, race, fail_copy):
    import os

    target = tmp_path / "bundle.mlbundle"

    def unsupported(temporary, destination):
        if race:
            destination.write_bytes(b"other writer")
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported)
    if fail_copy:
        original = os.fsync
        calls = []

        def fail_second(fd):
            calls.append(fd)
            if len(calls) == 2:
                raise OSError("simulated copy failure")
            original(fd)

        monkeypatch.setattr(os, "fsync", fail_second)
    if race or fail_copy:
        with pytest.raises(FileExistsError if race else OSError):
            write_container(
                manifest_data, {MemberName.SKOPS: b"model"}, destination=target
            )
        assert target.read_bytes() == b"other writer" if race else not target.exists()
    else:
        result = write_container(
            manifest_data, {MemberName.SKOPS: b"model"}, destination=target
        )
        assert target.read_bytes() == result
        if os.name != "nt":
            assert stat.S_IMODE(target.stat().st_mode) == 0o600
        with pytest.raises(FileExistsError):
            write_container(
                manifest_data, {MemberName.SKOPS: b"model"}, destination=target
            )
    assert not list(tmp_path.glob(".mlbundle-*"))
