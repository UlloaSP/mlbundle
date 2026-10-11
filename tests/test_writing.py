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
            "9ecfa6843fd40cb5e1f0e7dfa045990843916f5d46120a35e71f329eb02bb4bc",
        ),
        (
            random.Random(0).randbytes(200_000),
            "70dce8cf3fde079a3b8afe2de31d45ed94d16d9b0f8f1069d42a46ced962d45b",
        ),
        (
            b"hello world " * 10_000,
            "86ff212d87c2fbe73f06768ccf0ea02feb2bc183f03f076d844a8e63ec2a9d42",
        ),
        (
            bytes(range(256)) * 500,
            "35c93d469d64d6e25a9d0f0283a4f2cef5e54f84b07a6b294d7cdfae96d66d8b",
        ),
        (
            b"0" * 1000,
            "60e744b7d243af13e980de7b083715d638318d090c63f2a0290fe9ee366b9d52",
        ),
    ],
    ids=["native", "noise", "text", "uniform", "runs"],
)
def test_fixed_archive_bytes_on_both_platforms(manifest_data, data, expected):
    manifest_data["writer_version"] = "0.1.0"
    manifest_data["environment"]["packages"]["mlbundle"] = "0.1.0"
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
