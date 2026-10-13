"""CBM preflight rejects unsafe dimensions and malformed bytes before native load."""

import struct
from types import SimpleNamespace

import catboost as cb
import pytest

import mlbundle
from catboost_helpers import classifier, field, rewrite, tables, vector
from mlbundle._catboost import Adapter
from mlbundle._catboost.flatbuffer import Buffer


@pytest.mark.parametrize(
    "damage",
    [
        lambda b: b"",
        lambda b: b"CBM1",
        lambda b: b"random bytes",
        lambda b: b[: len(b) // 2],
        lambda b: b[:-1],
        lambda b: b + b"extra",
        lambda b: b[:4] + struct.pack("<I", 2**32 - 1) + b[8:],
        lambda b: b[:8] + b"\xff" * 4 + b[12:],
        lambda b: vector(b, 2, 100_001, count=True, fmt="I"),
        lambda b: vector(b, 2, 17),
        lambda b: vector(b, 2, -1),
        lambda b: vector(b, 3, 99),
        lambda b: vector(b, 1, 2**31 - 1),
        lambda b: vector(b, 8, float("nan"), "d"),
        lambda b: vector(b, 9, -1.0, "d"),
        lambda b: vector(b, 8, 2**32 - 1, count=True, fmt="I"),
        lambda b: field(b, 0, 100_001),
        lambda b: field(b, 1, 100_001, feature=True),
    ],
)
def test_preflight_damage_never_reaches_catboost(damage, monkeypatch, tmp_path):
    model = classifier()
    native = Adapter().serialize(model)
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model, path)
    bad = damage(native)
    monkeypatch.setattr(
        cb.CatBoost,
        "load_model",
        lambda *a, **k: pytest.fail("Native parser must not run"),
    )
    with pytest.raises(mlbundle.MLBundleError):
        Adapter().deserialize(bad)
    with pytest.raises(mlbundle.MLBundleError):
        mlbundle.load(
            rewrite(path, lambda m, b: b.update({mlbundle.MemberName.CBM: bad}))
        )


def test_native_exception_is_bounded(monkeypatch):
    native = Adapter().serialize(classifier())

    def failure(*a, **k):
        raise cb.CatBoostError("secret native message")

    monkeypatch.setattr(cb.CatBoost, "load_model", failure)
    with pytest.raises(
        mlbundle.InvalidModelError, match="Invalid CatBoost CBM payload"
    ) as error:
        Adapter().deserialize(native)
    assert "secret" not in str(error.value)


def test_byte_cap_and_closed_native_type(monkeypatch):
    import mlbundle._catboost.native as native_module

    native = Adapter().serialize(classifier())
    for bad in [bytearray(native), memoryview(native), "filename.cbm"]:
        with pytest.raises(mlbundle.InvalidModelError):
            Adapter().deserialize(bad)
    monkeypatch.setattr(
        native_module, "ArchiveLimits", lambda: SimpleNamespace(max_member_bytes=1)
    )
    with pytest.raises(mlbundle.InvalidModelError, match="bounded native"):
        Adapter().deserialize(native)


def test_flatbuffer_offsets_fields_and_work_limits():
    for raw, fields in [
        (b"\x00" * 4, 4),
        (struct.pack("<i", -9999), 4),
        (b"\x04\0\0\0\x03\0\x04\0", 4),
    ]:
        buffer = Buffer(memoryview(raw))
        with pytest.raises(mlbundle.InvalidModelError):
            buffer.table(0, fields)
    buffer = Buffer(memoryview(b"\x00" * 16))
    buffer.work = 200_000
    with pytest.raises(mlbundle.InvalidModelError, match="table work"):
        buffer.table(0, 4)


def test_unknown_fields_and_invalid_string_before_native(monkeypatch):
    native = Adapter().serialize(classifier())
    core, _ = tables(native)
    vtable = core.position - core.buffer.number(core.position, "i") + 8
    bad = bytearray(native)
    struct.pack_into("<H", bad, vtable, 14)
    monkeypatch.setattr(
        cb.CatBoost,
        "load_model",
        lambda *a, **kw: pytest.fail("Native parser must not run"),
    )
    with pytest.raises(mlbundle.InvalidModelError, match="table fields"):
        Adapter().deserialize(bytes(bad))
    position = core.field(0)
    start = core.buffer.indirect(position) + 4 + 8
    bad = bytearray(native)
    bad[start] = 255
    with pytest.raises(mlbundle.InvalidModelError, match="payload"):
        Adapter().deserialize(bytes(bad))


def test_reader_never_writes_a_temporary_model(tmp_path, monkeypatch):
    import mlbundle._catboost as adapter_module

    path = tmp_path / "model.mlbundle"
    mlbundle.export(classifier(), path)
    monkeypatch.setattr(
        adapter_module,
        "TemporaryDirectory",
        lambda *a, **k: pytest.fail("Reader must not write files"),
    )
    assert mlbundle.load(path).class_labels == [[0, 1, 2]]


def test_publication_rejects_roundtrip_mismatch(tmp_path, monkeypatch):
    original = Adapter.deserialize

    def changed(self, data):
        model = original(self, data)
        model.set_scale_and_bias(2.0, [0.0] * 3)
        return model

    monkeypatch.setattr(Adapter, "deserialize", changed)
    with pytest.raises(mlbundle.ClaimMismatchError):
        mlbundle.export(classifier(), tmp_path / "bad.mlbundle")
    assert not list(tmp_path.iterdir())


def test_structure_expansion_is_bounded_before_native(monkeypatch):
    from mlbundle._catboost import budget

    native = Adapter().serialize(classifier())
    monkeypatch.setattr(budget, "STRUCTURE_BYTES", 1)
    monkeypatch.setattr(
        cb.CatBoost,
        "load_model",
        lambda *a, **kw: pytest.fail("Native parser must not run"),
    )
    with pytest.raises(mlbundle.LimitExceededError, match="allocation budget"):
        Adapter().deserialize(native)
