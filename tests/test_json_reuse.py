"""One checked optional-member decode per operation, with exact comparison."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest

import mlbundle
from mlbundle import canonical, claims, container, model_io
from mlbundle.canonical import canonical_dumps
from mlbundle.contract import MemberName
from mlbundle.schema import reports_for
from mlbundle.training import TrainingDeclaration
from mlbundle.writing import write_container


@pytest.fixture
def native(monkeypatch, manifest_data):
    extracted = mlbundle.Manifest.from_bytes(
        canonical_dumps(manifest_data)
    ).verified_claims()
    structure = {"family": "linear", "coef": [1.0, -0.0]}
    adapter = SimpleNamespace(
        library=mlbundle.Library.SKLEARN,
        model_member=MemberName.SKOPS,
        prepare_export=lambda model, declarations: model,
        serialize=lambda model: b"native",
        deserialize=lambda data: "native",
        extract_claims=lambda model: extracted,
        extract_structure=lambda model: structure,
        predictor=lambda model, claims=None: object(),
    )
    monkeypatch.setattr(model_io, "find_adapter", lambda model: adapter)
    monkeypatch.setattr(model_io, "get_adapter", lambda library: adapter)
    schema = {
        "schema_version": "2",
        "source": "training_data",
        "onehot_separator": "__",
        "fields": [
            {"kind": "text", "label": name, "mappedTo": name, "required": True}
            for name in extracted.feature_names
        ],
        "reports": reports_for(extracted),
    }
    samples = [{"a": "é", "b": "😀"}]
    training = TrainingDeclaration(
        extracted.feature_names,
        ["object", "object"],
        canonical_dumps(schema),
        canonical_dumps(samples),
        mlbundle.FeatureOrigin.MODEL,
    )
    monkeypatch.setattr(model_io, "prepare_training", lambda *args, **kw: training)
    return structure, schema, samples


@pytest.mark.parametrize("operation", ["export", "inspect", "load"])
def test_optional_members_decode_once(native, operation, tmp_path, monkeypatch):
    structure, schema, samples = native
    destination = tmp_path / "model.mlbundle"
    mlbundle.export("native", destination)
    decodes = Counter()
    encodes = []
    decode = container.decode_json
    encode = claims.canonical_dumps

    def counted_decode(data, **kw):
        value = decode(data, **kw)
        decodes[data] += 1
        return value

    def counted_encode(value, **kw):
        if isinstance(value, dict) and value.get("family") == "linear":
            encodes.append(value)
        return encode(value, **kw)

    monkeypatch.setattr(container, "decode_json", counted_decode)
    monkeypatch.setattr(claims, "canonical_dumps", counted_encode)
    if operation == "export":
        result = mlbundle.export("native", destination, overwrite=True)
    else:
        result = getattr(mlbundle, operation)(destination)
    assert decodes == Counter({canonical_dumps(value): 1 for value in native})
    assert len(encodes) == {"export": 2, "inspect": 0, "load": 1}[operation]
    assert result.schema == schema and result.samples == samples
    if operation == "load":
        assert result.structure == structure


@pytest.mark.parametrize(
    "changed",
    [
        None,
        {"family": "linear", "coef": [1, -0.0]},
        {"family": "linear", "coef": [1.0, 0.0]},
    ],
)
def test_reused_comparison_rejects_presence_and_numeric_changes(
    native, changed, tmp_path
):
    destination = tmp_path / "model.mlbundle"
    mlbundle.export("native", destination)
    manifest, members = container.read_container(destination)
    del members[MemberName.MANIFEST]
    members[MemberName.STRUCTURE] = canonical_dumps(changed)
    bundle = write_container(manifest, members)
    with pytest.raises(mlbundle.ClaimMismatchError, match="structure"):
        mlbundle.load(bundle)


def test_noncanonical_structure_keeps_original_hash_and_canonical_comparison(
    native, tmp_path
):
    destination = tmp_path / "model.mlbundle"
    mlbundle.export("native", destination)
    manifest, members = container.read_container(destination)
    del members[MemberName.MANIFEST]
    wire = json.dumps(native[0], indent=2).encode()
    members[MemberName.STRUCTURE] = wire
    bundle = write_container(manifest, members)
    loaded = mlbundle.load(bundle)
    assert loaded.structure == native[0]
    assert container.read_container(bundle)[1][MemberName.STRUCTURE] == wire
    assert loaded.manifest.integrity[MemberName.STRUCTURE].size == len(wire)


def test_canonical_stored_structure_needs_no_comparison_reencoding(
    native, tmp_path, monkeypatch
):
    destination = tmp_path / "model.mlbundle"
    mlbundle.export("native", destination)
    monkeypatch.setattr(
        canonical, "_encode_json", lambda *args: pytest.fail("reencoded")
    )
    assert mlbundle.load(destination).structure == native[0]
