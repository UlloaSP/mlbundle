"""Nested skops bounds: accepted numeric buffers and malformed/rejection cases."""

import io
import json
import warnings
import zipfile
from copy import deepcopy

import numpy as np
import pytest
import skops.io as sio
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

from mlbundle import InvalidModelError, MLBundleError
from mlbundle._sklearn import Adapter
from mlbundle._sklearn.payload import validate_payload


def _archive(entries, *, compressed=False):
    buffer = io.BytesIO()
    with (
        warnings.catch_warnings(),
        zipfile.ZipFile(
            buffer,
            "w",
            compression=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED,
        ) as archive,
    ):
        warnings.filterwarnings(
            "ignore", message="Duplicate name:.*", category=UserWarning
        )
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


def _array(*, version=(1, 0), shape=(2,), objects=False):
    buffer = io.BytesIO()
    header = {
        "descr": "|O" if objects else "<f8",
        "fortran_order": False,
        "shape": shape,
    }
    if version == (1, 0):
        np.lib.format.write_array_header_1_0(buffer, header)
    elif version == (2, 0):
        np.lib.format.write_array_header_2_0(buffer, header)
    else:
        buffer.write(b"\x93NUMPY\x03\x00")
    buffer.write(b"\0" * 16)
    return buffer.getvalue()


def test_real_skops_and_supported_array_headers():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    validate_payload(sio.dumps(model))
    for version in [(1, 0), (2, 0)]:
        validate_payload(
            _archive([("schema.json", b"{}"), ("array.npy", _array(version=version))])
        )


@pytest.mark.parametrize(
    "entries,compressed",
    [
        ([("schema.json", b"{}"), ("schema.json", b"{}")], False),
        ([("other.json", b"{}")], False),
        ([("schema.json", b"{}"), ("../array.npy", b"bad")], False),
        ([("schema.json", b"{}"), ("raw.bin", b"0" * 100_000)], True),
        ([("schema.json", b" " * (8 * 1024 * 1024 + 1))], False),
        ([("schema.json", b"{}"), ("array.npy", _array(shape=(1_000_000,)))], False),
        ([("schema.json", b"{}"), ("array.npy", _array(objects=True))], False),
        ([("schema.json", b"{}"), ("array.npy", _array(version=(3, 0)))], False),
        ([("schema.json", b'{"x":1,"x":2}')], False),
    ],
)
def test_payload_rejections(entries, compressed):
    with pytest.raises(MLBundleError):
        Adapter().deserialize(_archive(entries, compressed=compressed))


@pytest.mark.parametrize("reference", ["renamed.bin", "missing.bin", 1])
def test_array_references_cannot_bypass_header_bounds(reference):
    schema = {"__loader__": "NdArrayNode", "file": reference}
    payload = _archive(
        [
            ("schema.json", json.dumps(schema).encode()),
            ("renamed.bin", _array(shape=(1_000_000_000,))),
        ]
    )
    with pytest.raises(InvalidModelError):
        Adapter().deserialize(payload)


def test_zero_itemsize_array_is_rejected():
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<U0", "fortran_order": False, "shape": (1_000_000_000,)}
    )
    with pytest.raises(InvalidModelError):
        Adapter().deserialize(
            _archive([("schema.json", b"{}"), ("array.npy", buffer.getvalue())])
        )


@pytest.mark.parametrize(
    "where,value",
    [("outputs", 1_000_000_000), ("outputs", 2), ("node_count", 1_000_000_000)],
)
def test_tree_constructor_bounds_before_skops_load(where, value, monkeypatch):
    model = DecisionTreeClassifier().fit([[0, 0], [0, 1]], [0, 1])
    with zipfile.ZipFile(io.BytesIO(sio.dumps(model))) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    schema = json.loads(entries["schema.json"])
    tree = schema["content"]["content"]["tree_"]
    target = (
        tree["__reduce__"]["args"]["content"][2]
        if where == "outputs"
        else tree["content"]["content"][where]
    )
    target["content"] = str(value)
    entries["schema.json"] = json.dumps(schema).encode()

    def never_load(*args, **kwargs):
        pytest.fail("Native construction must not run")

    monkeypatch.setattr(sio, "loads", never_load)
    with pytest.raises(InvalidModelError):
        Adapter().deserialize(_archive(entries.items()))


@pytest.mark.parametrize("identity", [987654321, "alias", True])
def test_direct_constructor_identity_aliases_are_checked(identity, monkeypatch):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    with zipfile.ZipFile(io.BytesIO(sio.dumps(model))) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    schema = json.loads(entries["schema.json"])
    attributes = schema["content"]["content"]
    args = attributes["tree_"]["__reduce__"]["args"]
    args["__id__"] = identity
    alias = deepcopy(args)
    alias["content"][2]["__id__"] = 987654322
    alias["content"][2]["content"] = "1000000000"
    schema["content"]["content"] = {"earlier_tuple": alias, **attributes}
    entries["schema.json"] = json.dumps(schema).encode()

    def never_load(*args, **kwargs):
        pytest.fail("Aliased native construction must not run")

    monkeypatch.setattr(sio, "loads", never_load)
    with pytest.raises(InvalidModelError):
        Adapter().deserialize(_archive(entries.items()))


@pytest.mark.parametrize(
    "loader", ["ConstructorFromReduceNode", "ObjectNode", "JsonNode"]
)
def test_native_tree_requires_the_validated_tree_loader(loader, monkeypatch):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    with zipfile.ZipFile(io.BytesIO(sio.dumps(model))) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    schema = json.loads(entries["schema.json"])
    schema["content"]["content"]["tree_"]["__loader__"] = loader
    entries["schema.json"] = json.dumps(schema).encode()

    def never_load(*args, **kwargs):
        pytest.fail("Unvalidated native tree loader must not run")

    monkeypatch.setattr(sio, "loads", never_load)
    with pytest.raises(InvalidModelError, match="native tree loader"):
        Adapter().deserialize(_archive(entries.items()))


@pytest.mark.parametrize(
    "alter",
    [
        lambda tree: setattr(tree, "max_depth", tree.max_depth + 1),
        lambda tree: tree.children_left.__setitem__(0, 0),
        lambda tree: tree.feature.__setitem__(0, 999),
        lambda tree: tree.children_left.__setitem__(0, -2),
    ],
)
def test_tree_indices_are_checked_before_prediction(alter):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    alter(model.tree_)
    with pytest.raises(InvalidModelError):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize(
    "model", [DecisionTreeClassifier(), RandomForestClassifier(n_estimators=2)]
)
def test_tree_owning_input_width_cannot_disagree(model):
    model.fit([[0, 0], [0, 1]], [0, 1])
    model.n_features_in_ = 1
    with pytest.raises(InvalidModelError, match="input widths"):
        Adapter().deserialize(sio.dumps(model))


def test_bagging_feature_mapping_is_bound_to_parent_width():
    model = BaggingClassifier(n_estimators=2, max_features=1, random_state=7).fit(
        [[0, 0], [0, 1]], [0, 1]
    )
    Adapter().deserialize(sio.dumps(model))
    model.estimators_features_[0][0] = 99
    with pytest.raises(InvalidModelError, match="feature mappings"):
        Adapter().deserialize(sio.dumps(model))


@pytest.mark.parametrize(
    "model,attribute,value",
    [
        (DecisionTreeClassifier(), "classes_", 1),
        (DecisionTreeClassifier(), "tree_", None),
        (DecisionTreeClassifier(), "n_outputs_", 2),
        (DecisionTreeClassifier(), "n_classes_", 1),
        (Ridge(), "n_features_in_", None),
        (Ridge(), "coef_", 1.0),
    ],
)
def test_malformed_fitted_attributes_have_bounded_errors(model, attribute, value):
    model.fit([[0], [1]], [0, 1])
    setattr(model, attribute, value)
    adapter = Adapter()
    with pytest.raises(MLBundleError):
        restored = adapter.deserialize(sio.dumps(model))
        adapter.extract_claims(restored)
        adapter.extract_structure(restored)
