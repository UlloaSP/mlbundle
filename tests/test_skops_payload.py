"""Nested skops bounds: accepted numeric buffers and malformed/rejection cases."""

import io
import warnings
import zipfile

import numpy as np
import pytest
import skops.io as sio
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
