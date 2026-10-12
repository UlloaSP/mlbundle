"""Preflight work and storage bounds precede skops audit and NumPy allocation."""

import io
import json
from zipfile import ZipFile

import numpy as np
import pytest
import skops.io as sio
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from mlbundle import InvalidModelError
from mlbundle._sklearn import Adapter
from mlbundle._sklearn.payload import validate_payload
from sklearn_helpers import alter_schema, pack, unpack


def _never(*args, **kwargs):
    pytest.fail("Expensive audit, construction or allocation must not run")


def test_deep_shared_reference_work_is_bounded_before_audit(monkeypatch):
    model = LinearRegression().fit([[0], [1]], [0, 1])

    def change(schema, entries):
        attrs = schema["content"]["content"]
        for index in range(18):
            attrs[f"extra_{index}"] = {
                "__class__": "list",
                "__module__": "builtins",
                "__loader__": "ListNode",
                "__id__": 10000 + index,
                "content": []
                if index == 0
                else [
                    {
                        "__class__": "list",
                        "__module__": "builtins",
                        "__loader__": "CachedNode",
                        "__id__": 9999 + index,
                    }
                ]
                * 2,
            }

    data = alter_schema(sio.dumps(model), change)
    monkeypatch.setattr(sio, "get_untrusted_types", _never)
    monkeypatch.setattr(sio, "loads", _never)
    with pytest.raises(InvalidModelError, match="expanded node visits"):
        Adapter().deserialize(data)


def test_small_shared_references_and_cycles():
    data = sio.dumps([[1], [1]])
    assert validate_payload(data)
    cycle = {
        "__class__": "list",
        "__module__": "builtins",
        "__loader__": "ListNode",
        "__id__": 1,
        "content": [
            {
                "__class__": "list",
                "__module__": "builtins",
                "__loader__": "CachedNode",
                "__id__": 1,
            }
        ],
    }
    with pytest.raises(InvalidModelError, match="Cyclic"):
        validate_payload(pack({"schema.json": json.dumps(cycle).encode()}))


def _malformed_sparse(elements):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, value in {
            "format": np.array(b"csr"),
            "shape": np.array([1, elements]),
            "indptr": np.array([0, elements]),
            "indices": np.array([], dtype=np.int32),
        }.items():
            content = io.BytesIO()
            np.save(content, value, allow_pickle=False)
            archive.writestr(name + ".npy", content.getvalue())
        content = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            content, {"descr": "<f8", "fortran_order": False, "shape": (elements,)}
        )
        archive.writestr("data.npy", content.getvalue())
    return buffer.getvalue()


@pytest.mark.parametrize("elements", [131_072, 1_048_576, 8_388_608])
def test_sparse_declared_array_size_checked_before_allocation(elements, monkeypatch):
    entries = unpack(sio.dumps(csr_matrix([[1.0]])))
    schema = json.loads(entries["schema.json"])
    entries[schema["file"]] = _malformed_sparse(elements)
    monkeypatch.setattr(np, "load", _never)
    monkeypatch.setattr(sio, "loads", _never)
    with pytest.raises(InvalidModelError, match="numeric array dimensions"):
        Adapter().deserialize(pack(entries))


def test_sparse_headers_and_shape_are_bounded():
    data = sio.dumps(csr_matrix([[0.0, 1.0]]))
    assert validate_payload(data)
    entries = unpack(data)
    schema = json.loads(entries["schema.json"])
    inner = unpack(entries[schema["file"]])
    buffer = io.BytesIO()
    np.save(buffer, np.array([1, 2**40]), allow_pickle=False)
    inner["shape.npy"] = buffer.getvalue()
    entries[schema["file"]] = pack(inner)
    with pytest.raises(InvalidModelError, match="declared shape"):
        validate_payload(pack(entries))


def test_repeated_buffer_storage_is_bounded_before_audit(monkeypatch):
    from mlbundle._sklearn import payload

    model = LinearRegression().fit([[0], [1]], [0, 1])

    def change(schema, entries):
        array = schema["content"]["content"]["coef_"]
        schema["content"]["content"]["singular_"] = {
            "__class__": "list",
            "__module__": "builtins",
            "__loader__": "ListNode",
            "content": [{**array, "__id__": 10000 + index} for index in range(40)],
        }

    data = alter_schema(sio.dumps(model), change)
    monkeypatch.setattr(payload, "BUFFER_BYTES", 4096, raising=False)
    monkeypatch.setattr(sio, "get_untrusted_types", _never)
    with pytest.raises(InvalidModelError, match="materialized buffer"):
        Adapter().deserialize(data)


def test_nested_member_count_checked_before_zipfile_objects(monkeypatch):
    from mlbundle._sklearn import archive

    data = pack({str(index): b"" for index in range(10)})
    monkeypatch.setattr(archive, "ZipFile", _never)
    with pytest.raises(InvalidModelError, match="member count"):
        archive.bounded_zip(data, 9)


def test_object_array_shape_cannot_be_ignored():
    # Skops constructs a one-dimensional object array from actual items and
    # otherwise ignores this declared size; the native round-trip check catches it.
    model = LinearRegression().fit([[0], [1]], [0, 1])
    model.rank_ = np.array([1, 2], dtype=object)

    def change(schema, entries):
        dimension = schema["content"]["content"]["rank_"]["shape"]["content"][0]
        dimension["content"] = "1000000000"
        dimension["__id__"] = 900001

    with pytest.raises(InvalidModelError, match="native construction"):
        Adapter().deserialize(alter_schema(sio.dumps(model), change))


def test_numeric_header_is_checked_against_actual_stream_bytes(monkeypatch):
    from mlbundle._sklearn.payload import _array_header

    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<f8", "fortran_order": False, "shape": (100,)}
    )
    buffer.write(b"\0" * 16)
    data = pack({"array.npy": buffer.getvalue()})
    monkeypatch.setattr(np, "load", _never)
    with ZipFile(io.BytesIO(data)) as archive:
        # ZIP readers do not require actual inflated bytes to match file_size.
        size = len(buffer.getvalue()) - 16 + 800
        archive.getinfo("array.npy").file_size = size
        with pytest.raises(InvalidModelError, match="stored size"):
            _array_header(archive, "array.npy", size)
