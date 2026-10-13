"""Small wire-format fixture shared by contract feature tests."""

import io
import struct
import warnings
import zipfile
import zlib

import pytest

from mlbundle import MemberName, __version__
from mlbundle.canonical import canonical_dumps
from mlbundle.integrity import hash_member


def schema_for_manifest(parts):
    model, features = parts["model"], parts["features"]
    reports = []
    label = "Predicted class" if model["kind"] == "classifier" else "Predicted value"
    for index in range(model["n_outputs"]):
        report = {
            "kind": model["kind"],
            "label": label if model["n_outputs"] == 1 else f"{label} {index}",
            "mappedTo": index,
        }
        if model["kind"] == "classifier":
            report.update(
                labels=[str(value) for value in parts["target"]["class_labels"][index]],
                showClassProbabilities=True,
            )
        reports.append(report)
    return {
        "schema_version": "2",
        "source": "default",
        "onehot_separator": "__",
        "fields": [
            {
                "kind": "text",
                "label": name,
                "mappedTo": name if features["origin"] == "model" else index,
                "required": True,
            }
            for index, name in enumerate(features["names"])
        ],
        "reports": reports,
    }


def align_schema(parts, members):
    members[MemberName.SCHEMA] = canonical_dumps(schema_for_manifest(parts))


@pytest.fixture
def manifest_data() -> dict:
    value = {
        "format_name": "mlbundle",
        "format_version": "1",
        "bundle_id": "f0d87d83-8cf9-4cf5-af08-e1c885156513",
        "created_at": "2026-10-11T10:00:00Z",
        "writer_version": __version__,
        "model": {
            "library": "sklearn",
            "estimator_class": "DecisionTreeClassifier",
            "kind": "classifier",
            "serialization": "skops",
            "hyperparameters": {"max_depth": 3, "random_state": 0},
            "n_features": 2,
            "n_classes": [2],
            "n_outputs": 1,
            "state_sha256": None,
        },
        "features": {
            "names": ["a", "b"],
            "dtypes": ["float64", None],
            "model_dtypes": None,
            "origin": "model",
        },
        "target": {"class_labels": [[1, "positive"]], "names": ["outcome"]},
        "environment": {
            "python_version": "3.14.0",
            "python_implementation": "CPython",
            "os": "Windows",
            "os_version": "11",
            "architecture": "AMD64",
            "packages": {"mlbundle": __version__, "scikit-learn": "1.9.1"},
            "onnx": None,
        },
        "integrity": {"model.skops": hash_member(b"native model bytes").model_dump()},
        "references": {"schema": "schema.json", "structure": None, "samples": None},
        "user_metadata": {"name": "example", "metrics": {"accuracy": 0.95}},
    }
    value["integrity"]["schema.json"] = hash_member(
        canonical_dumps(schema_for_manifest(value))
    ).model_dump()
    return value


def build_zip(parts, compression=0, *, archive_comment=b"", **attributes):
    buffer = io.BytesIO()
    with warnings.catch_warnings(record=True), zipfile.ZipFile(buffer, "w") as archive:
        warnings.simplefilter("ignore", UserWarning)
        archive.comment = archive_comment
        for name, data in parts:
            info = zipfile.ZipInfo(str(name))
            info.compress_type = compression
            for key, value in attributes.items():
                setattr(info, key, value)
            archive.writestr(info, data)
    return buffer.getvalue()


@pytest.fixture
def parts(manifest_data):
    return [
        ("manifest.json", canonical_dumps(manifest_data)),
        ("model.skops", b"native model bytes"),
        ("schema.json", canonical_dumps(schema_for_manifest(manifest_data))),
    ]


def patch(data, central_field, local_field, value, fmt="I", *, local=True):
    raw = bytearray(data)
    central = raw.find(b"PK\x01\x02", raw.find(b"PK\x01\x02") + 4)
    offset = struct.unpack_from("<I", raw, central + 42)[0]
    struct.pack_into("<" + fmt, raw, central + central_field, value)
    if local:
        struct.pack_into("<" + fmt, raw, offset + local_field, value)
    return bytes(raw)


def descriptor_zip(parts, *, wide=False, signed=True, corrupt=False):
    local = bytearray()
    central = bytearray()
    version = 45 if wide else 20
    for name, content in parts:
        raw_name = name.encode("ascii")
        crc = zlib.crc32(content)
        extra = struct.pack("<HHQQ", 1, 16, 0, 0) if wide else b""
        size = 0xFFFFFFFF if wide else 0
        offset = len(local)
        local.extend(
            struct.pack(
                "<4s5H3I2H",
                b"PK\x03\x04",
                version,
                8,
                0,
                0,
                0,
                0,
                size,
                size,
                len(raw_name),
                len(extra),
            )
            + raw_name
            + extra
            + content
        )
        local.extend(
            (b"PK\x07\x08" if signed else b"")
            + struct.pack(
                "<IQQ" if wide else "<III",
                0 if corrupt else crc,
                len(content),
                len(content),
            )
        )
        central.extend(
            struct.pack(
                "<4s6H3I5H2I",
                b"PK\x01\x02",
                version,
                version,
                8,
                0,
                0,
                0,
                crc,
                len(content),
                len(content),
                len(raw_name),
                0,
                0,
                0,
                0,
                0,
                offset,
            )
            + raw_name
        )
    return bytes(
        local
        + central
        + struct.pack(
            "<4s4H2IH",
            b"PK\x05\x06",
            0,
            0,
            len(parts),
            len(parts),
            len(central),
            len(local),
            0,
        )
    )


def deflate_boundary(bundle, *, truncate):
    raw = bytearray(bundle)
    directory = struct.unpack_from("<I", raw, len(raw) - 6)[0]
    central = raw.index(b"PK\x01\x02", directory + 4)
    local = struct.unpack_from("<I", raw, central + 42)[0]
    size = struct.unpack_from("<I", raw, central + 20)[0]
    shift = -1 if truncate else 1
    struct.pack_into("<I", raw, central + 20, size + shift)
    struct.pack_into("<I", raw, local + 18, size + shift)
    if truncate:
        del raw[directory - 1]
    else:
        raw[directory:directory] = b"\x00"
    struct.pack_into("<I", raw, len(raw) - 6, directory + shift)
    return bytes(raw)


def reverse_directory(bundle):
    directory = struct.unpack_from("<I", bundle, len(bundle) - 6)[0]
    position = directory
    records = []
    while bundle[position : position + 4] == b"PK\x01\x02":
        name, extra, comment = struct.unpack_from("<3H", bundle, position + 28)
        end = position + 46 + name + extra + comment
        records.append(bundle[position:end])
        position = end
    return bundle[:directory] + b"".join(reversed(records)) + bundle[position:]


UNSAFE_NAMES = [
    "..",
    "../model.skops",
    "dir/model.skops",
    r"dir\model.skops",
    "/model.skops",
    r"\model.skops",
    "C:model.skops",
    r"C:\model.skops",
    "unknown",
    " model.skops",
    "model.skops ",
    ".model.skops",
    "model.skops.",
    "model.skops/",
    "MODEL.SKOPS",
    "x" * 1000,
    "m\u00f6d\u00e9l.skops",
]


class BoundedSliceBytes(bytes):
    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is not None and key.stop is not None:
            assert key.stop - key.start <= 64, "large ZIP metadata slice"
        return super().__getitem__(key)


def oversized_descriptor(parts):
    raw = bytearray(descriptor_zip(parts))
    directory = struct.unpack_from("<I", raw, len(raw) - 6)[0]
    raw[directory:directory] = b"PK\x07\x08" + b"x" * 4092
    struct.pack_into("<I", raw, len(raw) - 6, directory + 4096)
    return BoundedSliceBytes(raw)


@pytest.fixture
def xgb_data():
    from sklearn.datasets import load_iris

    return load_iris(return_X_y=True)
