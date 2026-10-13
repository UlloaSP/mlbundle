"""Shared native XGBoost fixtures; tests remain grouped by behavior."""

import struct

import xgboost as xgb

from mlbundle import MemberName
from mlbundle._xgboost.ubjson import Buffer
from mlbundle.container import read_container
from mlbundle.writing import write_container


def classifier(x, y, **params):
    return xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, **params).fit(x, y)


def rewrite(path, change, *, schema=False):
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    del members[MemberName.MANIFEST]
    change(parts, members)
    if schema:
        from conftest import align_schema

        align_schema(parts, members)
    return write_container(parts, members)


def ubjson(value):
    if isinstance(value, Buffer):
        return (
            b"[$"
            + value.marker.encode()
            + b"#L"
            + struct.pack(">q", value.count)
            + bytes(value.data)
        )
    if isinstance(value, dict):
        return (
            b"{"
            + b"".join(
                b"L"
                + struct.pack(">q", len(key.encode()))
                + key.encode()
                + ubjson(item)
                for key, item in value.items()
            )
            + b"}"
        )
    if isinstance(value, list):
        return b"[" + b"".join(ubjson(item) for item in value) + b"]"
    if isinstance(value, str):
        return b"SL" + struct.pack(">q", len(value.encode())) + value.encode()
    return ubjson_scalar(value)


def ubjson_scalar(value):
    if type(value) is bool:
        return b"T" if value else b"F"
    if type(value) is int:
        return b"L" + struct.pack(">q", value)
    if type(value) is float:
        return b"D" + struct.pack(">d", value)
    assert value is None
    return b"Z"
