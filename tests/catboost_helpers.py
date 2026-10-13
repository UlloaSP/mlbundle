"""Small native fixtures and CBM byte mutations; never use production models."""

import struct

import catboost as cb
import numpy as np
import pandas as pd

from mlbundle import MemberName
from mlbundle._catboost.flatbuffer import Buffer
from mlbundle.container import read_container
from mlbundle.writing import write_container


def data():
    values = np.random.default_rng(7).normal(size=(45, 3))
    return pd.DataFrame(values, columns=list("abc")), np.arange(45) % 3


def classifier(frame=None, labels=None, **params):
    x, y = data()
    return cb.CatBoostClassifier(
        iterations=4,
        depth=2,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        **params,
    ).fit(x if frame is None else frame, y if labels is None else labels)


def regressor(loss="RMSE", **params):
    x, y = data()
    return cb.CatBoostRegressor(
        iterations=4,
        depth=2,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        loss_function=loss,
        **params,
    ).fit(x, y + 1.0)


def rewrite(path, change, *, schema=False):
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    del members[MemberName.MANIFEST]
    change(parts, members)
    if schema:
        from conftest import align_schema

        align_schema(parts, members)
    return write_container(parts, members)


def tables(data):
    buffer = Buffer(memoryview(data)[8:])
    core = buffer.table(buffer.indirect(0), 4)
    return core, core.child(1, 19)


def field(data, index, value, fmt="i", *, feature=False):
    _, tree = tables(data)
    target = tree.tables(5, 6, 100_000)[0] if feature else tree
    position = target.field(index, struct.calcsize(fmt))
    assert position is not None
    result = bytearray(data)
    struct.pack_into("<" + fmt, result, position + 8, value)
    return bytes(result)


def vector(data, index, value, fmt="i", *, count=False):
    _, tree = tables(data)
    position = tree.field(index)
    assert position is not None
    start = tree.buffer.indirect(position) + 8
    result = bytearray(data)
    struct.pack_into("<" + fmt, result, start if count else start + 4, value)
    return bytes(result)
