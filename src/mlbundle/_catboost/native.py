"""Numeric CBM preflight before CatBoost's native zero-copy loader sees bytes."""

import math
import struct
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from mlbundle.canonical import canonical_loads
from mlbundle.errors import InvalidModelError
from mlbundle.limits import (
    HYPERPARAMETER_BYTES,
    MANIFEST_BYTES,
    MAX_CLASSES,
    MAX_FEATURES,
    ArchiveLimits,
)

from .budget import structure_budget
from .flatbuffer import Buffer, Table
from .scope import CLASSIFIERS, MAX_DEPTH, MAX_TREES, REGRESSORS


@dataclass
class Snapshot:
    native: dict[str, Any]
    metadata: dict[str, str]
    loss: str
    width: int
    dimension: int


def _features(tree: Table) -> list[dict[str, Any]]:
    records = []
    remaining = MAX_TREES * MAX_DEPTH
    for feature in tree.tables(5, 6, MAX_FEATURES):
        has_nans = feature.scalar(0, "B")
        index, flat = (feature.scalar(i, "i", -1) for i in (1, 2))
        treatment = feature.scalar(5, "b")
        borders = feature.vector(3, "<f4", MAX_TREES * MAX_DEPTH)
        remaining -= len(borders)
        if (
            remaining < 0
            or index != len(records)
            or flat != len(records)
            or has_nans not in (0, 1)
            or treatment not in (0, 1, 2)
            or not 0 <= index < MAX_FEATURES
            or not 0 <= flat < MAX_FEATURES
            or not np.isfinite(borders).all()
            or np.any(borders[1:] <= borders[:-1])
        ):
            raise InvalidModelError("Invalid CatBoost numeric feature record.")
        name = feature.string(4, 4096)
        if len(name) > 1024:
            raise InvalidModelError("CatBoost feature name exceeds the limit.")
        records.append(
            {
                "index": index,
                "flat_index": flat,
                "name": name,
                "has_nans": bool(has_nans),
                "nan_value_treatment": treatment,
                "borders": borders,
            }
        )
    if (
        not records
        or [r["index"] for r in records] != list(range(len(records)))
        or [r["flat_index"] for r in records] != list(range(len(records)))
    ):
        raise InvalidModelError("Invalid CatBoost numeric feature order.")
    return records


def _metadata(core: Table) -> dict[str, str]:
    result = {}
    total = 0
    for item in core.tables(2, 2, 4096):
        key, value = item.string(0, 1024), item.string(1, MANIFEST_BYTES)
        total += len(key.encode()) + len(value.encode())
        if not key or key in result or total > MANIFEST_BYTES:
            raise InvalidModelError("Invalid or oversized CatBoost metadata.")
        result[key] = value
    for key in ("training", "output_options"):
        if key in result:
            canonical_loads(result[key].encode(), max_bytes=MANIFEST_BYTES)
    return result


def _classes(metadata: dict[str, str], count: int) -> None:
    classes = canonical_loads(
        metadata["class_params"].encode(), max_bytes=HYPERPARAMETER_BYTES
    )
    if not isinstance(classes, dict):
        raise InvalidModelError("Invalid CatBoost native class record.")
    mapping, names = classes["class_to_label"], classes["class_names"]
    if (
        not isinstance(mapping, list)
        or len(mapping) != count
        or not isinstance(names, list)
        or len(names) not in (0, count)
    ):
        raise InvalidModelError("Invalid CatBoost native class count.")
    declared = classes["classes_count"]
    if type(declared) is not int or declared not in (0, count):
        raise InvalidModelError("Invalid CatBoost declared class count.")
    if classes.get("class_label_type") not in (
        "Integer",
        "Float",
        "String",
        "Boolean",
    ) or any(
        type(label) not in (int, float) or not math.isfinite(cast(float, label))
        for label in mapping
    ):
        raise InvalidModelError("Invalid CatBoost native class mapping.")


def _loss(metadata: dict[str, str], dimension: int) -> str:
    params = canonical_loads(
        metadata["params"].encode(), max_bytes=HYPERPARAMETER_BYTES, max_depth=16
    )
    if not isinstance(params, dict):
        raise InvalidModelError("Invalid CatBoost native parameters.")
    params = cast(dict[str, Any], params)
    loss = params["loss_function"]["type"]
    if (
        loss not in CLASSIFIERS | REGRESSORS
        or params["tree_learner_options"]["grow_policy"] != "SymmetricTree"
    ):
        raise InvalidModelError("Unsupported CatBoost native loss or tree policy.")
    if loss == "MultiClass":
        if not 2 <= dimension <= MAX_CLASSES:
            raise InvalidModelError("Invalid CatBoost multiclass dimension.")
    elif dimension != 1:
        raise InvalidModelError("CatBoost requires a single target.")
    for metric in [
        params["metrics"]["eval_metric"],
        *params["metrics"]["custom_metrics"],
    ]:
        if metric["type"].startswith("Python"):
            raise InvalidModelError("Custom Python CatBoost metrics are unsupported.")
    if loss in CLASSIFIERS:
        _classes(metadata, dimension if loss == "MultiClass" else 2)
    return loss


def _trees(tree: Table) -> dict[str, Any]:
    dimension = tree.scalar(0, "i")
    if not 1 <= dimension <= MAX_CLASSES:
        raise InvalidModelError("Invalid CatBoost output dimension.")
    depths = tree.vector(2, "<i4", MAX_TREES)
    offsets = tree.vector(3, "<i4", MAX_TREES)
    splits = tree.vector(1, "<i4", MAX_TREES * MAX_DEPTH)
    if (
        not len(depths)
        or np.any(depths < 0)
        or np.any(depths > MAX_DEPTH)
        or len(offsets) != len(depths)
    ):
        raise InvalidModelError("Invalid CatBoost tree count or depth.")
    expected = np.concatenate(([0], np.cumsum(depths[:-1], dtype=np.int64)))
    if not np.array_equal(offsets, expected) or len(splits) != int(depths.sum()):
        raise InvalidModelError("Invalid CatBoost tree split offsets.")
    leaves = int(np.left_shift(np.int64(1), depths).sum())
    values = tree.vector(8, "<f8", ArchiveLimits().max_member_bytes // 8)
    weights = tree.vector(9, "<f8", ArchiveLimits().max_member_bytes // 8)
    if (
        len(values) != leaves * dimension
        or len(weights) != leaves
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise InvalidModelError("Invalid CatBoost leaf values or weights.")
    features = _features(tree)
    border_count = sum(len(feature["borders"]) for feature in features)
    if np.any(splits < 0) or np.any(splits >= border_count):
        raise InvalidModelError("Invalid CatBoost split feature indices.")
    scale, legacy_bias = tree.scalar(14, "d", 1.0), tree.scalar(15, "d", 0.0)
    bias = tree.vector(16, "<f8", MAX_CLASSES)
    if (
        not math.isfinite(scale)
        or not math.isfinite(legacy_bias)
        or len(bias) not in (0, dimension)
        or not np.isfinite(bias).all()
    ):
        raise InvalidModelError("Invalid CatBoost scale or bias.")
    if len(bias) and legacy_bias != 0.0:
        raise InvalidModelError("Ambiguous CatBoost bias records.")
    repacked = tree.vector(17, "<u4", MAX_TREES * MAX_DEPTH)
    if len(repacked) not in (0, len(splits)):
        raise InvalidModelError("Invalid CatBoost repacked split count.")
    return {
        "approx_dimension": dimension,
        "tree_depths": depths,
        "tree_split_offsets": offsets,
        "tree_splits": splits,
        "leaf_values": values,
        "leaf_weights": weights,
        "float_features": features,
        "scale": scale,
        "bias": bias if len(bias) else np.full(dimension, legacy_bias),
        "repacked_bins": repacked,
    }


def decode(data: bytes) -> Snapshot:
    if (
        type(data) is not bytes
        or len(data) > ArchiveLimits().max_member_bytes
        or len(data) < 12
        or data[:4] != b"CBM1"
    ):
        raise InvalidModelError("Expected a bounded native CatBoost CBM model.")
    size = struct.unpack_from("<I", data, 4)[0]
    if size != len(data) - 8:
        raise InvalidModelError(
            "Truncated CBM core or unsupported CatBoost model parts."
        )
    buffer = Buffer(memoryview(data)[8:])
    core = buffer.table(buffer.indirect(0), 4)
    if core.string(0, 256) != "FlabuffersModel_v1" or len(core.vector(3, "<u4", 0)):
        raise InvalidModelError("Unsupported CatBoost core version or model parts.")
    tree = core.child(1, 19)
    # No CTR parser, feature calcer, or non-oblivious traversal reaches native code.
    for field in (4, 6, 7, 10, 11, 12, 13, 18):
        if len(tree.vector(field, "<u4", 0)):
            raise InvalidModelError(
                "Unsupported CatBoost feature or non-symmetric tree state."
            )
    native = _trees(tree)
    structure_budget(native)
    metadata = _metadata(core)
    loss = _loss(metadata, native["approx_dimension"])
    return Snapshot(
        native,
        metadata,
        loss,
        len(native["float_features"]),
        native["approx_dimension"],
    )
