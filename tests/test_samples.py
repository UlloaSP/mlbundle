"""Explicit sample rows, privacy, conversion, limits and lying bundles."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle import InvalidSamplesError, MemberName
from mlbundle.canonical import canonical_dumps, canonical_loads
from mlbundle.container import read_container
from mlbundle.writing import write_container


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "number": [np.float32(12345.25), np.nan, np.inf],
            "count": pd.Series([np.int64(987654321), pd.NA, 3], dtype="Int64"),
            "text": ["ROW_SENTINEL_ALPHA", "ROW_SENTINEL_BETA", None],
            "category": pd.Categorical(
                ["SCHEMA_CATEGORY_ALPHA", "SCHEMA_CATEGORY_BETA", None],
                categories=[
                    "SCHEMA_CATEGORY_ALPHA",
                    "SCHEMA_CATEGORY_BETA",
                    "SCHEMA_UNUSED_GAMMA",
                ],
            ),
            "active": pd.Series([np.bool_(True), False, pd.NA], dtype="boolean"),
            "date": [
                pd.Timestamp("2031-09-17T12:34:56"),
                pd.NaT,
                pd.Timestamp("2031-09-18"),
            ],
            "target": ["TARGET_SENTINEL"] * 3,
        },
        index=pd.Index([0, 1, 2], name="INDEX_SENTINEL"),
    )


@pytest.fixture
def model(data):
    # Only category reaches the estimator, keeping row sentinels out of native storage.
    from sklearn.preprocessing import OneHotEncoder

    return Pipeline(
        [
            (
                "select",
                ColumnTransformer(
                    [
                        (
                            "category",
                            OneHotEncoder(handle_unknown="ignore", dtype="float64"),
                            ["category"],
                        )
                    ]
                ),
            ),
            ("tree", DecisionTreeClassifier()),
        ]
    ).fit(data.drop(columns="target"), [0, 1, 0])


def test_privacy_without_samples(data, model, tmp_path):
    path = tmp_path / "private.mlbundle"
    view = mlbundle.export(model, path, training_data=data, target_name="target")
    _, members = read_container(path)
    assert MemberName.SAMPLES not in members and view.samples is None
    contents = b"\n".join(members.values())
    for private in [
        b"ROW_SENTINEL_ALPHA",
        b"ROW_SENTINEL_BETA",
        b"TARGET_SENTINEL",
        b"INDEX_SENTINEL",
        b"2031-09-17",
        b"12345.25",
        b"987654321",
    ]:
        assert private not in contents
    # Category values are explicitly allowed schema information, including unused categories.
    schema_bytes = members[MemberName.SCHEMA]
    assert (
        b"SCHEMA_CATEGORY_ALPHA" in schema_bytes
        and b"SCHEMA_CATEGORY_BETA" in schema_bytes
    )
    number = view.schema["fields"][0]
    assert b"SCHEMA_UNUSED_GAMMA" in schema_bytes
    assert set(number) == {"kind", "label", "mappedTo", "required", "step"}
    assert number["required"] is False


def test_exact_sample_selection_and_conversion(data, model, tmp_path):
    path = tmp_path / "samples.mlbundle"
    view = mlbundle.export(
        model, path, training_data=data, target_name="target", sample_rows=2
    )
    expected = [
        {
            "number": 12345.25,
            "count": 987654321,
            "text": "ROW_SENTINEL_ALPHA",
            "category": "SCHEMA_CATEGORY_ALPHA",
            "active": True,
            "date": "2031-09-17T12:34:56",
        },
        {
            "number": None,
            "count": None,
            "text": "ROW_SENTINEL_BETA",
            "category": "SCHEMA_CATEGORY_BETA",
            "active": False,
            "date": None,
        },
    ]
    assert (
        view.samples
        == mlbundle.inspect(path).samples
        == mlbundle.load(path).samples
        == expected
    )
    _, members = read_container(path)
    assert members[MemberName.SAMPLES] == canonical_dumps(expected)
    assert set(view.members) == {
        MemberName.MANIFEST,
        MemberName.SKOPS,
        MemberName.SCHEMA,
        MemberName.SAMPLES,
    }
    # Include the final row only when the explicitly selected count is three.
    full = mlbundle.export(
        model,
        tmp_path / "full.mlbundle",
        training_data=data,
        target_name="target",
        sample_rows=3,
    )
    assert full.samples[-1]["number"] == {"$float": "+inf"}
    assert full.samples[-1]["active"] is None
    assert all(set(row) == set(data.columns) - {"target"} for row in full.samples)


def test_schema_structure_samples_coexist(tmp_path):
    frame = pd.DataFrame({"a": [0, 1], "target": [False, True]})
    model = DecisionTreeClassifier().fit(frame[["a"]], frame.target)
    view = mlbundle.export(
        model,
        tmp_path / "all.mlbundle",
        training_data=frame,
        target_name="target",
        sample_rows=1,
    )
    assert len(view.members) == 5
    assert view.samples == [{"a": 0}]
    assert mlbundle.load(tmp_path / "all.mlbundle").structure is not None


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1", 4])
def test_invalid_sample_count(count, data, model, tmp_path):
    path = tmp_path / "never.mlbundle"
    with pytest.raises(InvalidSamplesError, match="positive"):
        mlbundle.export(
            model, path, training_data=data, target_name="target", sample_rows=count
        )
    assert not path.exists()


def test_samples_require_frame_and_respect_byte_limit(data, model, tmp_path):
    path = tmp_path / "never.mlbundle"
    with pytest.raises(InvalidSamplesError, match="requires training_data"):
        mlbundle.export(model, path, sample_rows=1)
    with pytest.raises(mlbundle.LimitExceededError):
        mlbundle.export(
            model,
            path,
            training_data=data,
            target_name="target",
            sample_rows=2,
            limits=mlbundle.ArchiveLimits(max_samples_bytes=20),
        )
    assert not path.exists()


@pytest.mark.parametrize("value", [b"bytes", complex(1, 2), pd.Timedelta(days=1)])
def test_unsupported_sample_values(value, data, model, tmp_path):
    data["text"] = pd.Series([value] * 3, dtype=object)
    with pytest.raises(InvalidSamplesError):
        mlbundle.export(
            model,
            tmp_path / "never.mlbundle",
            training_data=data,
            target_name="target",
            sample_rows=1,
        )


def test_nested_dates_scalars_and_sample_depth(tmp_path):
    from datetime import UTC, date, datetime

    from mlbundle.samples import sample_bytes

    row = {
        "pair": (
            np.int32(1),
            {
                "when": np.datetime64("2026-10-01"),
                "stamp": datetime(2026, 10, 1, tzinfo=UTC),
                "date": date(2026, 10, 1),
            },
        )
    }
    frame = pd.DataFrame([row])
    assert canonical_loads(sample_bytes(frame, 1, 4096)) == [
        {
            "pair": [
                1,
                {
                    "when": "2026-10-01",
                    "stamp": "2026-10-01T00:00:00+00:00",
                    "date": "2026-10-01",
                },
            ]
        }
    ]
    short = pd.DataFrame({"a": ["á", "é"]})
    encoded = sample_bytes(short, 2, 4096)
    assert sample_bytes(short, 2, len(encoded)) == encoded
    with pytest.raises(mlbundle.LimitExceededError):
        sample_bytes(short, 2, len(encoded) - 1)
    value = 1
    for _ in range(32):
        value = [value]
    with pytest.raises(mlbundle.LimitExceededError, match="nesting"):
        sample_bytes(pd.DataFrame({"a": [value]}), 1, 4096)


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows[0].update(extra="value"),
        lambda rows: rows[0].pop("text"),
        lambda rows: rows[0].update(target="TARGET_SENTINEL"),
        lambda rows: rows.__setitem__(0, []),
    ],
)
def test_lying_samples(change, data, model, tmp_path):
    path = tmp_path / "samples.mlbundle"
    mlbundle.export(
        model, path, training_data=data, target_name="target", sample_rows=1
    )
    manifest, members = read_container(path)
    samples = canonical_loads(members[MemberName.SAMPLES])
    change(samples)
    members[MemberName.SAMPLES] = canonical_dumps(samples)
    del members[MemberName.MANIFEST]
    bundle = write_container(manifest, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidSamplesError):
            reader(bundle)


def test_manifest_target_cannot_be_a_sample_feature(data, model, tmp_path):
    path = tmp_path / "samples.mlbundle"
    mlbundle.export(
        model, path, training_data=data, target_name="target", sample_rows=1
    )
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    parts["target"]["name"] = "text"
    del members[MemberName.MANIFEST]
    bundle = write_container(parts, members)
    for reader in (mlbundle.inspect, mlbundle.load):
        with pytest.raises(InvalidSamplesError, match="target"):
            reader(bundle)


@pytest.mark.parametrize(
    "fraction,unit",
    [
        ("123456789012", "ps"),
        ("123456789012345", "fs"),
        ("123456789012345678", "as"),
    ],
)
def test_numpy_timestamp_precision(fraction, unit, data, model, tmp_path):
    expected = "1970-01-01T00:00:00." + fraction
    data["text"] = pd.Series(
        [np.datetime64(expected, unit), np.datetime64("NaT"), None], dtype=object
    )
    path = tmp_path / "precise.mlbundle"
    view = mlbundle.export(
        model, path, training_data=data, target_name="target", sample_rows=2
    )
    assert view.samples[0]["text"] == expected
    assert view.samples[1]["text"] is None
    assert mlbundle.load(path).samples == view.samples
