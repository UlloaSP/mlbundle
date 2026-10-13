"""Fast adapter paths retain the legacy commitment, checks and native snapshots."""

import hashlib
import io
import json
from zipfile import ZipFile

import numpy as np
import pytest
import skops.io as sio
import xgboost as xgb
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle import InvalidModelError, LimitExceededError
from mlbundle._sklearn import Adapter as SklearnAdapter, state
from mlbundle._sklearn.payload import validate_payload
from mlbundle._sklearn.references import References
from mlbundle._xgboost import Adapter as XGBoostAdapter
from mlbundle._xgboost.state import projection
from mlbundle._xgboost.ubjson import decode
from mlbundle.canonical import canonical_dumps, canonical_value
from mlbundle.declarations import export_declarations
from mlbundle.limits import MANIFEST_BYTES, STRUCTURE_BYTES


def _legacy_digest(data):
    with ZipFile(io.BytesIO(data)) as archive:
        schema = json.loads(archive.read("schema.json"))
        references = References(schema)
        memo, buffers = {}, {}

        def hashed(domain, value):
            return hashlib.sha256(domain + canonical_dumps(value)).hexdigest()

        def digest(value):
            if isinstance(value, dict):
                node = references.resolve(value) if "__loader__" in value else value
                if id(node) in memo:
                    return memo[id(node)]
                content = {}
                for key, child in node.items():
                    if key == "__id__":
                        continue
                    if key == "file":
                        if child not in buffers:
                            buffers[child] = state._buffer_digest(archive.read(child))
                        content[key] = hashed(b"file:", buffers[child])
                    else:
                        content[key] = digest(child)
                result = hashed(b"dict:", content)
                memo[id(node)] = result
                return result
            if isinstance(value, list):
                return hashed(b"list:", [digest(child) for child in value])
            return hashed(b"scalar:", value)

        return digest(schema)


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeClassifier(max_depth=2),
        RandomForestClassifier(n_estimators=3, max_depth=2, random_state=1),
        ExtraTreesClassifier(n_estimators=3, max_depth=2, random_state=1),
        GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=1),
        RandomForestRegressor(n_estimators=3, max_depth=2, random_state=1),
        make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=3, max_depth=2, random_state=1),
        ),
    ],
)
def test_legacy_merkle_bytes_and_round_trip(model):
    x = np.random.default_rng(1).normal(size=(30, 3))
    model.fit(x, np.arange(30) % 2)
    data = sio.dumps(model)
    assert validate_payload(data) == _legacy_digest(data)
    adapter = SklearnAdapter()
    serialized = adapter.serialize(model)
    restored = adapter.deserialize(serialized)
    assert adapter.extract_claims(model) == adapter.extract_claims(restored)
    assert adapter.extract_structure(model) == adapter.extract_structure(restored)


@pytest.mark.parametrize("value", [True, 1, 1.0, 0.0, -0.0, None, "é", "a\n\\b"])
def test_scalar_encoding_preserves_types_and_signed_zero(value):
    data = sio.dumps([value, value])
    assert validate_payload(data) == _legacy_digest(data)


@pytest.mark.parametrize(
    "domain,value",
    [(b"dict:", {'é"': "a" * 64}), (b"list:", ["b" * 64] * 3), (b"file:", "c" * 64)],
)
def test_digest_records_match_canonical_bytes(domain, value):
    assert (
        state._hash(domain, value)
        == hashlib.sha256(domain + canonical_dumps(value)).hexdigest()
    )


def test_digest_record_byte_limit_is_unchanged():
    with pytest.raises(LimitExceededError):
        state._hash(b"list:", ["a" * 64] * (MANIFEST_BYTES // 67 + 1))


def test_digest_dictionary_bound_precedes_encoding(monkeypatch):
    value = {"x": "a" * 64}
    monkeypatch.setattr(state, "MANIFEST_BYTES", len(canonical_dumps(value)) - 1)

    def never_encode(*args, **kwargs):
        pytest.fail("Over-budget digest records must fail before encoding")

    monkeypatch.setattr(state.json, "dumps", never_encode)
    with pytest.raises(LimitExceededError):
        state._hash(b"dict:", value)


@pytest.mark.parametrize("pipeline", [False, True])
def test_sklearn_payload_checked_once_per_native_snapshot(
    pipeline, monkeypatch, tmp_path
):
    import mlbundle._sklearn as sklearn_adapter
    from mlbundle._sklearn import structure

    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=1)
    if pipeline:
        model = make_pipeline(StandardScaler(), model)
    model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    calls, structures = [], []
    validate, extract = sklearn_adapter.validate_payload, structure.extract_structure

    def checked(data):
        calls.append(data)
        return validate(data)

    def extracted(*args, **kwargs):
        structures.append(True)
        return extract(*args, **kwargs)

    monkeypatch.setattr(sklearn_adapter, "validate_payload", checked)
    monkeypatch.setattr(sklearn_adapter, "extract_structure", extracted)
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model, path)
    assert len(calls) == 2
    assert len(structures) == 2
    calls.clear()
    structures.clear()
    mlbundle.load(path)
    assert len(calls) == 2
    assert len(structures) == 1
    # Another public operation must not trust state retained from a prior one.
    model.extra_ = 1
    with pytest.raises(InvalidModelError, match="Unknown sklearn estimator state"):
        mlbundle.export(model, path, overwrite=True)


def test_reused_sklearn_adapter_rechecks_new_payload_and_serialization():
    model = DecisionTreeClassifier(max_depth=2).fit([[0], [1]], [0, 1])
    adapter = SklearnAdapter()
    adapter.serialize(model)
    model.extra_ = 1
    with pytest.raises(InvalidModelError, match="Unknown sklearn estimator state"):
        adapter.serialize(model)
    with pytest.raises(InvalidModelError, match="Unknown sklearn estimator state"):
        adapter.deserialize(sio.dumps(model))


def _xgb_structure(model):
    return canonical_value(
        {
            "family": "xgboost",
            "native_model": projection(
                decode(bytes(model.get_booster().save_raw(raw_format="ubj")))[
                    "learner"
                ]["gradient_booster"]["model"]
            ),
            "trees": [
                json.loads(tree)
                for tree in model.get_booster().get_dump(
                    dump_format="json", with_stats=True
                )
            ],
            "feature_importances": model.feature_importances_,
        },
        max_bytes=STRUCTURE_BYTES,
    )


def test_xgboost_immutable_cache_keys_and_unexposed_values():
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(
        [[0, 1], [1, 0], [2, 1], [3, 0]], [0, 0, 1, 1]
    )
    adapter = XGBoostAdapter()
    prepared = adapter.prepare_export(model, export_declarations(None))
    assert adapter.extract_structure(prepared) == _xgb_structure(prepared)
    structure = adapter.extract_structure(prepared)
    structure["trees"].clear()
    assert adapter.extract_structure(prepared) == _xgb_structure(prepared)
    prepared.importance_type = "weight"
    assert adapter.extract_structure(prepared) == _xgb_structure(prepared)
    prepared.get_booster().feature_names = ["first", "second"]
    assert adapter.extract_structure(prepared) == _xgb_structure(prepared)
    prepared.get_booster().set_attr(extra="bad")
    with pytest.raises(InvalidModelError, match="Unknown XGBoost booster attribute"):
        adapter.extract_structure(prepared)


def test_xgboost_one_structure_and_two_payload_checks_per_export(monkeypatch, tmp_path):
    from mlbundle._xgboost import snapshots

    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    calls, structures = [], []
    validate, extract = snapshots.validate_payload, snapshots.extract_structure_bytes

    def checked(data):
        calls.append(data)
        return validate(data)

    def extracted(*args, **kwargs):
        structures.append(True)
        return extract(*args, **kwargs)

    monkeypatch.setattr(snapshots, "validate_payload", checked)
    monkeypatch.setattr(snapshots, "extract_structure_bytes", extracted)
    path = tmp_path / "model.mlbundle"
    mlbundle.export(model, path)
    assert len(calls) == 2  # Original unbound bytes and copied bound bytes.
    assert len(structures) == 1
    calls.clear()
    structures.clear()
    mlbundle.load(path)
    assert len(calls) == 1
    assert len(structures) == 1
