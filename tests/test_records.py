"""Record keys, scalar conversion, schema nullability and output normalization."""

import copy
import math
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import mlbundle
from mlbundle.canonical import canonical_dumps
from mlbundle.container import read_container
from mlbundle.writing import write_container
from record_helpers import (
    ACCEPTED_VALUES,
    REJECTED_VALUES,
    LookupOnly,
    bundle_at,
    categorical_model,
    disagreement_classifier,
    missing_pandas,
    named_pipeline,
    native_decision_model,
    threshold_classifier,
)


@pytest.fixture
def bundle(tmp_path):
    return bundle_at(tmp_path / "record.mlbundle")


@pytest.mark.parametrize("origin", ["model", "generated", "declared", "dataframe"])
@pytest.mark.parametrize("strings", [False, True])
def test_keys_order_and_ignored_extras(origin, strings, tmp_path):
    loaded = bundle_at(tmp_path / "keys.mlbundle", origin=origin, names=["12", "0"])
    keys = ["12", "0"] if origin == "model" else (["0", "1"] if strings else [0, 1])
    record = {"ignored": object(), keys[1]: 1, keys[0]: 0}
    assert loaded.row_from_record(record) == [0.0, 1.0]
    assert loaded.predict_record(record)[0].probabilities == (1.0, 0.0)


@pytest.mark.parametrize("method", ["row_from_record", "predict_record"])
@pytest.mark.parametrize("record", [None, [], [0, 1], "0", 1])
def test_mapping_required(bundle, method, record):
    with pytest.raises(mlbundle.PredictionInputError, match="mapping"):
        getattr(bundle, method)(record)


@pytest.mark.parametrize("record", [{0: 0, "0": 0, 1: 1}, {0: 0, 1: 1, "1": 1}])
def test_duplicate_position_forms(bundle, record):
    with pytest.raises(mlbundle.PredictionInputError, match="both integer and string"):
        bundle.predict_record(record)


@pytest.mark.parametrize("key", ["00", "+0", "-0", " 0", "0 ", "0.0", "\u0660"])
def test_noncanonical_position_is_ignored_and_missing(bundle, key):
    with pytest.raises(mlbundle.PredictionInputError, match='Missing features: "0"'):
        bundle.row_from_record({key: 0, "1": 1})


def test_lookup_only_mapping(bundle):
    assert bundle.predict_record(LookupOnly())[0].probabilities == (1.0, 0.0)


@pytest.mark.parametrize("value,expected", ACCEPTED_VALUES)
def test_accepted_values(bundle, value, expected):
    assert bundle.row_from_record({"0": value, "1": 1}) == [expected, 1.0]


@pytest.mark.parametrize("value,reason", REJECTED_VALUES)
def test_rejected_values_without_echo(bundle, value, reason):
    with pytest.raises(mlbundle.PredictionInputError, match=reason) as error:
        bundle.predict_record({"0": value, "1": 1})
    assert str(error.value).startswith('Feature "0": ')
    assert "private text" not in str(error.value)


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("row", [[0, 1], [1, 0]])
def test_outputs_equal_array_prediction_and_frozen(kind, outputs, row, tmp_path):
    loaded = bundle_at(tmp_path / "outputs.mlbundle", kind=kind, outputs=outputs)
    record = {str(index): str(value) for index, value in enumerate(row)}
    row = loaded.row_from_record(record)
    results = loaded.predict_record(record)
    assert isinstance(results, tuple) and len(results) == outputs
    if kind == "classifier":
        raw = loaded.predict_proba([row])
        matrices = [raw] if outputs == 1 else raw
        native = loaded.native_model.predict([row])[0]
        for output, (result, matrix) in enumerate(zip(results, matrices, strict=True)):
            assert (
                result.output == output and result.kind == mlbundle.ModelKind.CLASSIFIER
            )
            assert result.labels == tuple(loaded.schema["reports"][output]["labels"])
            assert result.probabilities == tuple(matrix[0]) and result.value is None
            assert result.predicted_label == str(
                native if outputs == 1 else native[output]
            )
    else:
        raw = loaded.predict([row])
        values = [raw[0]] if outputs == 1 else raw[0]
        for output, (result, value) in enumerate(zip(results, values, strict=True)):
            assert (
                result.output == output and result.kind == mlbundle.ModelKind.REGRESSOR
            )
            assert (
                result.labels is result.probabilities is None and result.value == value
            )
            assert result.predicted_label is None
    assert isinstance(results[0], mlbundle.OutputPrediction)
    assert not hasattr(loaded, "predict_class_indices")
    with pytest.raises(FrozenInstanceError):
        results[0].output = 99


@pytest.mark.parametrize("field_kind", ["number", "onehot-category", "series"])
@pytest.mark.parametrize("required", [True, False])
def test_none_uses_parent_required(field_kind, required, tmp_path):
    path = tmp_path / "nullable.mlbundle"
    names = (
        ["color__red", "color__blue"] if field_kind == "onehot-category" else ["a", "b"]
    )
    loaded = bundle_at(path, origin="model", names=names)
    manifest, members = read_container(path)
    schema = copy.deepcopy(loaded.schema)
    parent = schema["fields"][0]
    parent["required"] = required
    if field_kind == "series":
        parent.pop("step")
        parent.update(
            kind="series",
            columns=[
                {
                    "id": axis,
                    "kind": "number",
                    "label": axis,
                    "required": not required,
                    "step": 1.0,
                }
                for axis in ("field1", "field2")
            ],
        )
    members[mlbundle.MemberName.SCHEMA] = canonical_dumps(schema)
    del members[mlbundle.MemberName.MANIFEST]
    loaded = mlbundle.load(write_container(manifest, members))
    record = {names[0]: None, names[1]: 1}
    if required:
        with pytest.raises(
            mlbundle.PredictionInputError, match="required feature cannot be None"
        ):
            loaded.predict_record(record)
    else:
        assert math.isnan(loaded.row_from_record(record)[0])
        assert loaded.predict_record(record)[0].probabilities is not None


def test_missing_keys_and_untrusted_names_are_bounded(tmp_path):
    names = [f"private\n\x1b[31m{index}" + "\x1b" * 70 for index in range(12)]
    frame = pd.DataFrame([[0] * 12, [1] * 12], columns=names)
    path = tmp_path / "escaped.mlbundle"
    mlbundle.export(DecisionTreeClassifier().fit(frame, [0, 1]), path)
    loaded = mlbundle.load(path)
    with pytest.raises(mlbundle.PredictionInputError) as error:
        loaded.row_from_record({})
    assert "7 more" in str(error.value) and "\\n\\u001b" in str(error.value)
    assert len(str(error.value)) < 700
    assert "\n" not in str(error.value) and "\x1b" not in str(error.value)
    with pytest.raises(mlbundle.PredictionInputError) as error:
        loaded.row_from_record(dict.fromkeys(names, "secret value"))
    assert "\\n\\u001b" in str(error.value) and "secret value" not in str(error.value)


def test_array_path_stays_strict_and_schema_is_required(bundle):
    with pytest.raises(mlbundle.InvalidModelError):
        bundle.predict_proba([["0", "1"]])
    with pytest.raises(mlbundle.PredictionInputError, match="required"):
        bundle.row_from_record({"0": None, "1": 1})
    with pytest.raises(mlbundle.PredictionInputError, match="validated schema"):
        replace(bundle, schema=None).row_from_record({"0": 0, "1": 1})


@pytest.mark.parametrize("kind", ["classifier", "regressor", "composite"])
@pytest.mark.parametrize("position", [0, 1])
def test_named_pipeline_record(kind, position, tmp_path, monkeypatch):
    from mlbundle._sklearn import record_input

    model, frame = named_pipeline(kind)
    path = tmp_path / "named-pipeline.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    record = {name: str(value) for name, value in frame.iloc[position].items()}
    result = loaded.predict_record(record)[0]
    if kind == "regressor":
        assert result.value == loaded.predict(frame.iloc[position : position + 1])[0]
        assert result.predicted_label is None
    else:
        assert result.probabilities == tuple(
            loaded.predict_proba(frame.iloc[position : position + 1])[0]
        )
        assert result.predicted_label == str(
            model.predict(frame.iloc[position : position + 1])[0]
        )
    monkeypatch.setattr(record_input, "import_module", missing_pandas)
    with pytest.raises(mlbundle.MissingOptionalDependencyError) as error:
        loaded.predict_record({"a": 0, "b": 1})
    assert error.value.extra == "schema"


def test_named_record_without_pandas(tmp_path, monkeypatch):
    from mlbundle._sklearn import record_input

    loaded = bundle_at(tmp_path / "named.mlbundle", origin="model")

    monkeypatch.setattr(record_input, "import_module", pytest.fail)
    assert loaded.row_from_record({"age": "0", "active": "1"}) == [0.0, 1.0]
    assert loaded.predict_record({"age": "0", "active": "1"})[0].probabilities == (
        1.0,
        0.0,
    )


@pytest.mark.parametrize("composite", [False, True])
def test_numeric_record_preserves_xgboost_categories(composite, tmp_path):
    model = categorical_model(composite)
    path = tmp_path / "categorical.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert loaded.predict_record({"category": "0", "amount": "0.0"})[
        0
    ].probabilities == tuple(loaded.predict_proba([[0.0, 0.0]])[0])


@pytest.mark.parametrize("library", ["xgboost", "catboost", "onnx"])
def test_native_record_labels_on_several_rows(library, tmp_path):
    import numpy as np

    from onnx_helpers import ROWS

    native, options = native_decision_model(library)
    path = tmp_path / "decision.mlbundle"
    mlbundle.export(native, path, **options)
    loaded = mlbundle.load(path)
    expected = (
        loaded.native_model.session.run(
            ["label"], {"X": np.asarray(ROWS, dtype="float32")}
        )[0]
        if library == "onnx"
        else loaded.native_model.predict(ROWS).reshape(-1)
    )
    for row, label in zip(ROWS, expected, strict=True):
        result = loaded.predict_record(
            {str(index): value for index, value in enumerate(row)}
        )[0]
        assert result.predicted_label == (
            ["cold", "warm"][int(label)] if library == "xgboost" else str(label)
        )


def test_decisions_can_disagree_with_probabilities(tmp_path):
    for name, model, row in [
        ("threshold", threshold_classifier(), [1]),
        ("svc", disagreement_classifier(), [3.5]),
    ]:
        path = tmp_path / f"{name}.mlbundle"
        mlbundle.export(model, path)
        loaded = mlbundle.load(path)
        result = loaded.predict_record({"0": row[0]})[0]
        assert result.predicted_label == str(model.predict([row])[0])
        assert (
            result.predicted_label
            != result.labels[result.probabilities.index(max(result.probabilities))]
        )
        if name == "threshold":
            assert (
                0.5 < result.probabilities[1] < 0.8
                and result.predicted_label == "negative"
            )


def test_onnx_without_decision_returns_none(tmp_path):
    from onnx_helpers import softmax

    path = tmp_path / "probabilities-only.mlbundle"
    mlbundle.export(softmax(), path)
    assert (
        mlbundle.load(path).predict_record({"0": 0, "1": 1})[0].predicted_label is None
    )


def test_record_invalid_native_decision_is_bounded(bundle, monkeypatch):
    monkeypatch.setattr(bundle.native_model, "predict", lambda _: ["private class"])
    with pytest.raises(mlbundle.InvalidModelError) as error:
        bundle.predict_record({"0": 0, "1": 1})
    assert "private" not in str(error.value) and len(str(error.value)) < 160
