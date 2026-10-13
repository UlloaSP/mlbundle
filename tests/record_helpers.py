"""Small fitted record fixture without input or prediction logic."""

import math
import warnings
from collections.abc import Mapping

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mlbundle


class LookupOnly(Mapping):
    def __getitem__(self, key):
        assert key in (0, 1, "0", "1")
        return {"0": 0, "1": 1}[key]

    def __iter__(self):
        pytest.fail("record iterated")

    def __len__(self) -> int:
        raise AssertionError("record size requested")


def threshold_classifier():
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=2,
        depth=1,
        learning_rate=0.5,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    ).fit([[0], [1], [0], [1]], ["negative", "positive", "negative", "positive"])
    model.set_probability_threshold(0.8)
    return model


def disagreement_classifier():
    from sklearn.svm import SVC

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The `probability` parameter was deprecated",
            category=FutureWarning,
        )
        return SVC(probability=True, random_state=3).fit(
            [[index] for index in range(10)], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        )


def missing_pandas(_name):
    raise ModuleNotFoundError("pandas")


def named_pipeline(kind):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier

    frame = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    estimator = {
        "classifier": DecisionTreeClassifier(),
        "regressor": DecisionTreeRegressor(),
        "composite": XGBClassifier(n_estimators=1, n_jobs=1),
    }[kind]
    return make_pipeline(
        ColumnTransformer([("select", "passthrough", ["a"])]), estimator
    ).fit(frame, [0, 1]), frame


def categorical_model(composite):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    frame = pd.DataFrame(
        {"category": pd.Categorical([0, 1, 0, 1]), "amount": [0.0, 1.0, 2.0, 3.0]}
    )
    estimator = XGBClassifier(enable_categorical=True, n_estimators=1, n_jobs=1)
    model = (
        Pipeline([("identity", "passthrough"), ("leaf", estimator)])
        if composite
        else estimator
    )
    return model.fit(frame, [0, 1, 0, 1])


def native_decision_model(library):
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    from onnx_helpers import ROWS, classifier

    if library == "onnx":
        return classifier(("cold", "warm")), {}
    if library == "xgboost":
        return XGBClassifier(n_estimators=2, n_jobs=1).fit(ROWS, [0, 1, 0, 1]), {
            "class_labels": [["cold", "warm"]]
        }
    model = CatBoostClassifier(
        iterations=2, verbose=False, thread_count=1, allow_writing_files=False
    )
    return model.fit(ROWS, ["cold", "warm"] * 2), {}


def bundle_at(path, *, origin="generated", kind="classifier", outputs=1, names=None):
    frame = pd.DataFrame([[0, 1], [1, 0]], columns=names or ["age", "active"])
    cls = DecisionTreeClassifier if kind == "classifier" else DecisionTreeRegressor
    targets = [0, 1] if outputs == 1 else [[0, 2], [1, 3]]
    model = cls(random_state=7).fit(
        frame if origin == "model" else frame.to_numpy(), targets
    )
    options = {"training_data": frame} if origin == "dataframe" else {}
    if origin == "declared":
        options["feature_names"] = list(frame.columns)
    mlbundle.export(model, path, **options)
    return mlbundle.load(path)


ACCEPTED_VALUES = [
    (0, 0.0),
    (-40, -40.0),
    (1.5, 1.5),
    (True, 1.0),
    (False, 0.0),
    ("40", 40.0),
    ("-40", -40.0),
    ("+40", 40.0),
    ("1.5e3", 1500.0),
    ("1E-3", 0.001),
    (".5", 0.5),
    ("1.", 1.0),
    ("01", 1.0),
    ("true", 1.0),
    ("FALSE", 0.0),
    ("TrUe", 1.0),
    (" \t\r\n\v\f1.5 \t", 1.5),
    ("0" * 64, 0.0),
    (" " * 80 + "1", 1.0),
]


REJECTED_VALUES = [
    ("", "decimal"),
    (" \t", "decimal"),
    ("private text", "decimal"),
    ("NaN", "decimal"),
    ("inf", "decimal"),
    ("0x10", "decimal"),
    ("1_000", "decimal"),
    ("1,2", "decimal"),
    ("1e", "decimal"),
    ("1e+", "decimal"),
    ("1 2", "decimal"),
    ("\u0661", "decimal"),
    ("\u00a01\u00a0", "decimal"),
    ("1\x00", "decimal"),
    ("0" * 65, "64 characters"),
    ("1e309", "finite"),
    (math.inf, "finite"),
    (-math.inf, "finite"),
    (math.nan, "finite"),
    (10**400, "finite"),
    ([], "expected"),
    ({}, "expected"),
    (b"40", "expected"),
    (1j, "expected"),
    (object(), "expected"),
]
