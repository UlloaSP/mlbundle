"""MLP solver state round trips faithfully and remains closed and committed."""

import numpy as np
import pytest
import skops.io as sio
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlbundle
from mlbundle._sklearn import Adapter
from mlbundle._sklearn.payload import validate_payload
from sklearn_helpers import alter_schema, rewrite

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*:sklearn.exceptions.ConvergenceWarning"
)


def fitted(solver="adam", classifier=True, pipeline=False, early_stopping=False):
    rows = np.random.default_rng(7).normal(size=(40, 3))
    target = (rows[:, 0] > 0).astype(int) if classifier else rows[:, 0] + rows[:, 1]
    estimator = (MLPClassifier if classifier else MLPRegressor)(
        hidden_layer_sizes=(4, 2),
        solver=solver,
        max_iter=5,
        random_state=7,
        early_stopping=early_stopping,
    )
    model = (
        Pipeline([("scale", StandardScaler()), ("model", estimator)])
        if pipeline
        else estimator
    )
    model.fit(rows, target)
    return model, rows


@pytest.mark.parametrize("solver", ["adam", "sgd", "lbfgs"])
@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("pipeline", [False, True])
def test_mlp_all_solvers_roundtrip(solver, classifier, pipeline, tmp_path):
    model, rows = fitted(solver, classifier, pipeline)
    path = tmp_path / "mlp.mlbundle"
    before = validate_payload(sio.dumps(model))
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    method = "predict_proba" if classifier else "predict"
    assert np.array_equal(
        np.asarray(getattr(loaded, method)(rows)).squeeze(),
        getattr(model, method)(rows).squeeze(),
    )
    assert loaded.structure is None
    assert loaded.manifest.model.state_sha256 is not None
    assert validate_payload(sio.dumps(model)) == before


@pytest.mark.parametrize("solver", ["adam", "sgd"])
def test_mlp_early_stopping_state_roundtrip(solver, tmp_path):
    model, rows = fitted(solver, early_stopping=True)
    path = tmp_path / "early.mlbundle"
    mlbundle.export(model, path)
    loaded = mlbundle.load(path)
    assert np.array_equal(loaded.predict_proba(rows), model.predict_proba(rows))
    assert loaded.native_model.validation_scores_ == model.validation_scores_


@pytest.mark.parametrize("solver", ["adam", "sgd"])
def test_unknown_optimizer_state_rejected_on_both_paths(solver, tmp_path):
    model, _ = fitted(solver)
    model._optimizer.extra_ = 1
    for action in [
        lambda: mlbundle.export(model, tmp_path / "bad.mlbundle"),
        lambda: Adapter().deserialize(sio.dumps(model)),
    ]:
        with pytest.raises(
            mlbundle.InvalidModelError, match="Unknown sklearn estimator state"
        ):
            action()


@pytest.mark.parametrize("solver", ["adam", "sgd"])
def test_optimizer_load_bypasses_constructor(solver, monkeypatch):
    model, _ = fitted(solver)
    optimizer = model._optimizer
    assert type(optimizer).__new__ is object.__new__
    assert not hasattr(optimizer, "__setstate__")
    assert all(
        "__reduce__" not in vars(cls) and "__reduce_ex__" not in vars(cls)
        for cls in type(optimizer).__mro__[:-1]
    )
    payload = sio.dumps(model)
    monkeypatch.setattr(
        type(optimizer), "__init__", lambda *a, **kw: pytest.fail("constructor called")
    )
    restored = Adapter().deserialize(payload)
    assert type(restored._optimizer) is type(optimizer)
    assert vars(restored._optimizer).keys() == vars(optimizer).keys()


@pytest.mark.parametrize("solver", ["adam", "sgd"])
@pytest.mark.parametrize(
    "loader", ["ConstructorFromReduceNode", "ListNode", "TypeNode"]
)
def test_optimizer_cannot_select_another_loader(solver, loader, monkeypatch):
    model, _ = fitted(solver)

    def change(schema, entries):
        optimizer = schema["content"]["content"]["_optimizer"]
        optimizer["__loader__"] = loader

    payload = alter_schema(sio.dumps(model), change)
    monkeypatch.setattr(sio, "loads", lambda *a, **kw: pytest.fail("skops called"))
    with pytest.raises(mlbundle.InvalidModelError, match="optimizer loader"):
        Adapter().deserialize(payload)


@pytest.mark.parametrize(
    "attribute", ["coefs_", "intercepts_", "loss_curve_", "optimizer"]
)
def test_changed_mlp_state_rejected_by_commitment(attribute, tmp_path):
    model, _ = fitted()
    path = tmp_path / "original.mlbundle"
    mlbundle.export(model, path)
    if attribute == "optimizer":
        model._optimizer.ms[0][0, 0] += 1
    elif attribute == "loss_curve_":
        model.loss_curve_[0] += 1
    else:
        getattr(model, attribute)[0].flat[0] += 1
    hostile = rewrite(path, sio.dumps(model))
    with pytest.raises(mlbundle.ClaimMismatchError, match="state_sha256"):
        mlbundle.load(hostile)


@pytest.mark.parametrize(
    "solver,attribute",
    [
        ("adam", name)
        for name in (
            "beta_1",
            "beta_2",
            "epsilon",
            "learning_rate",
            "learning_rate_init",
            "ms",
            "t",
            "vs",
        )
    ]
    + [
        ("sgd", name)
        for name in (
            "learning_rate",
            "learning_rate_init",
            "lr_schedule",
            "momentum",
            "nesterov",
            "power_t",
            "velocities",
        )
    ],
)
def test_every_optimizer_attribute_is_committed(solver, attribute, tmp_path):
    model, _ = fitted(solver)
    path = tmp_path / "optimizer.mlbundle"
    mlbundle.export(model, path)
    value = getattr(model._optimizer, attribute)
    if isinstance(value, list):
        value[0].flat[0] += 1
    else:
        changed = (
            "invscaling"
            if isinstance(value, str)
            else not value
            if isinstance(value, bool)
            else value + 1
        )
        setattr(model._optimizer, attribute, changed)
    with pytest.raises(mlbundle.ClaimMismatchError, match="state_sha256"):
        mlbundle.load(rewrite(path, sio.dumps(model)))
