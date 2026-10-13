"""Small deterministic composite fixtures, never owner's model files."""

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from mlbundle.container import read_container


def data():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.uniform(0.2, 2, (24, 3)), columns=list("abc"))
    return frame, (frame.a > 1).astype(int)


def estimator(library="xgboost", regression=False):
    if library == "xgboost":
        return (XGBRegressor if regression else XGBClassifier)(
            n_estimators=2, max_depth=2, n_jobs=1
        )
    return (cb.CatBoostRegressor if regression else cb.CatBoostClassifier)(
        iterations=2, depth=2, thread_count=1, verbose=False, allow_writing_files=False
    )


def pipeline(library="xgboost", regression=False, prefix=True):
    frame, y = data()
    model = Pipeline(
        [
            *(([("scale", StandardScaler())]) if prefix else []),
            ("est", estimator(library, regression)),
        ]
    )
    return model.fit(frame, frame.a if regression else y), frame


def parts(source):
    manifest, members = read_container(source)
    members.pop("manifest.json")
    return manifest.model_dump(mode="json", by_alias=True), members
