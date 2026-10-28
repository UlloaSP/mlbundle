# A small random forest regressor. Its structure lists every fitted tree.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlbundle import export

rng = np.random.default_rng(3)
frame = pd.DataFrame(
    {
        "area_m2": rng.uniform(30, 160, 40).round(1),
        "rooms": rng.integers(1, 6, 40),
        "floor": rng.integers(0, 12, 40),
    }
)
frame["price_k"] = (
    frame["area_m2"] * 2.4 + frame["rooms"] * 9 + rng.normal(0, 8, 40)
).round(1)

model = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0)
model.fit(frame.drop(columns="price_k"), frame["price_k"])
# The browser cannot run uv: pass the lock prepared for this page's versions.
export(
    model,
    "regressor.mlbundle",
    training_data=frame,
    target_names=["price_k"],
    environment_lock="pylock.toml",
)

model.feature_importances_.round(3).tolist()
