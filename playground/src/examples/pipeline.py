# An all-scikit-learn Pipeline. Its claims cover every step; version 1 records no
# structure for it, because the final step only sees transformed features.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mlbundle import export

rows = [[0.0, 10.0], [1.0, 12.0], [2.0, 9.0], [3.0, 15.0], [4.0, 11.0]]
target = [1.2, 2.1, 2.9, 4.2, 5.1]
model = make_pipeline(StandardScaler(), SVR(C=3.0)).fit(rows, target)

# Without a frame, names are declared and the schema comes from one default row.
# The browser cannot run uv: pass the lock prepared for this page's versions.
view = export(
    model,
    "pipeline.mlbundle",
    feature_names=["dose_mg", "weight_kg"],
    environment_lock="pylock.toml",
)
{"source": view.schema["source"], "dtypes": view.manifest.features.dtypes}
