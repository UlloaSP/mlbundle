# mlbundle rejects unsupported models on the training machine, before writing anything.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from mlbundle import MLBundleError, export

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
labels = ["cold", "warm", "cold", "warm"]


class CustomTree(DecisionTreeClassifier):
    """A user subclass: not on mlbundle's closed trust list."""


attempts = {
    "custom subclass": lambda: export(
        CustomTree().fit(rows, labels), "custom.mlbundle"
    ),
    "HistGradientBoosting": lambda: export(
        HistGradientBoostingClassifier(max_iter=3).fit(rows * 5, labels * 5),
        "hist.mlbundle",
    ),
    "samples without a frame": lambda: export(
        DecisionTreeClassifier().fit(rows, labels), "sampled.mlbundle", sample_rows=2
    ),
}
results = []
for name, attempt in attempts.items():
    try:
        attempt()
        results.append({"case": name, "error": None})
    except MLBundleError as error:
        results.append(
            {"case": name, "error": type(error).__name__, "message": str(error)}
        )
results
