# plan__basic and plan__premium become one onehot-category field.
# Samples are real rows: they are included only because sample_rows is set.
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlbundle import export

features = pd.DataFrame(
    {
        "age": [34.0, 51.0, 27.0, 45.0, 62.0, 38.0],
        "plan__basic": [1, 0, 1, 0, 0, 1],
        "plan__premium": [0, 1, 0, 1, 1, 0],
        "active": [True, False, True, True, False, True],
    }
)
churned = ["no", "yes", "no", "no", "yes", "no"]
model = LogisticRegression(max_iter=500).fit(features, churned)

view = export(model, "samples.mlbundle", training_data=features, sample_rows=3)
[field["kind"] for field in view.schema["fields"]]
