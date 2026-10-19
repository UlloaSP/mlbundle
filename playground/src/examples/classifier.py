# A decision tree classifier exported with a schema inferred from its training frame.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlbundle import export

training = pd.DataFrame(
    {
        "temperature": [2.0, 31.0, 5.5, 28.0, 12.0, 25.5, 1.0, 34.0],
        "humidity": [0.81, 0.32, 0.77, 0.41, 0.66, 0.35, 0.90, 0.28],
        "weather": ["cold", "warm", "cold", "warm", "cold", "warm", "cold", "warm"],
    }
)
features = training.drop(columns="weather")
model = DecisionTreeClassifier(max_depth=3, random_state=7)
model.fit(features, training["weather"])

# Writes classifier.mlbundle into this run's working directory. The playground then
# inspects it and verifies it with mlbundle.load before showing it on the right.
view = export(
    model,
    "classifier.mlbundle",
    training_data=training,
    target_names=["weather"],
    metadata={"purpose": "playground example"},
)
view.manifest.target.class_labels
