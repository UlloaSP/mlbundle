import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlbundle import export, load

training = pd.DataFrame(
    [[0.0, 1.0, "cold"], [1.0, 0.0, "warm"]] * 2,
    columns=["temperature", "humidity", "target"],
)
features = training.drop(columns="target")
model = DecisionTreeClassifier(random_state=7)
model.fit(features, training["target"])

export(
    model,
    "trained.mlbundle",
    training_data=training,
    target_names=["target"],
)

bundle = load("trained.mlbundle")
record = {"temperature": 0.0, "humidity": 1.0}
prediction = bundle.predict_record(record)[0]
print(prediction.predicted_label, prediction.probabilities)
