---
title: mlbundle documentation
description: Package a trained model, its manifest, a locally inferred schema and model structure in one verifiable .mlbundle file, without pickle.
tableOfContents: false
hero:
  title: One verifiable file for a trained model.
  tagline: mlbundle puts the model in its native format, a manifest of model and training-environment metadata, a locally inferred input schema and the model's structure into one .mlbundle file. Before it hands you the model, the reader recomputes every model-derived claim and rejects disagreement.
  actions:
    - text: Quick start
      link: /getting-started/quick-start/
      variant: primary
    - text: What is verified
      link: /guides/trust/
      variant: secondary
    - text: Format 1 contract
      link: /format/
      variant: minimal
---

```python title="train.py"
from sklearn.tree import DecisionTreeClassifier

from mlbundle import export

rows = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
model = DecisionTreeClassifier(random_state=7).fit(rows, ["cold", "warm"] * 2)
export(model, "trained.mlbundle", feature_names=["temperature", "humidity"])
```

```python title="serve.py"
from mlbundle import load

bundle = load("trained.mlbundle")  # verified before it is returned
print(bundle.predict_proba([[0.0, 1.0]]))  # [[1.0, 0.0]]
print(bundle.predict_record({"0": "0.0", "1": "true"})[0].predicted_label)  # cold
```

No pickle, joblib or cloudpickle is involved at any step. scikit-learn models are stored with
skops against a closed allowlist, XGBoost as native UBJSON, CatBoost as native CBM, and ONNX files
byte for byte. Loading rebuilds the model through mlbundle's own adapter, extracts its claims and
structure with the same code the writer used, and rejects the bundle when they disagree.

:::note[Before you accept uploads]
Format 1 is not frozen, so a future change may require re-exporting bundles. The security design
has not had an independent review: do not accept bundles from untrusted parties.
:::

## Find your path

<div class="path-grid not-content">
  <a class="path-card" href="/getting-started/quick-start/">
    <strong>Bundle a first model</strong>
    <span>Export a classifier, inspect it, load it and predict.</span>
  </a>
  <a class="path-card" href="/guides/export/">
    <strong>Control what goes in</strong>
    <span>Feature names, labels, targets, metadata and <code>overwrite</code>.</span>
  </a>
  <a class="path-card" href="/guides/schema-and-samples/">
    <strong>Keep data local</strong>
    <span>What the schema carries, and why samples need <code>sample_rows=N</code>.</span>
  </a>
  <a class="path-card" href="/guides/inspect-and-load/">
    <strong>Read a bundle</strong>
    <span><code>inspect</code> for metadata, <code>load</code> for a verified model.</span>
  </a>
  <a class="path-card" href="/guides/records/">
    <strong>Predict from a form</strong>
    <span>Records keyed by the schema, frozen per-output results.</span>
  </a>
  <a class="path-card" href="/adapters/overview/">
    <strong>Check your model</strong>
    <span>scikit-learn, XGBoost, CatBoost, ONNX and composite pipelines.</span>
  </a>
</div>

## Why a bundle

Uploading a pickled model and a DataFrame leaves three gaps that mlbundle closes:

| Gap | What mlbundle does |
| --- | --- |
| Loading a pickle can execute code. | Closed adapters use skops, native UBJSON/CBM or the original ONNX file. There is no pickle fallback. |
| The training versions are lost. | The manifest records Python, OS, architecture and installed package versions next to model and feature metadata. |
| Training data leaves the machine. | Schema inference runs locally. Rows are included only when you ask for them explicitly. |

mlbundle is the training-side companion to MLSuite, whose runtime accepts only `.mlbundle`
uploads. It is open source under the MIT license and part of [eshkra](https://eshkra.dev). Read
the source on [GitHub](https://github.com/UlloaSP/mlbundle).
