---
title: ONNX
description: Bundle an existing self-contained tabular ONNX file unchanged, with verified signatures, labels and a CPU session on load.
---

Install `mlbundle[onnx,schema]`. mlbundle does not convert models to ONNX. It bundles an ONNX file
you already have, byte for byte, after validating that it is a self-contained tabular model.

This example assumes `trained.onnx` is a classifier with two numeric features and two probability
columns, and no label output:

```python
from pathlib import Path

from mlbundle import export, load

original = Path("trained.onnx").read_bytes()
export("trained.onnx", "onnx.mlbundle", class_labels=[["cold", "warm"]])

bundle = load("onnx.mlbundle")
assert bundle.native_model.data == original
assert bundle.native_model.session.get_providers() == ["CPUExecutionProvider"]
print(bundle.class_labels)  # [['cold', 'warm']]
print(bundle.predict_proba([[0.0, 0.0]]))
```

`export` accepts a path, bytes or an `onnx.ModelProto` (serialized once). `native_model` holds the
original `data` and an onnxruntime `session` on the CPU execution provider.

## Labels

- Labels in the graph (`ZipMap`, `LinearClassifier`, `TreeEnsembleClassifier` or `SVMClassifier`
  attributes) are verified; supplied `class_labels=` must agree with them.
- Without graph labels, supplied labels are declarations checked only for count; without either,
  encoded indices `0..N-1` are used.
- `predicted_label` comes from the graph's label output. Without one it is `None`; mlbundle never
  computes an argmax.

## Supported graphs

- Rank-two numeric inputs (`float32`, `float64`, `int32`, `int64`) with a fixed width from 1 to
  100,000, or several width-one inputs whose names become verified feature names.
- A batch dimension that is unknown, symbolic or exactly one.
- One target: a regression output of shape `[batch]` or `[batch, 1]`, or a probability matrix or
  ZipMap sequence, optionally with a label tensor.

Kind is determined from the graph's outputs, never declared. Verified claims include input and
output signatures, opsets, IR version, producer, model metadata properties and native labels.
ONNX has no visualization structure in version 1.

## Rejected before any session

External tensor data, custom operator domains, model-local functions, runtime or training
configuration, image, string or sequence inputs, multi-target regression, fixed batches above one
and outputs whose kind cannot be determined. Validation scans the protobuf with bounded work before
onnxruntime sees it.

Tested with ONNX 1.23.0 and onnxruntime 1.30.0. The exact layouts and bounds are in
[ONNX mapping](/format/onnx/).
