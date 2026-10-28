---
title: What is verified
description: Which parts of a bundle mlbundle checks, which it recomputes from the model, and which remain the author's unauthenticated declarations.
---

The author of a bundle may be an attacker. mlbundle therefore treats everything in it as a claim
and sorts each claim into one of three groups.

## Recomputed from the model on `load`

`load` and `mlbundle verify` rebuild the model through a closed adapter and extract these values
with the same code the writer used. Any difference in canonical JSON bytes raises
`ClaimMismatchError`.

- library, estimator class and kind;
- hyperparameters the native format can recover;
- feature, output and per-output class counts;
- native feature names and model-exposed dtypes;
- class labels owned by the model, and labels bound to XGBoost or present in an ONNX graph;
- prediction state commitments (`state_sha256`) for scikit-learn and composites;
- the complete `structure.json`: trees, ensembles, coefficients, importances, XGBoost dumps and
  CatBoost's symmetric trees;
- schema reports, through the verified class labels.

A reviewer never sees a tree that differs from the model used for prediction.

## Validated by `inspect` and `load`

These checks need no model library:

- ZIP layout, ZIP64 records, CRCs and compression; only allowed member names, no paths, symlinks,
  duplicates, encryption or bytes outside the archive layout;
- member count, sizes, compression ratio and JSON depth within [limits](/reference/limits/);
- a strict manifest with exact types and a known format version;
- a canonical `pylock.toml` whose targets, wheel choices and Python agree with the manifest;
- SHA-256 and size of every member except the manifest, which cannot contain its own hash;
- schema coverage, mappings by feature origin and report shape; sample keys.

## Declarations nobody can verify

| Declaration | Why it cannot be checked |
| --- | --- |
| Training environment and the runtime lock | Recorded by the writer; any author can edit them. Installing a lock runs its packages' code. |
| Training dtypes | Describe a frame that is not in the bundle. |
| Schema `source`, separator and nullability | Inference history is not reproducible from the bundle. |
| Declared or frame feature names | Labels for a model without native names; count and alignment are checked. |
| `target_names`, `user_metadata`, composite step names | Free author text. |
| XGBoost label binding history | Load verifies consistency with the native copy, not how it was trained. |
| ONNX labels for a label-free graph | Only their count is checked. |
| Samples | Real rows as the author wrote them. |

## What hashes do not prove

SHA-256 detects corruption, not tampering: an attacker can replace a member and update its hash.
There is no signing, so nothing proves who trained a model, its training history, quality or
fairness. Bundle signing is out of scope.

## What a runtime still owns

Pickle-free formats do not sandbox native parsers or inference. A service that accepts uploads must
still provide authentication, authorization, process isolation and memory, CPU and row limits, and
keep model libraries patched. mlbundle's safety ceilings bound archive parsing and model state, not
process memory or the rows you predict.

mlbundle has not had an independent security review. Do not accept bundles from untrusted parties.
Read the [security policy](/security/) for the threat model, guarantees and private vulnerability
reporting.
