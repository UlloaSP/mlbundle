---
title: Limits
description: The archive, member and model ceilings mlbundle enforces before allocating or loading content, and how to lower them.
---

Readers and writers enforce these ceilings before allocating or decompressing content. They are
safety maxima: you can lower them for a call, never raise them.

## `ArchiveLimits`

| Field | Default and maximum |
| --- | --- |
| `max_members` | 6 (minimum 2) |
| `max_archive_bytes` | 512 MiB |
| `max_compressed_bytes` | 512 MiB |
| `max_member_bytes` | 512 MiB |
| `max_total_bytes` | 1 GiB, all members uncompressed |
| `max_compression_ratio` | 200 |
| `max_manifest_bytes` | 8 MiB |
| `max_structure_bytes` | 256 MiB |
| `max_schema_bytes` | 16 MiB |
| `max_samples_bytes` | 16 MiB |

A specific member cap never exceeds `max_member_bytes`.

```python
from mlbundle import ArchiveLimits, inspect

strict = ArchiveLimits(
    max_archive_bytes=16 * 1024 * 1024, max_structure_bytes=4 * 1024 * 1024
)
view = inspect("trained.mlbundle", limits=strict)
```

Values above a ceiling are rejected by validation. Pass the same object to `export`, `inspect` or
`load`; an upload service would typically lower `max_archive_bytes` to its request size.

## Fixed ceilings

| Item | Ceiling |
| --- | --- |
| JSON nesting depth | 32 |
| Features | 100,000 |
| Classes per output | 100,000 |
| Outputs | 1,024; XGBoost, CatBoost and ONNX: 1 |
| `user_metadata` | 1,024 top-level keys, 64 KiB, depth 8 |
| Hyperparameters | 4,096 keys, 1 MiB, depth 16 |
| XGBoost | 100,000 trees, depth 14 |
| Record string values | 64 characters |

## What limits do not cover

These ceilings bound archive parsing and the native state mlbundle validates. They do not cap
process memory, prediction rows or CPU time during inference. A runtime must enforce those itself.

The normative container limits are in [Container and limits](/format/container/), native work
ceilings in [Prediction state](/format/state/#work-and-allocation-ceilings) and ONNX bounds in
[ONNX mapping](/format/onnx/).
