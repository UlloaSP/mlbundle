# Security policy

mlbundle is pre-alpha. Phases 1–3 implement strict manifest validation,
contract helpers, secure container reading/writing, metadata-only `inspect`,
public export/load and the sklearn adapter.
The container validates ZIP/ZIP64 layout, bounds sources and actual decompression,
checks CRC and member integrity, and rejects unsafe members and polyglot wrappers.
The writer publishes complete validated files atomically where hard links work;
the exclusive-creation fallback cleans up on failure but exposes the destination
while copying and cannot guarantee crash-atomic publication. The version 1 contract
is defined in [docs/format.md](docs/format.md). XGBoost and ONNX loading remain
unimplemented; their security requirements remain obligations for later phases.

## Threat model

A runtime such as MLSuite will receive `.mlbundle` files uploaded by users.
Treat the entire file and every manifest field as attacker-controlled, even
when the archive appears to come from a trusted training workflow. An attacker
may try to execute code during loading, select arbitrary classes, exhaust
memory or CPU with an archive, expose training data, or supply metadata and
visualisations that do not describe the actual model.

The writer runs on the training machine and must reject unsupported models
there. The reader must independently validate uploads; writer checks do not
make a received file trustworthy.

## Required reader guarantees

- Never use pickle, joblib or cloudpickle, even as an option or fallback.
- Never import or instantiate from strings in the manifest. Classes must come
  from a closed list inside mlbundle.
- Keep the skops trusted-types list inside mlbundle. Never accept a list from
  the bundle or let its author expand the reader's trust boundary.
- Export must fail on the user's machine when a model contains custom types,
  listing unsupported types from `skops.io.get_untrusted_types`. This must be
  discovered before upload, not at runtime.
- Read zip content in memory by allowed member names; never extract to disk.
  Reject unknown and duplicate members, names with path separators or `..`,
  symlinks and encrypted entries. Limit member count, compressed size,
  uncompressed size and compression ratio before allocating or loading content.
- Validate local and central records and any data descriptors; require exact
  decompressed sizes and CRC. Reject comments, non-ZIP64 extra fields and bytes
  outside the contiguous ZIP layout. ZIP64 and validated streaming descriptors
  are accepted within the same resource ceilings.
- Verify each member's SHA-256 against the manifest and validate member sizes.
- Validate the manifest strictly and reject unknown format versions.
- Provide a metadata-only `inspect` operation that does not deserialize a model.
  It must not present model-derived claims as verified against the model.
- Treat every model-derived value as a claim. Recompute it from the loaded model
  using the same extraction code as the writer, and reject mismatches, including
  disagreements in visualisation structure.

The supported storage formats are skops for scikit-learn, native UBJSON for
XGBoost and the user's original ONNX file. There is no conversion or fallback
serialization path. Adapters must use lazy imports for optional dependencies.

## Implemented sklearn trust boundary

Export and load call skops' `get_untrusted_types` and reject all types outside
the fixed supplemental list `["sklearn.tree._tree.Tree"]`. Load passes exactly
that list to `skops.io.loads`, alongside skops' built-in trusted defaults. This
entry is native numeric tree storage required by DecisionTree, RandomForest,
ExtraTrees and GradientBoosting. It does not add arbitrary functions or I/O
constructors. The bundle cannot provide additional trusted types.

Skops 0.16.0 deliberately distrusts tree storage because unchecked child and
feature indices can crash native inference. mlbundle checks loaded tree dimensions,
indices, reachability, depth and owning-estimator dimensions before computing
importances or predicting. Ensemble feature mappings must stay within parent
input widths. It bounds tree constructor dimensions and class counts before
native construction, and checks state-array shapes against those dimensions.
It also bounds the inner skops ZIP and validates every schema-referenced numeric
array header, including renamed files, before skops reads it. Zero-itemsize
arrays are rejected; object arrays never use pickle loading. Malformed native
payloads are wrapped in bounded `InvalidModelError` messages. These checks do not
make the native parser a sandbox.

The installed skops load path builds nodes, audits them and only then constructs
objects; NumPy buffers use `allow_pickle=False`. No pickle/joblib loading fallback
is used. Skops imports `pickle` only to recognize an unrelated save-side exception;
this does not deserialize pickle bytes. Custom estimators/functions and XGBoost
estimators inside sklearn pipelines are rejected. HistGradientBoosting's separate
`TreePredictor` storage is also excluded. Broadening the supplemental list requires
source review and hostile-payload tests, never author approval or bundle metadata.

Both export and load recompute the same model claims and structures and compare
canonical bytes, preserving class-label types and order. Export additionally
compares the original and reloaded native models before publication. Loaded results
contain the native model for runtime explanations; mutating it invalidates the
relationship to the verified metadata. Pipeline structure is null because v1
does not describe its input transformations.

## Limits of these guarantees

SHA-256 hashes detect corruption, not deliberate tampering: an attacker can
replace a member and update its hash in the manifest. They do not prove who
trained the model or authenticate its provenance. Bundle signing is out of
scope. Training-environment records and free-form user metadata are assertions
by the author; checking model-derived claims does not authenticate those records.

Validation does not prove model quality, fairness, training history or the
truth of user metadata. Pickle-free formats do not sandbox model execution or
eliminate vulnerabilities in native parsers and dependencies. A consuming
runtime still owns authentication, authorization, process isolation and
execution resource limits.

Training rows must stay on the training machine unless the user explicitly
requests samples. Samples are real data; schemas, feature names and user
metadata may also contain sensitive information. Users must review what they
choose to share.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting for UlloaSP/mlbundle](https://github.com/UlloaSP/mlbundle/security/advisories/new).
Include the affected commit, a minimal reproducer, the expected and observed
behavior, and the impact. Share malicious bundles or sensitive details through
that private report rather than a public issue or pull request. Remove personal
or confidential training data from any reproducer you submit.

There are no published versions yet. Report findings against the repository's
current code and identify the commit you tested.
