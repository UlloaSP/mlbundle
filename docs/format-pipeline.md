# Version 1 composite pipelines

Normative section of [the format contract](format.md). This pre-release format
extension preserves reading of standalone bundles. Older readers reject composites.

## Representation and declarations

Only an exact outer `sklearn.pipeline.Pipeline` ending in a supported exact
XGBClassifier/XGBRegressor or CatBoostClassifier/CatBoostRegressor is composite.
All-scikit-learn pipelines retain their single skops member and null structure.
Every native adapter's standalone scope, preflight, claims and structure rules apply.
No native estimator or booster type is added to the skops trust list.

`model.library` is `sklearn`, `serialization` is `pipeline`, `estimator_class`
is `Pipeline`, and `state_sha256` is null. Kind, class/output counts and target
labels equal the native component claims. `model.n_features` and manifest
features describe original inputs. Outer hyperparameters have exactly `memory`
(null), `verbose` (boolean), `transform_input` (null). ONNX fields are absent/null.

`model.pipeline` is required exactly for this serialization. Its strict frozen
records reject unknown fields and coercions. The record contains:

- `step_names`: 1–128 unique strings, 1–256 characters, without `__` or any
  of `steps`, `memory`, `verbose`, `transform_input`. Names are data.
- `transformers`: null for one step; otherwise `member="model.skops"`,
  `estimator_class="Pipeline"`, `hyperparameters` and non-null `state_sha256`.
  Parameters use the existing sklearn projection, including `$estimator_ref`
  in constructor collections and full records in named deep keys, with the
  existing 4,096-key, 1 MiB and depth-16 bounds.
- `estimator`: `library` (`xgboost` or `catboost`), paired `serialization` and
  `member`, plus the complete shared `ModelClaims` record under `claims`.
  No ONNX fields or absent classifier labels are admitted.
- `transformed_features`: `n_features` (1–100,000), ordered unique `names`
  (1–1,024 characters each), `origin` (`transformer`, `model`, `input`, `generated`).
  Width equals the native component feature count. Generated names are exactly
  `transformed_feature_1` through `transformed_feature_N`.

Final step names and outer options are reconstruction declarations. Prefix
names/order/options are also verified against the loaded skops member. Rebuilding
declarations preserves them; it does not authenticate their historical values.
No manifest class name, parameter projection or step name selects loading code.

Example (native claims and transformer parameters shortened for readability):

```json
{
  "library":"sklearn", "serialization":"pipeline", "estimator_class":"Pipeline",
  "kind":"classifier", "n_features":3, "n_classes":[2], "n_outputs":1,
  "state_sha256":null,
  "hyperparameters":{"memory":null,"verbose":false,"transform_input":null},
  "pipeline":{
    "step_names":["scale","est"],
    "transformers":{
      "member":"model.skops", "estimator_class":"Pipeline",
      "hyperparameters":{"steps":[["scale",{"$estimator_ref":"StandardScaler"}]]},
      "state_sha256":"506feb7a8fd45118e204d24081be7ec4d8b8998e73fa4581863741fb1852533f"
    },
    "estimator":{"library":"xgboost","serialization":"ubjson","member":"model.ubj",
      "claims":{"estimator_class":"XGBClassifier","n_features":3}},
    "transformed_features":{"n_features":3,"names":["a","b","c"],"origin":"transformer"}
  }
}
```

## Members and component I/O

Standalone bundles have exactly one of the four model members. A composite
with a prefix has exactly `model.skops` plus `model.ubj` or `model.cbm`.
A single-step composite has only its native member and null `transformers`.
Passthrough/None preceding steps still require the prefix member to preserve
the recipe. A pair without its declaration, a missing declared member, ONNX
pairs, two boosters or three model members fail. Manifest integrity covers
exactly the declared model members and referenced optional members.

Composite `structure.json` is required. Schema and explicit samples are optional;
six members permit both with the two model members and manifest. Every other
archive ceiling is unchanged. Callers may lower member count and other limits.

Export gates scope/options before serialization. The leaf adapter receives
`ExportDeclarations`, prepares its native copy where required, serializes and
deserializes bytes with its own preflight. Shared claims and native structure
must survive that round trip. The prefix is a transformer-only exact Pipeline:
the shared skops byte route applies the owned trust/catalog, payload/buffer
bounds and payload-versus-reserialized commitment comparison. It checks fitted
transformers and projects their parameters with the standalone owner. Each component
reuses verified snapshots within the synchronous operation, as defined in
[state verification](format-state.md); public calls never share that state.

Shared composite extraction verifies fitted widths and names, recomputes the
recipe and outer claims for both original and restored components, and compares
canonical bytes. The reconstructed Pipeline and extended native structure must
survive the full comparison before atomic publication. Nothing fits a model,
transforms sample/dummy rows as verification, or changes the caller's pipeline.

Load first validates container/manifest/integrity and decodes each JSON member
once. Schema/samples remain core-owned. The native leaf and optional prefix
load from bytes through their own checked routes. A constant imported Pipeline
class combines the loaded steps; no manifest-directed import or constructor is
used. The writer's extraction recomputes every component/outer claim and the
extended structure. Only exact canonical agreement permits a loaded result.

## Fitted widths and feature names

Widths derive from fitted state without data. Every effective transformer input
width agrees with the preceding output. Intermediate/final widths are positive
and bounded by 100,000; an empty ColumnTransformer branch contributes zero.
Nesting is at most 16. Input width/names come from the first effective fitted
transformer, or the native leaf for an identity-only prefix/single-step pipeline.

The closed exact classes and rules are:

| Transformer | Width rule and consistency checks |
| --- | --- |
| StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer | Preserve width; fitted vectors/options and feature dimensions agree. |
| FunctionTransformer | Identity or the nine owned NumPy elementwise functions in [state](format-state.md); preserves width. Nonempty `kw_args`/`inv_kw_args` are excluded. |
| PCA | `components_.shape[0]` equals bounded `n_components_`; component input width and fitted mean/variance/singular-value vectors agree. |
| VarianceThreshold | Count true entries in the validated fitted support mask; variance vector has input width. |
| SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect | Count the checked support mask; fitted scores/p-values align; score functions remain the owned closed list. |
| SimpleImputer | Keep input width when `keep_empty_features`; otherwise count non-NaN fitted statistics, including dropped empty columns. Add the checked MissingIndicator feature count when requested. |
| KBinsDiscretizer | `ordinal` preserves width; onehot/onehot-dense sum fitted `n_bins_`. Edge lengths/order and the internal encoder counts/categories/sparsity agree. |
| PolynomialFeatures | Bounded combinatorial count from degree range, interaction-only and bias options; checked against both fitted output counts. No powers array expansion for width proof. |
| OneHotEncoder | Sum category counts after checked infrequent grouping and drop state; cross-check cached output counts, mapping lengths/indices, and original/grouped drop indices. |
| OrdinalEncoder | Preserve width; categories, missing indices and infrequent mappings align. |
| Pipeline, ColumnTransformer | Recursively compose/sum the closed rules. ColumnTransformer selectors, cached input indices, remainder complement, branch names/order, declared/fitted parameters, sparsity flag and output slices agree. |
| `"passthrough"`, None | Preserve width and recipe position. |

Other transformers (including KNNImputer and FeatureUnion) have no composite
width rule. Existing skops/parameter restrictions also apply: callable selectors,
scalar/slice-valued constructor selectors and arbitrary function/type parameters have
no supported projection. Catalog admission alone does not guarantee export.

Effective original names use the existing model/dataframe/declared/generated
rules. Checked `get_feature_names_out` methods propagate them; known elementwise
operations propagate directly. Otherwise use verified native leaf names, then
generated transformed names. Names must match width and remain unique/bounded. Polynomial names are generated
without dense powers_ arrays, with at most 100,000 term visits. ColumnTransformer
verbose_feature_names_out must be boolean; arbitrary format expansion is excluded.
Native names/dtypes stay separate in leaf claims: null XGBoost names or positional
CatBoost names are preserved. Native tree dumps are never renamed. This supplies
display coordinates, not inverse transformations or original-feature attribution.

## Structure, prediction and inspection

All native structure fields remain unchanged, with one sibling `pipeline`:

```json
"pipeline": {
  "step_name":"est",
  "features":{"n_features":3,"names":["a","b","c"],"origin":"transformer"}
}
```

Family remains xgboost/catboost. Tree indices/importances address transformed
features. This extension is recomputed and allowed only for composites.
Stored structure bytes are compared with fresh canonical extraction through
the existing single-decode owner. Legacy all-sklearn structure remains null.

`LoadedBundle.native_model` is a real Pipeline. Bundle input features/schema/
dtypes/samples describe original inputs. The composite predictor checks original
rank/width and numeric cells, admitting strings only for supported fitted encoders;
it transforms once, checks unchanged row count and delegates to a cached validated
native predictor. Native input/output checks, probability tolerances and prediction
settings remain authoritative. Single-step prediction delegates directly, preserving
native categorical admission. Transform uses default global output and disabled
metadata routing while retaining explicit per-step default/pandas output.

For XGBoost, bundle probability columns use bound original labels while the native
Pipeline's classes/predict remain encoded. CatBoost retains its native labels and
rejects `class_labels=`. Mutating a loaded native object invalidates verification.

Inspect imports no model library and attests only metadata/integrity. CLI inspect
and verify display escaped/bounded steps, final estimator, input/transformed widths
and transformed names. Inspect retains its unverified-model notice; JSON includes
the complete typed recipe. Verify loads and checks both parts and structure.

## Rejections

Existing skops/native/schema/sample/resource messages retain their owners.
Recipe/pairing failures use `InvalidManifestError: Invalid mlbundle manifest.`
Candidate model sets use `InvalidContainerError: Invalid model member combination.`
Actual/reference disagreement remains `Members disagree with manifest references.`
Canonical disagreement uses ClaimMismatchError with ordinary outer fields,
`pipeline.transformers`, `pipeline.step_names`, `pipeline.estimator.<field>`,
`pipeline.transformed_features`, or `structure`. Unsupported native placements
list sanitized unsupported types after the placement message below.

The new scope, state and prediction messages are:

- `Composite models require an exact outer Pipeline with 1 to 128 steps.`
- `Pipeline step names must be unique bounded sklearn names.`
- `Unknown outer Pipeline state.`
- `Composite Pipeline memory must be None; caching is unsupported.`
- `Composite Pipeline verbose must be a boolean.`
- `Composite Pipeline transform_input and metadata routing are unsupported.`
- `Composite export requires default global transform_output; use per-step set_output.`
- `Composite transformers support only default or pandas set_output.`
- `Native estimators are supported only as the last step of the outermost Pipeline.`
- `Composite final estimator must be a supported XGBoost or CatBoost wrapper.`
- `Transformer member must be a nonempty exact Pipeline.`
- `Transformer component contains a non-transformer.`
- `Composite transformers must be fitted.`
- `Composite input width cannot be recovered from fitted state.`
- `Transformer output width cannot be verified without data.`
- `Transformer nesting exceeds the limit.`
- `Transformer feature-name work exceeds the limit.`
- `Invalid transformer feature width.`
- `Transformer input widths disagree.`
- `Transformer fitted dimensions disagree.`
- `Invalid ColumnTransformer input mapping.`
- `Invalid ColumnTransformer output mapping.`
- `ColumnTransformer declared and fitted transformers disagree.`
- `Transformer output width disagrees with final estimator.`
- `Invalid transformed feature names.`
- `Transformed names and width disagree.`
- `Invalid composite Pipeline state.` (bounded unexpected-state wrapper)
- `Composite Pipeline transformation failed.`
- `Composite transformation must preserve prediction rows.`

Native estimators in ColumnTransformer, FeatureUnion, stacking, voting,
calibration, search, TransformedTargetRegressor, nested pipelines or any nonfinal
position remain excluded. Caching, metadata requests, non-null transform_input,
unknown outer state and unverifiable fitted widths fail before publication.
Separate byte preflights do not sandbox native parsers or prediction. Host process,
memory/CPU/row limits and dependency maintenance remain necessary.
