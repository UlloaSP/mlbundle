# Version 1 container rules

Normative section of [the format contract](format.md).

## Container and resource limits

A bundle is one ZIP file with the `.mlbundle` extension. Its root contains only
the case-sensitive names below. Names are ASCII, with no directory entries,
slashes, backslashes, `..`, NULs or alternative spellings. Unknown or duplicate
names, symlinks, other non-regular entries, encrypted entries, unsupported
compression methods, multi-disk archives and malformed ZIP metadata must be
rejected. Only ZIP_STORED (method 0) and ZIP_DEFLATED (method 8) are allowed.
ZIP64 is allowed within the limits, including forced ZIP64 local records for
small files. Only a single ZIP64 extra field (ID 0x0001) is permitted in a header,
and it must contain exactly the replacement values required by that header's
sentinel fields, in ZIP order. Reject other, redundant, duplicate or malformed
extra fields. ZIP64 end records have the standard fixed 44-byte body, with no
extensible data sector; their locator and ordinary end record must agree.

Data descriptors are allowed for streaming ZIP writers. Accept the signed and
unsigned 32-bit and ZIP64 forms only when CRC and both sizes match the central
directory. With a descriptor, local CRC/sizes may be zero or match the directory;
ZIP64 sentinel sizes must be resolved through the local ZIP64 extra field.
Without a descriptor, local CRC and sizes must match the directory exactly.
The version-needed field is 10 (stored only), 20 or 45; ZIP64 headers use 45.
Allowed general-purpose flags are the UTF-8 name flag, the descriptor flag and
the two deflate-option bits (deflated entries only). Reject all other flags.

Reject archive and member comments. The first local header starts at byte zero;
local records, payloads and any descriptors cover the archive contiguously up
to the central directory. The directory ends exactly at the end record or
ZIP64 end record, followed only by its locator and the ordinary end record.
Reject prepended/appended bytes, gaps, overlapping records and unlisted local
records. Central entries may list local records in another order. These rules
exclude polyglot wrappers and metadata the mlbundle writer never produces.
File attributes may omit a Unix file type (as ordinary ZIP writers do); an
explicit type must be regular, and the DOS directory bit must be clear.

Readers must bound the source before buffering it: stat filesystem paths and
read streams in bounded chunks with at most one excess byte for detection.
Streams are consumed from their current position and are not closed by mlbundle.
Readers must read named members into memory, never extract them to disk. They
must inspect directory metadata and enforce limits before reading, allocating
member buffers or loading models; they must also limit actual bytes produced
during decompression, rather than trusting directory sizes. Reject inconsistencies
between headers, directory entries, actual sizes and ZIP CRC checks. No path,
module name, trusted-types list or external file reference can select loading
code. ONNX external tensor data is unsupported: the original file must be
self-contained. Native model formats do not sandbox their parsers or execution.
Decompression uses chunks of at most 64 KiB and a one-byte lookahead beyond the
declared output size; the lookahead is rejected before it is retained. Require
the exact output size, CRC and complete deflate stream, with no trailing packed
data. Declared sizes are checked against member, aggregate and ratio limits first.

One MiB is 1,048,576 bytes. `ArchiveLimits` defines these defaults and ceilings:

| Limit | Maximum |
| --- | ---: |
| Member count, including manifest | 5 (minimum configured value 2) |
| Entire archive, including headers | 512 MiB |
| Sum of compressed member sizes | 512 MiB |
| Any uncompressed member | 512 MiB |
| Sum of all uncompressed member sizes, including manifest | 1 GiB |
| Each member's uncompressed/compressed ratio | 200 |
| `manifest.json` | 8 MiB |
| `structure.json` | 256 MiB |
| `schema.json` | 16 MiB |
| `samples.json` | 16 MiB |
| JSON container nesting depth | 32 |

The effective member cap is the smaller of its specific cap and the general
member cap. Check the ratio as `uncompressed_size <= 200 * compressed_size`;
an empty member with both sizes zero is exempt, a nonempty member with compressed
size zero is invalid. This applies to stored members too. The mlbundle writer
always uses deflate level 6 with the RLE strategy and rejects inputs exceeding the ratio ceiling;
the reader also accepts stored entries from other tools. These limits admit
forests whose native model is tens of MiB and whose JSON tree arrays are larger.
Callers may lower archive limits, never raise the ceilings in version 1.
Consumers still need process memory, CPU and prediction limits; a 1 GiB byte
budget is not a guarantee that a parsed model uses at most 1 GiB of RAM.

### Container functions and inspection

The internal functions are `mlbundle.container.read_container(source, *,
limits=None)` and `mlbundle.writing.write_container(manifest, members, *,
destination=None, overwrite=False, limits=None)`. Reading returns a validated
`Manifest` and a dictionary of original uncompressed bytes keyed by `MemberName`,
including `manifest.json`. Source is a path, bytes or a binary stream. It never
deserializes a native model or resolves an adapter.

Writing accepts a `Manifest` or JSON-ready manifest parts, plus non-manifest
member bytes. The writer computes and replaces all integrity records; callers
must already set the model pairing and optional references. JSON-ready parts
may omit integrity. Optional JSON members pass the bounded canonical decoder;
their original bytes are preserved. The container validates JSON syntax and
bounds, while later model/schema APIs own semantic structure and schema checks.
Metadata inspection does not infer whether a model requires structure.

Internally, optional JSON members retain their validated values within one
operation. Inspect/load reuse that decode for schema/sample checks, and load
compares freshly encoded native extraction with stored structure bytes. Equality
proves canonical identity; other wire spellings encode the retained value once.
There is no global cache or bundle-supplied validation record. Public export
reuses the writer's complete container validation when returning metadata.

The writer emits manifest, model, schema, structure, samples in that fixed order,
omitting absent members. ZIP timestamps are 1980-01-01 00:00:00, the creating
platform is Unix, permissions are regular-file 0600, compression is deflate
level 6 with the RLE strategy, memory level 8 and 64 KiB input chunks, and
extra fields/comments are empty. RLE avoids the different general match-search
algorithms used by zlib and zlib-ng across platforms; it compresses repeated-byte
runs and uses Huffman coding for other bytes. It creates no descriptors or ZIP64
records within the version 1 ceilings. Identical inputs produce identical bytes
on Windows and Linux; the compression implementation is the standard zlib
provided by Python (byte identity across different future zlib implementations
is not a format requirement).
Compression output is bounded by the remaining archive and aggregate compressed
budgets, reserving the fixed header/directory overhead before compression.

Writing returns bytes and optionally publishes them to a path after validating
the complete archive through the reader. Publication uses a flushed temporary
file in the destination directory: atomic replacement when overwrite is true,
or atomic hard-link creation otherwise, followed by temporary-file cleanup.
When hard links are unsupported, publication uses exclusive destination creation,
flush and fsync, removing its own destination on failure. This fallback prevents
overwrites and leaves no partial file after a reported failure, but the destination
is visible while copying; it is not an atomic visibility guarantee or crash-safe
publication. Other filesystem errors propagate. Existing
destinations raise `FileExistsError` unless overwrite is explicitly requested;
the no-overwrite decision also holds if another writer wins a race.

Public `mlbundle.inspect(source, *, limits=None)` returns `MetadataView` with
the manifest and fixed-order member listing. It validates ZIP records, limits,
manifest, member hashes and JSON syntax without importing model libraries.
It cannot attest that model-derived claims or structure match the model; only
the full validating `load` can do that. Inspection additionally validates and
exposes schema and samples under [their declaration contract](format-schema.md),
without pandas or mlschema. Export requires explicit `sample_rows=` for real rows.

Container/layout/CRC inconsistencies raise `InvalidContainerError`; resource
ceilings raise `LimitExceededError`. Invalid manifest JSON or typed fields raise
`InvalidManifestError`; unknown version strings under the mlbundle identity
raise `UnsupportedFormatVersionError` before validating version-specific fields.
Invalid optional JSON raises `InvalidJSONError`. Member hash or integrity-size
disagreement raises `IntegrityMismatchError` naming the allowed member.
Messages do not include untrusted names, JSON blobs or ZIP exception text.
