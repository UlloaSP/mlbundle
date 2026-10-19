export interface RuntimeVersions {
  python: string;
  mlbundle: string;
  mlschema: string;
  "scikit-learn": string;
  skops: string;
  pandas: string;
  numpy: string;
}

/** What the last expression of a run produced, as reported by harness.py. */
export type RunOutcome =
  | { kind: "json"; text: string }
  | { kind: "repr"; text: string; typeName: string }
  | { kind: "none" }
  | { kind: "error"; text: string };

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface MemberRecord {
  name: string;
  sha256?: string;
  size?: number;
}

/**
 * One bundle as mlbundle reported it. `verified`: inspect and load both passed, so
 * model claims and structure were recomputed. `inspected`: metadata only, the model
 * was never deserialized. `unverified`: inspect passed, load rejected it.
 * `rejected`: inspect rejected it.
 */
export interface BundleReport {
  fileName: string;
  sizeBytes: number;
  source: "generated" | "uploaded";
  status: "verified" | "inspected" | "unverified" | "rejected";
  error?: string;
  manifest?: { [key: string]: Json };
  members?: MemberRecord[];
  schema?: { [key: string]: Json } | null;
  samples?: { [key: string]: Json }[] | null;
  structure?: { [key: string]: Json } | null;
  /** The exact bytes mlbundle wrote, for download. Generated bundles only. */
  bytes?: Uint8Array;
}

export interface RunReport {
  outcome: RunOutcome;
  bundles: BundleReport[];
  /** Bundles beyond the per-run limit that were not inspected. */
  skipped: number;
}

export type OutputStream = "stdout" | "stderr";

export type ToWorker =
  | { type: "run"; id: number; code: string }
  | { type: "inspect"; id: number; name: string; data: ArrayBuffer };

export type FromWorker =
  | { type: "progress"; message: string }
  | { type: "ready"; versions: RuntimeVersions }
  | { type: "boot-error"; message: string }
  | { type: "stream"; id: number; stream: OutputStream; text: string }
  | { type: "done"; id: number; report: RunReport; durationMs: number };
