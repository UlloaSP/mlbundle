import type { SourceTree } from "./session.ts";
import pyproject from "../../../pyproject.toml?raw";
import lock from "./pylock.toml?raw";

// The repository's real package source, bundled at build time. Never a copy.
// state_fields.json is package data the sklearn adapter reads at import time.
const files = import.meta.glob<string>(
  ["../../../src/mlbundle/**/*.py", "../../../src/mlbundle/**/*.json"],
  { query: "?raw", import: "default", eager: true },
);

export const MLBUNDLE_SOURCES: SourceTree = Object.fromEntries(
  Object.entries(files).map(([path, text]) => [path.slice(path.indexOf("mlbundle/")), text]),
);

export const PYPROJECT = pyproject;

// Written by scripts/lock.py for this session's exact versions; Pyodide cannot run uv.
export const PLAYGROUND_LOCK = lock;
