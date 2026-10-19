// The stored run of examples/train.py; regenerate with `vp run example`.
import data from "./example.json";

export interface CliRun {
  exit: number;
  stdout: string;
  stderr: string;
}

export const example = data as typeof data & {
  inspect: CliRun;
  verify: CliRun;
  editedInspect: CliRun;
  editedVerify: CliRun;
};

// The hero shows the MLbundle part of the script: everything from the export call on.
const lines = example.code.trimEnd().split("\n");
export const excerpt = lines
  .slice(lines.findIndex((line) => line.startsWith("export(")))
  .join("\n");

const units = ["B", "KB", "MB"];
export const formatBytes = (bytes: number): string => {
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
};
