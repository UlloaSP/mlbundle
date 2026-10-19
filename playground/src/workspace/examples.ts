const sources = import.meta.glob<string>("../examples/*.py", {
  query: "?raw",
  import: "default",
  eager: true,
});

export interface CodeExample {
  id: string;
  label: string;
  code: string;
}

const CATALOG = [
  ["classifier", "Classifier with schema"],
  ["regressor", "Random forest regressor"],
  ["samples", "Explicit samples and one-hot"],
  ["pipeline", "sklearn Pipeline"],
  ["errors", "Rejected models"],
] as const;

export const CODE_EXAMPLES: readonly CodeExample[] = CATALOG.map(([id, label]) => {
  const code = sources[`../examples/${id}.py`];
  if (code === undefined) throw new Error(`Missing example file: ${id}.py`);
  return { id, label, code };
});

export const getCodeExample = (id: unknown): CodeExample =>
  CODE_EXAMPLES.find((example) => example.id === id) ?? CODE_EXAMPLES[0]!;
