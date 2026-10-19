// Development links point at the local landing and docs servers, as the landing does.
const dev = import.meta.env.DEV;

export const LINKS = {
  home: dev ? "http://127.0.0.1:4340/" : "https://mlbundle.eshkra.dev/",
  docs: dev ? "http://127.0.0.1:4341/" : "https://docs.mlbundle.eshkra.dev/",
  github: "https://github.com/UlloaSP/mlbundle",
  eshkra: "https://eshkra.dev/",
} as const;
