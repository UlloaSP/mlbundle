// Sibling sites: local dev servers while developing, the production hosts in built output.
const dev = import.meta.env.DEV;

export const landingUrl = dev ? "http://localhost:4340/" : "https://mlbundle.eshkra.dev/";
export const playgroundUrl = dev ? "http://localhost:10021/" : "https://playground.mlbundle.eshkra.dev/";
