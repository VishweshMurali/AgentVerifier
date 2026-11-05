/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Which data provider to use: "static" (default) or "api". */
  readonly VITE_DATA_PROVIDER?: string;
  /** Origin of the local server for the "api" provider; empty when it serves the page. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
