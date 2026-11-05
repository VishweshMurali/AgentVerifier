/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Which data provider to use. Only "static" exists now. */
  readonly VITE_DATA_PROVIDER?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
