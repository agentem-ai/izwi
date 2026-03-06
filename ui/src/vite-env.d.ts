/// <reference types="vite/client" />

declare const __APP_VERSION__: string;

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface Window {
  __IZWI_SERVER_URL__?: string;
}
