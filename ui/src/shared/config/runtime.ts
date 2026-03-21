interface ResolveApiBaseUrlOptions {
  envBaseUrl?: string | null;
  windowServerUrl?: string | null;
  defaultBaseUrl?: string;
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function resolveBooleanEnv(
  value: string | boolean | null | undefined,
  defaultValue: boolean,
): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value !== "string") {
    return defaultValue;
  }
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return defaultValue;
  }
  return !(normalized === "0" || normalized === "false" || normalized === "no");
}

export function resolveApiBaseUrl(
  options: ResolveApiBaseUrlOptions = {},
): string {
  const { envBaseUrl = null, windowServerUrl = null, defaultBaseUrl = "/v1" } =
    options;

  if (windowServerUrl && windowServerUrl.trim()) {
    return `${trimTrailingSlash(windowServerUrl.trim())}/v1`;
  }

  if (envBaseUrl && envBaseUrl.trim()) {
    return trimTrailingSlash(envBaseUrl.trim());
  }

  return defaultBaseUrl;
}

const browserServerUrl =
  typeof window !== "undefined" ? window.__IZWI_SERVER_URL__ ?? null : null;

export const APP_VERSION = __APP_VERSION__;
export const APP_ICON_URL = `/app-icon.png?v=${APP_VERSION}`;
export const API_BASE_URL = resolveApiBaseUrl({
  envBaseUrl: import.meta.env.VITE_API_BASE_URL ?? null,
  windowServerUrl: browserServerUrl,
});
export const VOICE_STUDIO_ENABLED = resolveBooleanEnv(
  import.meta.env.VITE_VOICE_STUDIO_ENABLED,
  true,
);
