import { invoke, isTauri } from "@tauri-apps/api/core";

export type AnalyticsValue = string | number;
export type AnalyticsProps = Record<string, AnalyticsValue>;

let analyticsEnabled = false;

export function setAnalyticsEnabled(enabled: boolean) {
  analyticsEnabled = enabled;
}

export function getAnalyticsEnabled(): boolean {
  return analyticsEnabled;
}

export async function trackAnalyticsEvent(
  name: string,
  props?: AnalyticsProps,
): Promise<void> {
  if (!analyticsEnabled || !isTauri()) {
    return;
  }

  try {
    await invoke("plugin:aptabase|track_event", { name, props });
  } catch (error) {
    if (import.meta.env.DEV) {
      console.debug("Aptabase event dropped", { name, error });
    }
  }
}
