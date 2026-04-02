import { trackEvent as aptabaseTrackEvent } from "@aptabase/tauri";

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
  if (!analyticsEnabled) {
    return;
  }

  try {
    await aptabaseTrackEvent(name, props);
  } catch (error) {
    if (import.meta.env.DEV) {
      console.debug("Aptabase event dropped", { name, error });
    }
  }
}
