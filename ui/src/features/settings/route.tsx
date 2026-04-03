import { Loader2, Moon, Sun, SunMoon } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "@/api";
import { setAnalyticsEnabled } from "@/app/analytics/client";
import {
  trackAnalyticsConsentChanged,
  trackThemePreferenceChanged,
} from "@/app/analytics/events";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { useTheme } from "@/app/providers/ThemeProvider";
import { PageHeader, PageShell } from "@/components/PageShell";
import { cn } from "@/lib/utils";
import { APP_VERSION } from "@/shared/config/runtime";

const THEME_OPTIONS: Array<{
  id: "system" | "light" | "dark";
  title: string;
  description: string;
  icon: typeof SunMoon;
}> = [
  {
    id: "system",
    title: "Auto",
    description: "Follow your operating system preference.",
    icon: SunMoon,
  },
  {
    id: "light",
    title: "Light",
    description: "Always use light mode.",
    icon: Sun,
  },
  {
    id: "dark",
    title: "Dark",
    description: "Always use dark mode.",
    icon: Moon,
  },
];

export function SettingsPage() {
  const { notify } = useNotifications();
  const {
    availableUpdate,
    status: updateStatus,
    lastCheckedAt,
    health: updaterHealth,
    errorMessage: updateErrorMessage,
    openPrompt: openUpdatePrompt,
    checkForUpdates,
  } = useAppUpdates();
  const { themePreference, setThemePreference, resolvedTheme } = useTheme();
  const [analyticsOptIn, setAnalyticsOptIn] = useState(false);
  const [isLoadingPreferences, setIsLoadingPreferences] = useState(true);
  const [isSavingPreference, setIsSavingPreference] = useState(false);

  const handleThemePreferenceChange = (nextPreference: "system" | "light" | "dark") => {
    if (themePreference === nextPreference) {
      return;
    }

    setThemePreference(nextPreference);
    void trackThemePreferenceChanged(nextPreference);
  };

  useEffect(() => {
    let active = true;

    api
      .getPreferences()
      .then((preferences) => {
        if (!active) {
          return;
        }
        setAnalyticsOptIn(preferences.analytics_opt_in);
        setAnalyticsEnabled(preferences.analytics_opt_in);
      })
      .catch((error) => {
        console.error("Failed to load user preferences:", error);
        if (!active) {
          return;
        }
        notify({
          title: "Could not load settings",
          description:
            "Some settings may not reflect your latest saved preferences.",
          tone: "warning",
        });
      })
      .finally(() => {
        if (active) {
          setIsLoadingPreferences(false);
        }
      });

    return () => {
      active = false;
    };
  }, [notify]);

  const handleAnalyticsToggle = async (nextValue: boolean) => {
    if (isSavingPreference) {
      return;
    }

    const previousValue = analyticsOptIn;
    setAnalyticsOptIn(nextValue);
    setIsSavingPreference(true);

    try {
      const response = await api.updateAnalyticsPreference({ opt_in: nextValue });
      setAnalyticsOptIn(response.analytics_opt_in);
      setAnalyticsEnabled(response.analytics_opt_in);
      void trackAnalyticsConsentChanged(
        response.analytics_opt_in ? "opted_in" : "opted_out",
        "settings",
      );
      notify({
        title: response.analytics_opt_in
          ? "Anonymous analytics enabled"
          : "Anonymous analytics disabled",
        tone: "success",
      });
    } catch (error) {
      console.error("Failed to update analytics preference:", error);
      setAnalyticsOptIn(previousValue);
      notify({
        title: "Could not update analytics preference",
        description: "Please try again.",
        tone: "warning",
      });
    } finally {
      setIsSavingPreference(false);
    }
  };

  return (
    <PageShell className="pb-10">
      <PageHeader
        title="Settings"
        description="Manage preferences for appearance and anonymous usage analytics."
      />

      <div className="space-y-6">
        <section className="rounded-2xl border border-border/70 bg-[var(--bg-surface-1)]/75 p-5 sm:p-6">
          <div className="mb-4">
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              Appearance
            </h2>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Choose how Izwi should render colors across pages.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            {THEME_OPTIONS.map((option) => {
              const Icon = option.icon;
              const isActive = themePreference === option.id;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => handleThemePreferenceChange(option.id)}
                  className={cn(
                    "rounded-xl border px-4 py-3 text-left transition-colors",
                    isActive
                      ? "border-[var(--border-strong)] bg-[var(--bg-surface-2)]"
                      : "border-border/70 bg-[var(--bg-surface-2)]/40 hover:bg-[var(--bg-surface-2)]/70",
                  )}
                >
                  <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                    <Icon className="h-4 w-4" />
                    {option.title}
                  </div>
                  <p className="mt-1 text-xs text-[var(--text-muted)]">
                    {option.description}
                  </p>
                </button>
              );
            })}
          </div>

          <p className="mt-3 text-xs uppercase tracking-[0.18em] text-[var(--text-subtle)]">
            Current effective theme: {resolvedTheme}
          </p>
        </section>

        <section className="rounded-2xl border border-border/70 bg-[var(--bg-surface-1)]/75 p-5 sm:p-6">
          <div className="mb-4">
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              App Updates
            </h2>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Izwi checks for updates automatically and can install new desktop
              builds in-app.
            </p>
          </div>

          <div className="rounded-xl border border-border/70 bg-[var(--bg-surface-2)]/40 px-4 py-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-sm text-[var(--text-secondary)]">
                <span className="block font-semibold text-[var(--text-primary)]">
                  Current version: v{APP_VERSION}
                </span>
                <span>
                  {lastCheckedAt
                    ? `Last checked ${new Date(lastCheckedAt).toLocaleString()}`
                    : "No update check has completed yet."}
                </span>
                {updaterHealth && !updaterHealth.enabled ? (
                  <span className="mt-1 block text-[var(--status-warning-text)]">
                    Updates disabled:{" "}
                    {updaterHealth.disableReason ?? "runtime policy"}
                  </span>
                ) : null}
                {updateErrorMessage ? (
                  <span className="mt-1 block text-[var(--status-warning-text)]">
                    Last error: {updateErrorMessage}
                  </span>
                ) : null}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className={cn(
                    "rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors",
                    "border-border/80 bg-[var(--bg-surface-2)] hover:bg-[var(--bg-surface-3)]",
                    (updateStatus === "checking" ||
                      updateStatus === "downloading") &&
                      "cursor-not-allowed opacity-60",
                  )}
                  disabled={
                    updateStatus === "checking" || updateStatus === "downloading"
                  }
                  onClick={() => void checkForUpdates(true)}
                >
                  {updateStatus === "checking"
                    ? "Checking..."
                    : "Check for updates"}
                </button>
                {availableUpdate ? (
                  <button
                    type="button"
                    className="rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-surface-3)]"
                    onClick={openUpdatePrompt}
                  >
                    View {availableUpdate.version}
                  </button>
                ) : null}
              </div>
            </div>
            {updaterHealth ? (
              <div className="mt-3 text-[11px] leading-5 text-[var(--text-subtle)]">
                timeout={updaterHealth.requestTimeoutMs}ms • attempts=
                {updaterHealth.maxCheckAttempts} • backoff=
                {updaterHealth.retryBackoffMs}ms
                {updaterHealth.forcedManifestUrl
                  ? ` • override=${updaterHealth.forcedManifestUrl}`
                  : ""}
              </div>
            ) : null}
          </div>
        </section>

        <section className="rounded-2xl border border-border/70 bg-[var(--bg-surface-1)]/75 p-5 sm:p-6">
          <div className="mb-4">
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              Privacy and Analytics
            </h2>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Help improve Izwi with anonymous usage metrics. No prompts,
              transcripts, audio, or personal identifiers are collected.
            </p>
          </div>

          {isLoadingPreferences ? (
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading preference...
            </div>
          ) : (
            <label className="flex items-start gap-3 rounded-xl border border-border/70 bg-[var(--bg-surface-2)]/40 px-4 py-3">
              <input
                type="checkbox"
                className="app-checkbox mt-0.5 h-4 w-4"
                checked={analyticsOptIn}
                disabled={isSavingPreference}
                onChange={(event) =>
                  void handleAnalyticsToggle(event.target.checked)
                }
              />
              <span className="text-sm text-[var(--text-secondary)]">
                <span className="block font-semibold text-[var(--text-primary)]">
                  Share anonymous usage data
                </span>
                {isSavingPreference
                  ? "Saving your preference..."
                  : "You can change this at any time."}
              </span>
            </label>
          )}
        </section>
      </div>
    </PageShell>
  );
}
