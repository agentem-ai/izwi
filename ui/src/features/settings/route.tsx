import {
  Loader2,
  Moon,
  RefreshCw,
  ShieldCheck,
  Sun,
  SunMoon,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { api } from "@/api";
import { setAnalyticsEnabled } from "@/app/analytics/client";
import {
  trackAnalyticsConsentChanged,
  trackThemePreferenceChanged,
} from "@/app/analytics/events";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { type UpdateStatus, useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { useTheme } from "@/app/providers/ThemeProvider";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { APP_VERSION } from "@/shared/config/runtime";

const THEME_OPTIONS: Array<{
  id: "system" | "light" | "dark";
  title: string;
  icon: typeof SunMoon;
}> = [
  {
    id: "system",
    title: "Auto",
    icon: SunMoon,
  },
  {
    id: "light",
    title: "Light",
    icon: Sun,
  },
  {
    id: "dark",
    title: "Dark",
    icon: Moon,
  },
];

function formatDateTime(value: number | string | null): string {
  if (value == null) {
    return "Not checked yet";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unavailable";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function getUpdateBadge(
  status: UpdateStatus,
  hasAvailableUpdate: boolean,
  updatesEnabled: boolean,
): { label: string; tone: "neutral" | "info" | "success" | "warning" } {
  if (!updatesEnabled) {
    return { label: "Updates Off", tone: "warning" };
  }

  if (status === "checking") {
    return { label: "Checking", tone: "info" };
  }

  if (status === "downloading") {
    return { label: "Downloading", tone: "info" };
  }

  if (status === "downloaded") {
    return { label: "Ready To Restart", tone: "success" };
  }

  if (status === "error") {
    return { label: "Needs Attention", tone: "warning" };
  }

  if (status === "available" || hasAvailableUpdate) {
    return { label: "Update Available", tone: "info" };
  }

  return { label: "Current", tone: "success" };
}

function SettingsSection({
  icon,
  title,
  description,
  children,
}: {
  icon: ReactNode;
  title: string;
  description?: ReactNode;
  children: ReactNode;
}) {
  const hasIntro = Boolean(description);

  return (
    <section
      className={cn(
        "grid px-5 py-4 sm:px-6 sm:py-5 lg:grid-cols-[144px_minmax(0,1fr)]",
        hasIntro ? "gap-2" : "gap-1",
      )}
    >
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 text-[var(--text-primary)]">
          {icon}
          <h2 className="text-sm font-semibold uppercase tracking-[0.18em] text-[var(--text-subtle)]">
            {title}
          </h2>
        </div>
        {description ? (
          <p className="max-w-xs text-sm leading-6 text-[var(--text-muted)]">
            {description}
          </p>
        ) : null}
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}

function SettingsRow({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  const hasHeaderCopy = Boolean(title || description);

  return (
    <div
      className={cn(
        "border-b border-border/70 py-3 last:border-b-0 last:pb-0 first:pt-0",
        className,
      )}
    >
      {hasHeaderCopy || action ? (
        <div
          className={cn(
            "flex flex-col md:flex-row md:items-start md:justify-between",
            hasHeaderCopy && action ? "gap-3" : "gap-0",
          )}
        >
          {hasHeaderCopy ? (
            <div className="min-w-0 max-w-2xl">
              {title ? (
                <div className="text-sm font-semibold text-[var(--text-primary)]">
                  {title}
                </div>
              ) : null}
              {description ? (
                <div
                  className={cn(
                    "text-sm leading-6 text-[var(--text-muted)]",
                    title ? "mt-1" : undefined,
                  )}
                >
                  {description}
                </div>
              ) : null}
            </div>
          ) : null}
          {action ? (
            <div className="shrink-0 md:pt-0.5">{action}</div>
          ) : null}
        </div>
      ) : null}
      {children ? (
        <div
          className={cn(
            hasHeaderCopy ? "mt-3" : action ? "mt-1.5" : "mt-0",
          )}
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}

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

  const updatesEnabled = updaterHealth?.enabled ?? true;
  const updateBadge = useMemo(
    () => getUpdateBadge(updateStatus, Boolean(availableUpdate), updatesEnabled),
    [availableUpdate, updateStatus, updatesEnabled],
  );

  const handleThemePreferenceChange = (
    nextPreference: "system" | "light" | "dark",
  ) => {
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
        description="Appearance, updates, and privacy controls for this device."
      />

      <div className="overflow-hidden rounded-[24px] border border-border/70 bg-[var(--bg-surface-1)]">
        <div className="divide-y divide-border/70">
          <SettingsSection
            icon={<SunMoon className="h-4 w-4" />}
            title="Appearance"
            description="Choose how the app should render."
          >
            <SettingsRow
              action={
                <div className="text-sm font-medium text-[var(--text-secondary)]">
                  {resolvedTheme === "dark" ? "Dark" : "Light"}
                </div>
              }
            >
              <div className="grid gap-2 sm:grid-cols-3">
                {THEME_OPTIONS.map((option) => {
                  const Icon = option.icon;
                  const isActive = themePreference === option.id;

                  return (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => handleThemePreferenceChange(option.id)}
                      className={cn(
                        "group rounded-[18px] border px-4 py-3 text-left transition-all duration-150",
                        isActive
                          ? "border-[var(--border-strong)] bg-[var(--bg-surface-0)]"
                          : "border-border/70 bg-transparent hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-0)]/60",
                      )}
                      aria-pressed={isActive}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                          <Icon className="h-4 w-4" />
                          {option.title}
                        </span>
                        <span
                          className={cn(
                            "h-2.5 w-2.5 rounded-full transition-colors",
                            isActive
                              ? "bg-[var(--text-primary)]"
                              : "bg-[var(--border-strong)] group-hover:bg-[var(--text-muted)]",
                          )}
                        />
                      </div>
                    </button>
                  );
                })}
              </div>
            </SettingsRow>
          </SettingsSection>

          <SettingsSection
            icon={<RefreshCw className="h-4 w-4" />}
            title="Updates"
            description={`Current version v${APP_VERSION}`}
          >
            <SettingsRow
              description={
                availableUpdate ? (
                  <>
                    Version{" "}
                    <span className="font-medium text-[var(--text-secondary)]">
                      {availableUpdate.version}
                    </span>{" "}
                    is available.
                  </>
                ) : (
                  `Last checked ${formatDateTime(lastCheckedAt)}.`
                )
              }
              action={
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge tone={updateBadge.tone}>{updateBadge.label}</StatusBadge>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => void checkForUpdates(true)}
                    disabled={
                      updateStatus === "checking" ||
                      updateStatus === "downloading"
                    }
                  >
                    {updateStatus === "checking" ? "Checking..." : "Check now"}
                  </Button>
                  {availableUpdate ? (
                    <Button
                      type="button"
                      size="sm"
                      onClick={openUpdatePrompt}
                    >
                      View {availableUpdate.version}
                    </Button>
                  ) : null}
                </div>
              }
            >
              {!updatesEnabled ? (
                <p className="text-sm leading-6 text-[var(--status-warning-text)]">
                  {updaterHealth?.disableReason ?? "Updates are disabled."}
                </p>
              ) : null}
              {updateErrorMessage ? (
                <p className="text-sm leading-6 text-[var(--status-warning-text)]">
                  Last error: {updateErrorMessage}
                </p>
              ) : null}
            </SettingsRow>
          </SettingsSection>

          <SettingsSection
            icon={<ShieldCheck className="h-4 w-4" />}
            title="Privacy"
            description="Control anonymous usage telemetry."
          >
            <SettingsRow
              title="Anonymous analytics"
              description={
                isLoadingPreferences ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading saved preference...
                  </span>
                ) : analyticsOptIn ? (
                  "Enabled. Prompts, transcripts, audio, and personal identifiers are never sent."
                ) : (
                  "Disabled for this device."
                )
              }
              action={
                <div className="flex items-center gap-3">
                  <span className="text-sm font-medium text-[var(--text-secondary)]">
                    {analyticsOptIn ? "On" : "Off"}
                  </span>
                  <Switch
                    checked={analyticsOptIn}
                    disabled={isLoadingPreferences || isSavingPreference}
                    onCheckedChange={(checked) => void handleAnalyticsToggle(checked)}
                    aria-label="Share anonymous usage data"
                  />
                </div>
              }
            >
              {isSavingPreference ? (
                <p className="text-sm text-[var(--text-muted)]">
                  Saving your preference...
                </p>
              ) : null}
            </SettingsRow>
          </SettingsSection>
        </div>
      </div>
    </PageShell>
  );
}
