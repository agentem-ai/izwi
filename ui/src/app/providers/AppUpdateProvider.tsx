import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { isTauri } from "@tauri-apps/api/core";

import {
  trackUpdateCheckCompleted,
  trackUpdateCheckStarted,
  trackUpdateInstallCompleted,
  trackUpdateInstallFailed,
  trackUpdateInstallStarted,
} from "@/app/analytics/events";
import { useNotifications } from "@/app/providers/NotificationProvider";
import {
  checkForBetaUpdate,
  getUpdaterHealthSnapshot,
  installBetaUpdate,
  relaunchAfterUpdate,
  type AppUpdateMetadata,
  type UpdaterHealthStatus,
} from "@/app/updates/client";

const UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;

export type UpdateStatus =
  | "idle"
  | "checking"
  | "available"
  | "downloading"
  | "downloaded"
  | "error";

interface AppUpdateContextValue {
  availableUpdate: AppUpdateMetadata | null;
  status: UpdateStatus;
  lastCheckedAt: number | null;
  errorMessage: string | null;
  isPromptOpen: boolean;
  progressPercent: number | null;
  health: UpdaterHealthStatus | null;
  checkForUpdates: (manual?: boolean) => Promise<void>;
  installUpdate: () => Promise<void>;
  restartToApply: () => Promise<void>;
  openPrompt: () => void;
  dismissPrompt: () => void;
}

const AppUpdateContext = createContext<AppUpdateContextValue | null>(null);

interface AppUpdateProviderProps {
  children: ReactNode;
}

export function AppUpdateProvider({ children }: AppUpdateProviderProps) {
  const { notify } = useNotifications();
  const [availableUpdate, setAvailableUpdate] = useState<AppUpdateMetadata | null>(
    null,
  );
  const [status, setStatus] = useState<UpdateStatus>("idle");
  const [lastCheckedAt, setLastCheckedAt] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPromptOpen, setIsPromptOpen] = useState(false);
  const [progressPercent, setProgressPercent] = useState<number | null>(null);
  const [health, setHealth] = useState<UpdaterHealthStatus | null>(null);
  const statusRef = useRef<UpdateStatus>("idle");
  const healthRef = useRef<UpdaterHealthStatus | null>(null);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    healthRef.current = health;
  }, [health]);

  const checkForUpdates = useCallback(
    async (manual = false) => {
      if (!isTauri()) {
        return;
      }
      const currentHealth = healthRef.current;
      if (currentHealth && !currentHealth.enabled) {
        statusRef.current = "error";
        setStatus("error");
        setErrorMessage(currentHealth.disableReason ?? "Updates are disabled.");
        if (manual) {
          notify({
            title: "Updates are disabled",
            description:
              currentHealth.disableReason ?? "Runtime policy has disabled updates.",
            tone: "warning",
          });
        }
        return;
      }
      const currentStatus = statusRef.current;
      if (currentStatus === "checking" || currentStatus === "downloading") {
        return;
      }

      statusRef.current = "checking";
      setStatus("checking");
      setErrorMessage(null);
      setProgressPercent(null);
      void trackUpdateCheckStarted(manual ? "manual" : "background");

      try {
        const update = await checkForBetaUpdate();
        setLastCheckedAt(Date.now());

        if (!update) {
          setAvailableUpdate(null);
          statusRef.current = "idle";
          setStatus("idle");
          void trackUpdateCheckCompleted("no_update");
          if (manual) {
            notify({
              title: "You’re up to date",
              description: "No new beta update is available right now.",
              tone: "success",
            });
          }
          return;
        }

        setAvailableUpdate(update);
        statusRef.current = "available";
        setStatus("available");
        setIsPromptOpen(true);
        void trackUpdateCheckCompleted("update_available", update.version);

        if (!manual) {
          notify({
            title: `Update available: ${update.version}`,
            description: "Open the update prompt to install this release.",
            tone: "info",
          });
        }
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unknown update-check error.";
        statusRef.current = "error";
        setStatus("error");
        setErrorMessage(message);
        void trackUpdateCheckCompleted("failed", undefined, message);
        if (manual) {
          notify({
            title: "Could not check for updates",
            description: message,
            tone: "warning",
          });
        }
      }
    },
    [notify],
  );

  const installUpdate = useCallback(async () => {
    if (!availableUpdate || !isTauri()) {
      return;
    }
    if (status === "downloading") {
      return;
    }

    setStatus("downloading");
    setErrorMessage(null);
    setProgressPercent(0);
    void trackUpdateInstallStarted(availableUpdate.version);

    let downloadedBytes = 0;
    let contentLength: number | null = null;

    try {
      const result = await installBetaUpdate((event) => {
        if (event.event === "Started") {
          contentLength = event.data.contentLength ?? null;
          setProgressPercent(contentLength ? 0 : null);
          return;
        }
        if (event.event === "Progress") {
          downloadedBytes += event.data.chunkLength;
          contentLength = event.data.contentLength ?? contentLength;
          if (contentLength && contentLength > 0) {
            const nextPercent = Math.min(
              100,
              Math.round((downloadedBytes / contentLength) * 100),
            );
            setProgressPercent(nextPercent);
          }
          return;
        }
        setProgressPercent(100);
      });

      void trackUpdateInstallCompleted(availableUpdate.version);

      if (result.appExitsDuringInstall) {
        notify({
          title: "Applying update",
          description:
            "Windows will close Izwi to run the installer automatically.",
          tone: "info",
        });
        return;
      }

      setStatus("downloaded");
      setIsPromptOpen(true);
      notify({
        title: "Update installed",
        description: "Restart Izwi when you’re ready to apply it.",
        tone: "success",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown update-install error.";
      setStatus("available");
      setErrorMessage(message);
      void trackUpdateInstallFailed(message);
      notify({
        title: "Could not install update",
        description: message,
        tone: "warning",
      });
    }
  }, [availableUpdate, notify, status]);

  const restartToApply = useCallback(async () => {
    await relaunchAfterUpdate();
  }, []);

  const openPrompt = useCallback(() => {
    if (!availableUpdate) {
      return;
    }
    setIsPromptOpen(true);
  }, [availableUpdate]);

  const dismissPrompt = useCallback(() => {
    setIsPromptOpen(false);
  }, []);

  useEffect(() => {
    if (!isTauri()) {
      return;
    }

    getUpdaterHealthSnapshot()
      .then((snapshot) => {
        healthRef.current = snapshot;
        setHealth(snapshot);
      })
      .catch(() => {
        healthRef.current = null;
        setHealth(null);
      });

    void checkForUpdates(false);
    const timer = window.setInterval(() => {
      void checkForUpdates(false);
    }, UPDATE_CHECK_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [checkForUpdates]);

  const value = useMemo<AppUpdateContextValue>(
    () => ({
      availableUpdate,
      status,
      lastCheckedAt,
      errorMessage,
      isPromptOpen,
      progressPercent,
      health,
      checkForUpdates,
      installUpdate,
      restartToApply,
      openPrompt,
      dismissPrompt,
    }),
    [
      availableUpdate,
      status,
      lastCheckedAt,
      errorMessage,
      isPromptOpen,
      progressPercent,
      health,
      checkForUpdates,
      installUpdate,
      restartToApply,
      openPrompt,
      dismissPrompt,
    ],
  );

  return (
    <AppUpdateContext.Provider value={value}>{children}</AppUpdateContext.Provider>
  );
}

export function useAppUpdates() {
  const context = useContext(AppUpdateContext);

  if (!context) {
    throw new Error("useAppUpdates must be used within AppUpdateProvider");
  }

  return context;
}
