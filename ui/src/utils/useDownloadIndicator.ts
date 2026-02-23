import { useCallback, useEffect, useRef, useState } from "react";

type DownloadState = "idle" | "downloading" | "success" | "error";

const SUCCESS_TIMEOUT_MS = 2200;
const ERROR_TIMEOUT_MS = 3600;

function formatDownloadError(error: unknown): string {
  if (error instanceof Error && error.message.trim()) {
    return `Download failed: ${error.message.trim()}`;
  }
  return "Download failed";
}

export function useDownloadIndicator() {
  const [downloadState, setDownloadState] = useState<DownloadState>("idle");
  const [downloadMessage, setDownloadMessage] = useState<string | null>(null);
  const clearTimerRef = useRef<number | null>(null);

  const clearTimer = useCallback(() => {
    if (clearTimerRef.current !== null) {
      window.clearTimeout(clearTimerRef.current);
      clearTimerRef.current = null;
    }
  }, []);

  const scheduleReset = useCallback(
    (delayMs: number) => {
      clearTimer();
      clearTimerRef.current = window.setTimeout(() => {
        setDownloadState("idle");
        setDownloadMessage(null);
        clearTimerRef.current = null;
      }, delayMs);
    },
    [clearTimer],
  );

  useEffect(() => clearTimer, [clearTimer]);

  const beginDownload = useCallback(
    (message = "Downloading audio...") => {
      clearTimer();
      setDownloadState("downloading");
      setDownloadMessage(message);
    },
    [clearTimer],
  );

  const completeDownload = useCallback(
    (message = "Download complete") => {
      setDownloadState("success");
      setDownloadMessage(message);
      scheduleReset(SUCCESS_TIMEOUT_MS);
    },
    [scheduleReset],
  );

  const failDownload = useCallback(
    (error: unknown) => {
      setDownloadState("error");
      setDownloadMessage(formatDownloadError(error));
      scheduleReset(ERROR_TIMEOUT_MS);
    },
    [scheduleReset],
  );

  const clearDownloadStatus = useCallback(() => {
    clearTimer();
    setDownloadState("idle");
    setDownloadMessage(null);
  }, [clearTimer]);

  return {
    downloadState,
    downloadMessage,
    isDownloading: downloadState === "downloading",
    beginDownload,
    completeDownload,
    failDownload,
    clearDownloadStatus,
  };
}
