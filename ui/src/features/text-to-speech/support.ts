import { type SpeechHistoryProcessingStatus } from "@/api";

export function normalizeSpeechProcessingStatus(
  status: SpeechHistoryProcessingStatus | null | undefined,
  error: string | null | undefined,
): SpeechHistoryProcessingStatus {
  if (status === "pending" || status === "processing" || status === "failed") {
    return status;
  }
  if (error) {
    return "failed";
  }
  return "ready";
}

export function speechProcessingStatusLabel(
  status: SpeechHistoryProcessingStatus,
): string {
  switch (status) {
    case "pending":
      return "Queued";
    case "processing":
      return "Processing";
    case "failed":
      return "Failed";
    case "ready":
    default:
      return "Ready";
  }
}

export function formatSpeechCreatedAt(timestampMs: number): string {
  if (!Number.isFinite(timestampMs)) {
    return "Unknown time";
  }

  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown time";
  }

  return value.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatSpeechDuration(durationSecs: number | null): string {
  if (
    durationSecs === null ||
    !Number.isFinite(durationSecs) ||
    durationSecs < 0
  ) {
    return "Unknown length";
  }

  if (durationSecs < 60) {
    return `${durationSecs.toFixed(1)}s`;
  }

  const minutes = Math.floor(durationSecs / 60);
  const seconds = Math.floor(durationSecs % 60);
  return `${minutes}m ${seconds}s`;
}
