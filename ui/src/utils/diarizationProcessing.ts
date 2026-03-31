import type { DiarizationProcessingStatus } from "@/api";

export function normalizeDiarizationProcessingStatus(
  status: DiarizationProcessingStatus | null | undefined,
  error: string | null | undefined,
): DiarizationProcessingStatus {
  if ((error ?? "").trim().length > 0) {
    return "failed";
  }

  switch (status) {
    case "pending":
    case "processing":
    case "failed":
    case "ready":
      return status;
    default:
      return "ready";
  }
}
