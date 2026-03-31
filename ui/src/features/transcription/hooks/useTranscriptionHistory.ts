import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type TranscriptionRecordSummary } from "@/api";
import {
  normalizeProcessingStatus,
  normalizeSummaryStatus,
} from "@/features/transcription/playground/support";

export interface UseTranscriptionHistoryResult {
  records: TranscriptionRecordSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useTranscriptionHistory(): UseTranscriptionHistoryResult {
  const [records, setRecords] = useState<TranscriptionRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextRecords = await api.listTranscriptionRecords();
      setRecords(nextRecords);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load transcription history.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const pollingRequired = useMemo(
    () =>
      records.some((record) => {
        const processingStatus = normalizeProcessingStatus(
          record.processing_status,
          record.processing_error,
        );
        if (processingStatus === "pending" || processingStatus === "processing") {
          return true;
        }

        return (
          normalizeSummaryStatus(
            record.summary_status,
            record.summary_preview,
            null,
          ) === "pending"
        );
      }),
    [records],
  );

  useEffect(() => {
    if (!pollingRequired) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refresh();
    }, 2500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [pollingRequired, refresh]);

  return {
    records,
    loading,
    error,
    refresh,
  };
}
