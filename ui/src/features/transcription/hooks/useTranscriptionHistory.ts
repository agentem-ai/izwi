import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  const recordsRef = useRef<TranscriptionRecordSummary[]>([]);

  useEffect(() => {
    recordsRef.current = records;
  }, [records]);

  const loadRecords = useCallback(async (background = false) => {
    const hasVisibleRecords = recordsRef.current.length > 0;
    const backgroundRefresh = background && hasVisibleRecords;
    if (!backgroundRefresh) {
      setLoading(true);
      setError(null);
    }

    try {
      const nextRecords = await api.listTranscriptionRecords();
      setRecords(nextRecords);
      setError(null);
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

  const refresh = useCallback(async () => {
    await loadRecords(false);
  }, [loadRecords]);

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
      void loadRecords(true);
    }, 2500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [loadRecords, pollingRequired]);

  return {
    records,
    loading,
    error,
    refresh,
  };
}
