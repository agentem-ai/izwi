import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type SpeechHistoryRecordSummary } from "@/api";
import { normalizeSpeechProcessingStatus } from "@/features/text-to-speech/support";

export interface UseTextToSpeechHistoryResult {
  records: SpeechHistoryRecordSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useTextToSpeechHistory(): UseTextToSpeechHistoryResult {
  const [records, setRecords] = useState<SpeechHistoryRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const recordsRef = useRef<SpeechHistoryRecordSummary[]>([]);

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
      const nextRecords = await api.listTextToSpeechRecords();
      setRecords(nextRecords);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load text-to-speech history.",
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
        const processingStatus = normalizeSpeechProcessingStatus(
          record.processing_status,
          record.processing_error,
        );
        return processingStatus === "pending" || processingStatus === "processing";
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
