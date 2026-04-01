import { useCallback, useEffect, useMemo, useState } from "react";

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

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextRecords = await api.listTextToSpeechRecords();
      setRecords(nextRecords);
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
