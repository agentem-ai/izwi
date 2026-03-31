import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type TranscriptionRecord } from "@/api";
import {
  normalizeProcessingStatus,
  normalizeSummaryStatus,
} from "@/features/transcription/playground/support";

export interface UseTranscriptionRecordResult {
  record: TranscriptionRecord | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useTranscriptionRecord(
  recordId: string | null | undefined,
): UseTranscriptionRecordResult {
  const [record, setRecord] = useState<TranscriptionRecord | null>(null);
  const [loading, setLoading] = useState(Boolean(recordId));
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!recordId) {
      setRecord(null);
      setLoading(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const nextRecord = await api.getTranscriptionRecord(recordId);
      setRecord(nextRecord);
    } catch (err) {
      setRecord(null);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load transcription record.",
      );
    } finally {
      setLoading(false);
    }
  }, [recordId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const pollingRequired = useMemo(() => {
    if (!record) {
      return false;
    }

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
        record.summary_text,
        record.summary_error,
      ) === "pending"
    );
  }, [
    record,
  ]);

  useEffect(() => {
    if (!recordId || !pollingRequired) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refresh();
    }, 2500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [pollingRequired, recordId, refresh]);

  return {
    record,
    loading,
    error,
    refresh,
  };
}
