import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  const recordRef = useRef<TranscriptionRecord | null>(null);

  useEffect(() => {
    recordRef.current = record;
  }, [record]);

  const loadRecord = useCallback(async (background = false) => {
    if (!recordId) {
      setRecord(null);
      setLoading(false);
      setError(null);
      return;
    }

    const hasVisibleRecord = recordRef.current !== null;
    if (!background || !hasVisibleRecord) {
      setLoading(true);
      setError(null);
    }

    try {
      const nextRecord = await api.getTranscriptionRecord(recordId);
      setRecord(nextRecord);
      setError(null);
    } catch (err) {
      if (!background || !hasVisibleRecord) {
        setRecord(null);
      }
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load transcription record.",
      );
    } finally {
      setLoading(false);
    }
  }, [recordId]);

  const refresh = useCallback(async () => {
    await loadRecord(recordRef.current !== null);
  }, [loadRecord]);

  useEffect(() => {
    recordRef.current = null;
    setRecord(null);
    setLoading(Boolean(recordId));
    setError(null);
    void loadRecord(false);
  }, [loadRecord, recordId]);

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
      void loadRecord(true);
    }, 2500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [loadRecord, pollingRequired, recordId]);

  return {
    record,
    loading,
    error,
    refresh,
  };
}
