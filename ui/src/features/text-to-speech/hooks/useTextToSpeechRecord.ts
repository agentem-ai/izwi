import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type SpeechHistoryRecord } from "@/api";
import { normalizeSpeechProcessingStatus } from "@/features/text-to-speech/support";

export interface UseTextToSpeechRecordResult {
  record: SpeechHistoryRecord | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useTextToSpeechRecord(
  recordId: string | null | undefined,
): UseTextToSpeechRecordResult {
  const [record, setRecord] = useState<SpeechHistoryRecord | null>(null);
  const [loading, setLoading] = useState(Boolean(recordId));
  const [error, setError] = useState<string | null>(null);
  const recordRef = useRef<SpeechHistoryRecord | null>(null);

  useEffect(() => {
    recordRef.current = record;
  }, [record]);

  const loadRecord = useCallback(
    async (background = false) => {
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
        const nextRecord = await api.getTextToSpeechRecord(recordId);
        setRecord(nextRecord);
        setError(null);
      } catch (err) {
        if (!background || !hasVisibleRecord) {
          setRecord(null);
        }
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load text-to-speech record.",
        );
      } finally {
        setLoading(false);
      }
    },
    [recordId],
  );

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

    const processingStatus = normalizeSpeechProcessingStatus(
      record.processing_status,
      record.processing_error,
    );
    return processingStatus === "pending" || processingStatus === "processing";
  }, [record]);

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
