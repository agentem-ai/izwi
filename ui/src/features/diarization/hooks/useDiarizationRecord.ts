import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type DiarizationRecord } from "@/api";
import { normalizeDiarizationSummaryStatus } from "@/utils/diarizationSummary";

export interface UseDiarizationRecordResult {
  record: DiarizationRecord | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useDiarizationRecord(
  recordId: string | null | undefined,
): UseDiarizationRecordResult {
  const [record, setRecord] = useState<DiarizationRecord | null>(null);
  const [loading, setLoading] = useState(Boolean(recordId));
  const [error, setError] = useState<string | null>(null);
  const recordRef = useRef<DiarizationRecord | null>(null);

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
        const nextRecord = await api.getDiarizationRecord(recordId);
        setRecord(nextRecord);
        setError(null);
      } catch (err) {
        if (!background || !hasVisibleRecord) {
          setRecord(null);
        }
        setError(
          err instanceof Error ? err.message : "Failed to load diarization record.",
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

  const pollingRequired = useMemo(
    () =>
      !!record &&
      normalizeDiarizationSummaryStatus(
        record.summary_status,
        record.summary_text,
        record.summary_error,
      ) === "pending",
    [record],
  );

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
