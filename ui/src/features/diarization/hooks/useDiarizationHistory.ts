import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type DiarizationRecordSummary } from "@/api";
import { normalizeDiarizationSummaryStatus } from "@/utils/diarizationSummary";

export interface UseDiarizationHistoryResult {
  records: DiarizationRecordSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useDiarizationHistory(): UseDiarizationHistoryResult {
  const [records, setRecords] = useState<DiarizationRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextRecords = await api.listDiarizationRecords();
      setRecords(nextRecords);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load diarization history.",
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
      records.some(
        (record) =>
          normalizeDiarizationSummaryStatus(
            record.summary_status,
            record.summary_preview,
            null,
          ) === "pending",
      ),
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
