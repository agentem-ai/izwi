import { useCallback, useEffect, useState } from "react";

import { api, type DiarizationRecordSummary } from "@/api";

const HISTORY_PAGE_LIMIT = 25;

export interface UseDiarizationHistoryResult {
  records: DiarizationRecordSummary[];
  loading: boolean;
  loadingMore: boolean;
  error: string | null;
  hasMoreRecords: boolean;
  loadMoreRecords: () => Promise<void>;
  refresh: () => Promise<void>;
}

export function useDiarizationHistory(): UseDiarizationHistoryResult {
  const [records, setRecords] = useState<DiarizationRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await api.listDiarizationRecordPage({
        limit: HISTORY_PAGE_LIMIT,
        cursor: null,
      });
      setRecords(page.items);
      setNextCursor(page.pagination.next_cursor);
      setHasMore(page.pagination.has_more);
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

  const loadMoreRecords = useCallback(async () => {
    if (loading || loadingMore || !nextCursor || !hasMore) {
      return;
    }
    setLoadingMore(true);
    setError(null);
    try {
      const page = await api.listDiarizationRecordPage({
        limit: HISTORY_PAGE_LIMIT,
        cursor: nextCursor,
      });
      setRecords((current) => {
        const seen = new Set(current.map((record) => record.id));
        const nextItems = page.items.filter((record) => !seen.has(record.id));
        return [...current, ...nextItems];
      });
      setNextCursor(page.pagination.next_cursor);
      setHasMore(page.pagination.has_more);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load more diarization history.",
      );
    } finally {
      setLoadingMore(false);
    }
  }, [hasMore, loading, loadingMore, nextCursor]);

  return {
    records,
    loading,
    loadingMore,
    error,
    hasMoreRecords: hasMore && Boolean(nextCursor),
    loadMoreRecords,
    refresh,
  };
}
