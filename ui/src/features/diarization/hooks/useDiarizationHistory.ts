import { useCallback, useEffect, useState } from "react";

import { api, type DiarizationRecordSummary } from "@/api";

const HISTORY_PAGE_LIMIT = 25;

export interface UseDiarizationHistoryResult {
  records: DiarizationRecordSummary[];
  loading: boolean;
  error: string | null;
  currentPage: number;
  canGoPreviousPage: boolean;
  canGoNextPage: boolean;
  goToPreviousPage: () => void;
  goToNextPage: () => void;
  refresh: () => Promise<void>;
}

export function useDiarizationHistory(): UseDiarizationHistoryResult {
  const [records, setRecords] = useState<DiarizationRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [pageCursors, setPageCursors] = useState<Array<string | null>>([null]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const currentCursor = pageCursors[pageIndex] ?? null;

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await api.listDiarizationRecordPage({
        limit: HISTORY_PAGE_LIMIT,
        cursor: currentCursor,
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
  }, [currentCursor]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const goToPreviousPage = useCallback(() => {
    setPageIndex((current) => Math.max(0, current - 1));
  }, []);

  const canGoNextPage = hasMore && Boolean(nextCursor);
  const goToNextPage = useCallback(() => {
    if (!nextCursor || !hasMore) {
      return;
    }
    setPageCursors((current) => {
      const next = [...current];
      next[pageIndex + 1] = nextCursor;
      return next;
    });
    setPageIndex((current) => current + 1);
  }, [hasMore, nextCursor, pageIndex]);

  return {
    records,
    loading,
    error,
    currentPage: pageIndex + 1,
    canGoPreviousPage: pageIndex > 0,
    canGoNextPage,
    goToPreviousPage,
    goToNextPage,
    refresh,
  };
}
