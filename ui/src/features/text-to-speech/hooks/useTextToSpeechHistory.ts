import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type SpeechHistoryRecordSummary } from "@/api";
import { normalizeSpeechProcessingStatus } from "@/features/text-to-speech/support";

const HISTORY_PAGE_LIMIT = 25;

export interface UseTextToSpeechHistoryResult {
  records: SpeechHistoryRecordSummary[];
  loading: boolean;
  error: string | null;
  currentPage: number;
  canGoPreviousPage: boolean;
  canGoNextPage: boolean;
  goToPreviousPage: () => void;
  goToNextPage: () => void;
  refresh: () => Promise<void>;
}

export function useTextToSpeechHistory(): UseTextToSpeechHistoryResult {
  const [records, setRecords] = useState<SpeechHistoryRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [pageCursors, setPageCursors] = useState<Array<string | null>>([null]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const recordsRef = useRef<SpeechHistoryRecordSummary[]>([]);
  const currentCursor = pageCursors[pageIndex] ?? null;

  useEffect(() => {
    recordsRef.current = records;
  }, [records]);

  const loadRecords = useCallback(
    async (background = false, cursor: string | null = null) => {
      const hasVisibleRecords = recordsRef.current.length > 0;
      const backgroundRefresh = background && hasVisibleRecords;
      if (!backgroundRefresh) {
        setLoading(true);
        setError(null);
      }

      try {
        const page = await api.listTextToSpeechRecordPage({
          limit: HISTORY_PAGE_LIMIT,
          cursor,
        });
        setRecords(page.items);
        setNextCursor(page.pagination.next_cursor);
        setHasMore(page.pagination.has_more);
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
    },
    [],
  );

  const refresh = useCallback(async () => {
    await loadRecords(false, currentCursor);
  }, [currentCursor, loadRecords]);

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
    if (!pollingRequired || pageIndex !== 0) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadRecords(true, currentCursor);
    }, 2500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [currentCursor, loadRecords, pageIndex, pollingRequired]);

  useEffect(() => {
    if (loading || error || pageIndex === 0 || records.length > 0) {
      return;
    }
    setPageIndex((current) => Math.max(0, current - 1));
  }, [error, loading, pageIndex, records.length]);

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
