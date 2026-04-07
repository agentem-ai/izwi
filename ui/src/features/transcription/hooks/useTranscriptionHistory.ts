import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type TranscriptionRecordSummary } from "@/api";
import {
  normalizeProcessingStatus,
  normalizeSummaryStatus,
} from "@/features/transcription/playground/support";

const HISTORY_PAGE_LIMIT = 25;

export interface UseTranscriptionHistoryResult {
  records: TranscriptionRecordSummary[];
  loading: boolean;
  error: string | null;
  currentPage: number;
  canGoPreviousPage: boolean;
  canGoNextPage: boolean;
  goToPreviousPage: () => void;
  goToNextPage: () => void;
  refresh: () => Promise<void>;
}

export function useTranscriptionHistory(): UseTranscriptionHistoryResult {
  const [records, setRecords] = useState<TranscriptionRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [pageCursors, setPageCursors] = useState<Array<string | null>>([null]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const recordsRef = useRef<TranscriptionRecordSummary[]>([]);
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
        const page = await api.listTranscriptionRecordPage({
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
            : "Failed to load transcription history.",
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
            record.summary_preview,
            null,
          ) === "pending"
        );
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
