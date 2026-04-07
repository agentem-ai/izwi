import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type SpeechHistoryRecordSummary } from "@/api";
import { normalizeSpeechProcessingStatus } from "@/features/text-to-speech/support";

const HISTORY_PAGE_LIMIT = 25;

export interface UseTextToSpeechHistoryResult {
  records: SpeechHistoryRecordSummary[];
  loading: boolean;
  loadingMore: boolean;
  error: string | null;
  hasMoreRecords: boolean;
  loadMoreRecords: () => Promise<void>;
  refresh: () => Promise<void>;
}

export function useTextToSpeechHistory(): UseTextToSpeechHistoryResult {
  const [records, setRecords] = useState<SpeechHistoryRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const recordsRef = useRef<SpeechHistoryRecordSummary[]>([]);

  useEffect(() => {
    recordsRef.current = records;
  }, [records]);

  const loadFirstPage = useCallback(
    async (background = false) => {
      const hasVisibleRecords = recordsRef.current.length > 0;
      const backgroundRefresh = background && hasVisibleRecords;
      if (!backgroundRefresh) {
        setLoading(true);
        setError(null);
      }

      try {
        const page = await api.listTextToSpeechRecordPage({
          limit: HISTORY_PAGE_LIMIT,
          cursor: null,
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
    await loadFirstPage(false);
  }, [loadFirstPage]);

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

  const pollingEnabled = records.length <= HISTORY_PAGE_LIMIT;

  useEffect(() => {
    if (!pollingRequired || !pollingEnabled) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadFirstPage(true);
    }, 2500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [loadFirstPage, pollingEnabled, pollingRequired]);

  const loadMoreRecords = useCallback(async () => {
    if (loading || loadingMore || !nextCursor || !hasMore) {
      return;
    }
    setLoadingMore(true);
    setError(null);
    try {
      const page = await api.listTextToSpeechRecordPage({
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
          : "Failed to load more text-to-speech history.",
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
