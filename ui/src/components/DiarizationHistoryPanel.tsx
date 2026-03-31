import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Loader2,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import clsx from "clsx";

import { RouteHistoryDrawer } from "@/components/RouteHistoryDrawer";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { StatusBadge } from "@/components/ui/status-badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  api,
  type DiarizationRecord,
  type DiarizationRecordRerunRequest,
  type DiarizationRecordSummary,
  type ModelInfo,
} from "@/api";
import {
  formattedTranscriptFromRecord,
  previewTranscript,
  transcriptEntriesFromRecord,
} from "../utils/diarizationTranscript";
import {
  buildDiarizationSummaryPreview,
  diarizationSummaryStatusLabel,
  diarizationSummaryStatusTone,
  normalizeDiarizationSummaryStatus,
} from "../utils/diarizationSummary";
import { DiarizationExportDialog } from "./DiarizationExportDialog";
import { DiarizationQualityPanel } from "./DiarizationQualityPanel";
import { DiarizationReviewWorkspace } from "./DiarizationReviewWorkspace";
import { DiarizationSpeakerManager } from "./DiarizationSpeakerManager";

interface DiarizationHistoryPanelProps {
  latestRecord?: DiarizationRecord | null;
  historyRecords?: DiarizationRecordSummary[];
  historyLoading?: boolean;
  historyError?: string | null;
  selectedRecordId?: string | null;
  selectedRecord?: DiarizationRecord | null;
  selectedRecordLoading?: boolean;
  selectedRecordError?: string | null;
  summaryModelReady?: boolean;
  summaryModelStatus?: ModelInfo["status"] | null;
  summaryModelId?: string | null;
  onOpenRecord: (recordId: string) => void;
  onCloseRecord: () => void;
  onDeleteRecord: (recordId: string) => Promise<void> | void;
  onSaveSpeakerCorrections: (
    recordId: string,
    speakerNameOverrides: Record<string, string>,
  ) => Promise<void> | void;
  onRerunRecord: (
    recordId: string,
    request: DiarizationRecordRerunRequest,
  ) => Promise<void> | void;
  onRegenerateSummary: (recordId: string) => Promise<void> | void;
  onSummaryModelRequired?: () => void;
  historyActionContainer?: HTMLElement | null;
}

function formatCreatedAt(timestampMs: number): string {
  if (!Number.isFinite(timestampMs)) {
    return "Unknown time";
  }
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown time";
  }
  return value.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatAudioDuration(durationSecs: number | null): string {
  if (
    durationSecs === null ||
    !Number.isFinite(durationSecs) ||
    durationSecs < 0
  ) {
    return "Unknown length";
  }
  if (durationSecs < 60) {
    return `${durationSecs.toFixed(1)}s`;
  }
  const minutes = Math.floor(durationSecs / 60);
  const seconds = Math.floor(durationSecs % 60);
  return `${minutes}m ${seconds}s`;
}

function summarizeRecord(record: DiarizationRecord): DiarizationRecordSummary {
  const entries = transcriptEntriesFromRecord(record);
  const preview = previewTranscript(
    entries,
    record.transcript ?? "",
    record.raw_transcript ?? "",
  );
  const formatted = formattedTranscriptFromRecord(record);

  return {
    id: record.id,
    created_at: record.created_at,
    model_id: record.model_id,
    speaker_count: record.speaker_count ?? 0,
    corrected_speaker_count:
      record.corrected_speaker_count ?? record.speaker_count ?? 0,
    speaker_name_override_count: Object.keys(
      record.speaker_name_overrides ?? {},
    ).length,
    duration_secs: record.duration_secs,
    processing_time_ms: record.processing_time_ms,
    rtf: record.rtf,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
    transcript_preview: preview || "No transcript",
    transcript_chars: Array.from(formatted).length,
    summary_status: normalizeDiarizationSummaryStatus(
      record.summary_status,
      record.summary_text,
      record.summary_error,
    ),
    summary_preview: buildDiarizationSummaryPreview(record.summary_text),
    summary_chars: Array.from(record.summary_text ?? "").length,
  };
}

export function DiarizationHistoryPanel({
  latestRecord = null,
  historyRecords = [],
  historyLoading = false,
  historyError = null,
  selectedRecordId = null,
  selectedRecord = null,
  selectedRecordLoading = false,
  selectedRecordError = null,
  summaryModelReady = true,
  summaryModelStatus = null,
  summaryModelId = "Qwen3.5-4B",
  onOpenRecord,
  onCloseRecord,
  onDeleteRecord,
  onSaveSpeakerCorrections,
  onRerunRecord,
  onRegenerateSummary,
  onSummaryModelRequired,
  historyActionContainer,
}: DiarizationHistoryPanelProps) {
  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState(false);
  const [historyTranscriptCopied, setHistoryTranscriptCopied] = useState(false);
  const [recordWorkspaceTab, setRecordWorkspaceTab] = useState("transcript");
  const [speakerUpdatePending, setSpeakerUpdatePending] = useState(false);
  const [speakerUpdateError, setSpeakerUpdateError] = useState<string | null>(
    null,
  );
  const [rerunPending, setRerunPending] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [summaryRefreshPendingId, setSummaryRefreshPendingId] = useState<
    string | null
  >(null);
  const [summaryRefreshError, setSummaryRefreshError] = useState<string | null>(
    null,
  );
  const [deleteTargetRecordId, setDeleteTargetRecordId] = useState<
    string | null
  >(null);
  const [deleteRecordPending, setDeleteRecordPending] = useState(false);
  const [deleteRecordError, setDeleteRecordError] = useState<string | null>(
    null,
  );

  const summaryModelGuidance = useMemo(() => {
    if (summaryModelReady) {
      return null;
    }
    const modelName = summaryModelId || "Qwen3.5-4B";
    switch (summaryModelStatus) {
      case "downloaded":
        return `Load ${modelName} in Diarization Models to generate summaries.`;
      case "downloading":
        return `${modelName} is downloading. Wait for download to complete, then try again.`;
      case "loading":
        return `${modelName} is loading. Wait until it is ready, then try again.`;
      case "not_downloaded":
      case "error":
      default:
        return `Download and load ${modelName} in Diarization Models to generate summaries.`;
    }
  }, [summaryModelId, summaryModelReady, summaryModelStatus]);

  const requireSummaryModel = useCallback(() => {
    if (summaryModelReady) {
      return true;
    }
    onSummaryModelRequired?.();
    setSummaryRefreshError(
      summaryModelGuidance ||
        "Download and load Qwen3.5-4B in Diarization Models to generate summaries.",
    );
    return false;
  }, [onSummaryModelRequired, summaryModelGuidance, summaryModelReady]);

  const visibleHistoryRecords = useMemo(() => {
    const nextRecords = latestRecord
      ? [summarizeRecord(latestRecord), ...historyRecords]
      : historyRecords;
    const deduped = nextRecords.filter(
      (record, index, list) =>
        list.findIndex((candidate) => candidate.id === record.id) === index,
    );
    deduped.sort((left, right) => right.created_at - left.created_at);
    return deduped;
  }, [historyRecords, latestRecord]);

  const selectedHistorySummary = useMemo(
    () =>
      selectedRecordId
        ? (visibleHistoryRecords.find((record) => record.id === selectedRecordId) ??
          null)
        : null,
    [selectedRecordId, visibleHistoryRecords],
  );

  const activeHistoryRecord = useMemo(() => {
    if (!selectedRecordId) {
      return null;
    }
    if (selectedRecord?.id === selectedRecordId) {
      return selectedRecord;
    }
    if (latestRecord?.id === selectedRecordId) {
      return latestRecord;
    }
    return null;
  }, [latestRecord, selectedRecord, selectedRecordId]);

  const activeHistorySummaryStatus = useMemo(
    () =>
      normalizeDiarizationSummaryStatus(
        activeHistoryRecord?.summary_status,
        activeHistoryRecord?.summary_text,
        activeHistoryRecord?.summary_error,
      ),
    [
      activeHistoryRecord?.summary_error,
      activeHistoryRecord?.summary_status,
      activeHistoryRecord?.summary_text,
    ],
  );

  const deleteTargetRecord = useMemo(() => {
    if (!deleteTargetRecordId) {
      return null;
    }
    const fromSummary = visibleHistoryRecords.find(
      (record) => record.id === deleteTargetRecordId,
    );
    if (fromSummary) {
      return fromSummary;
    }
    if (
      activeHistoryRecord &&
      activeHistoryRecord.id === deleteTargetRecordId
    ) {
      return summarizeRecord(activeHistoryRecord);
    }
    return null;
  }, [activeHistoryRecord, deleteTargetRecordId, visibleHistoryRecords]);

  const selectedHistoryAudioUrl = useMemo(
    () =>
      selectedRecordId ? api.diarizationRecordAudioUrl(selectedRecordId) : null,
    [selectedRecordId],
  );

  const selectedHistoryIndex = useMemo(
    () =>
      selectedRecordId
        ? visibleHistoryRecords.findIndex((record) => record.id === selectedRecordId)
        : -1,
    [selectedRecordId, visibleHistoryRecords],
  );

  const canOpenNewerHistory = selectedHistoryIndex > 0;
  const canOpenOlderHistory =
    selectedHistoryIndex >= 0 &&
    selectedHistoryIndex < visibleHistoryRecords.length - 1;

  const normalizedActiveTranscript = useMemo(
    () =>
      activeHistoryRecord
        ? formattedTranscriptFromRecord(activeHistoryRecord)
        : "",
    [activeHistoryRecord],
  );

  useEffect(() => {
    if (!selectedRecordId) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onCloseRecord();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [onCloseRecord, selectedRecordId]);

  useEffect(() => {
    if (!selectedRecordId) {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [selectedRecordId]);

  useEffect(() => {
    setHistoryTranscriptCopied(false);
    setRecordWorkspaceTab("transcript");
    setSpeakerUpdateError(null);
    setRerunError(null);
    setSummaryRefreshError(null);
    setSummaryRefreshPendingId(null);
  }, [selectedRecordId]);

  const handleHistoryDrawerOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen && deleteTargetRecordId) {
        return;
      }
      setIsHistoryDrawerOpen(nextOpen);
    },
    [deleteTargetRecordId],
  );

  const openDeleteRecordConfirm = useCallback((recordId: string) => {
    setDeleteTargetRecordId(recordId);
    setDeleteRecordError(null);
  }, []);

  const closeDeleteRecordConfirm = useCallback(() => {
    if (deleteRecordPending) {
      return;
    }
    setDeleteTargetRecordId(null);
    setDeleteRecordError(null);
  }, [deleteRecordPending]);

  const openAdjacentHistoryRecord = useCallback(
    (direction: "newer" | "older") => {
      if (selectedHistoryIndex < 0) {
        return;
      }
      const targetIndex =
        direction === "newer"
          ? selectedHistoryIndex - 1
          : selectedHistoryIndex + 1;
      if (targetIndex < 0 || targetIndex >= visibleHistoryRecords.length) {
        return;
      }
      const target = visibleHistoryRecords[targetIndex];
      if (!target) {
        return;
      }
      onOpenRecord(target.id);
    },
    [onOpenRecord, selectedHistoryIndex, visibleHistoryRecords],
  );

  const handleCopyHistoryTranscript = useCallback(async () => {
    if (!normalizedActiveTranscript) {
      return;
    }
    await navigator.clipboard.writeText(normalizedActiveTranscript);
    setHistoryTranscriptCopied(true);
    window.setTimeout(() => setHistoryTranscriptCopied(false), 1800);
  }, [normalizedActiveTranscript]);

  const handleSaveSpeakerCorrections = useCallback(
    async (speakerNameOverrides: Record<string, string>) => {
      if (!activeHistoryRecord || speakerUpdatePending) {
        return;
      }

      setSpeakerUpdatePending(true);
      setSpeakerUpdateError(null);
      try {
        await onSaveSpeakerCorrections(
          activeHistoryRecord.id,
          speakerNameOverrides,
        );
      } catch (err) {
        setSpeakerUpdateError(
          err instanceof Error
            ? err.message
            : "Failed to save speaker corrections.",
        );
      } finally {
        setSpeakerUpdatePending(false);
      }
    },
    [activeHistoryRecord, onSaveSpeakerCorrections, speakerUpdatePending],
  );

  const handleRerunRecord = useCallback(
    async (request: DiarizationRecordRerunRequest) => {
      if (!activeHistoryRecord || rerunPending) {
        return;
      }

      setRerunPending(true);
      setRerunError(null);
      setSpeakerUpdateError(null);
      setSummaryRefreshError(null);
      setSummaryRefreshPendingId(null);

      try {
        await onRerunRecord(activeHistoryRecord.id, request);
        setRecordWorkspaceTab("transcript");
      } catch (err) {
        setRerunError(
          err instanceof Error ? err.message : "Failed to rerun diarization.",
        );
      } finally {
        setRerunPending(false);
      }
    },
    [activeHistoryRecord, onRerunRecord, rerunPending],
  );

  const handleRegenerateSummary = useCallback(async () => {
    if (!requireSummaryModel()) {
      return;
    }
    if (
      !activeHistoryRecord ||
      summaryRefreshPendingId === activeHistoryRecord.id
    ) {
      return;
    }

    setSummaryRefreshPendingId(activeHistoryRecord.id);
    setSummaryRefreshError(null);

    try {
      await onRegenerateSummary(activeHistoryRecord.id);
    } catch (err) {
      setSummaryRefreshError(
        err instanceof Error
          ? err.message
          : "Failed to regenerate diarization summary.",
      );
    } finally {
      setSummaryRefreshPendingId(null);
    }
  }, [
    activeHistoryRecord,
    onRegenerateSummary,
    requireSummaryModel,
    summaryRefreshPendingId,
  ]);

  const confirmDeleteRecord = useCallback(async () => {
    if (!deleteTargetRecordId || deleteRecordPending) {
      return;
    }

    setDeleteRecordPending(true);
    setDeleteRecordError(null);

    try {
      await onDeleteRecord(deleteTargetRecordId);
      setDeleteTargetRecordId(null);
      setDeleteRecordError(null);
    } catch (err) {
      setDeleteRecordError(
        err instanceof Error
          ? err.message
          : "Failed to delete diarization record.",
      );
    } finally {
      setDeleteRecordPending(false);
    }
  }, [deleteRecordPending, deleteTargetRecordId, onDeleteRecord]);

  const historyDrawer = (
    <RouteHistoryDrawer
      title="Diarization History"
      countLabel={`${visibleHistoryRecords.length} ${visibleHistoryRecords.length === 1 ? "record" : "records"}`}
      triggerCount={visibleHistoryRecords.length}
      open={isHistoryDrawerOpen}
      onOpenChange={handleHistoryDrawerOpenChange}
    >
      {({ close }) => (
        <>
          <div className="app-sidebar-list">
            {historyLoading ? (
              <div className="app-sidebar-loading">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Loading history...
              </div>
            ) : visibleHistoryRecords.length === 0 ? (
              <div className="app-sidebar-empty">
                No saved diarization records yet.
              </div>
            ) : (
              <div className="flex flex-col gap-2.5">
                {visibleHistoryRecords.map((record) => {
                  const isActive = record.id === selectedRecordId;
                  const summaryStatus = normalizeDiarizationSummaryStatus(
                    record.summary_status,
                    record.summary_preview,
                    null,
                  );
                  return (
                    <div
                      key={record.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => {
                        onOpenRecord(record.id);
                        close();
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          onOpenRecord(record.id);
                          close();
                        }
                      }}
                      className={clsx(
                        "app-sidebar-row",
                        isActive
                          ? "app-sidebar-row-active"
                          : "app-sidebar-row-idle",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="app-sidebar-row-label truncate">
                          {record.audio_filename ||
                            record.model_id ||
                            "Diarization run"}
                        </span>
                        <div className="inline-flex items-center gap-1.5 shrink-0">
                          <span className="app-sidebar-row-meta">
                            {formatCreatedAt(record.created_at)}
                          </span>
                          <button
                            onPointerDown={(event) => {
                              event.stopPropagation();
                            }}
                            onClick={(event) => {
                              event.preventDefault();
                              event.stopPropagation();
                              openDeleteRecordConfirm(record.id);
                            }}
                            className="app-sidebar-delete-btn"
                            title="Delete record"
                            aria-label={`Delete ${record.audio_filename || record.model_id || "diarization transcript"}`}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                      <p
                        className="app-sidebar-row-preview"
                        style={{
                          display: "-webkit-box",
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden",
                        }}
                      >
                        {record.transcript_preview}
                      </p>
                      <div className="mt-1 flex items-center justify-between gap-2">
                        <span className="app-sidebar-row-meta">
                          {formatAudioDuration(record.duration_secs)}
                        </span>
                        <StatusBadge
                          tone={diarizationSummaryStatusTone(summaryStatus)}
                          className="px-2 py-0.5 text-[9px] tracking-[0.1em]"
                        >
                          {diarizationSummaryStatusLabel(summaryStatus)}
                        </StatusBadge>
                      </div>
                      {record.summary_preview ? (
                        <p
                          className="mt-1 text-[11px] leading-snug text-[var(--text-muted)]"
                          style={{
                            display: "-webkit-box",
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: "vertical",
                            overflow: "hidden",
                          }}
                        >
                          Summary: {record.summary_preview}
                        </p>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <AnimatePresence>
            {historyError && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="rounded border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-xs text-[var(--danger-text)]"
              >
                {historyError}
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </RouteHistoryDrawer>
  );

  return (
    <>
      {historyActionContainer === undefined
        ? historyDrawer
        : historyActionContainer
          ? createPortal(historyDrawer, historyActionContainer)
          : null}

      <AnimatePresence>
        {selectedRecordId ? (
          <motion.div
            className="fixed inset-0 z-50 bg-black/70 p-3 backdrop-blur-sm sm:p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onCloseRecord}
          >
            <motion.div
              initial={{ y: 18, opacity: 0, scale: 0.985 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 18, opacity: 0, scale: 0.985 }}
              transition={{ duration: 0.18 }}
              onClick={(event) => event.stopPropagation()}
              className="mx-auto flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] shadow-2xl"
            >
              <div className="flex items-center justify-between gap-3 border-b border-[var(--border-muted)] px-4 py-3.5 sm:px-5 sm:py-4">
                <div className="min-w-0 flex-1">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-subtle)]">
                    Diarization Record
                  </p>
                  <div className="mt-1 flex items-center gap-3">
                    <h2 className="truncate text-[1.95rem] font-semibold leading-none tracking-[-0.03em] text-[var(--text-primary)]">
                      {selectedHistorySummary?.audio_filename ||
                        activeHistoryRecord?.audio_filename ||
                        selectedHistorySummary?.model_id ||
                        activeHistoryRecord?.model_id ||
                        "Diarization transcript"}
                    </h2>
                  </div>
                  <div className="mt-2.5 flex flex-wrap items-center gap-2">
                    <span className="text-xs text-[var(--text-muted)]">
                      {selectedHistorySummary
                        ? formatCreatedAt(selectedHistorySummary.created_at)
                        : "No record selected"}
                    </span>
                    {activeHistoryRecord ? (
                      <>
                        <span className="text-[var(--text-subtle)]">•</span>
                        <span className="inline-flex items-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-0.5 text-[10px] font-medium tracking-[0.08em] text-[var(--text-secondary)]">
                          {formatAudioDuration(activeHistoryRecord.duration_secs)}
                        </span>
                        <span className="inline-flex items-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-0.5 text-[10px] font-medium tracking-[0.08em] text-[var(--text-secondary)]">
                          {activeHistoryRecord.corrected_speaker_count ??
                            activeHistoryRecord.speaker_count}{" "}
                          speakers
                        </span>
                        <StatusBadge
                          tone={diarizationSummaryStatusTone(
                            activeHistorySummaryStatus,
                          )}
                        >
                          {diarizationSummaryStatusLabel(activeHistorySummaryStatus)}
                        </StatusBadge>
                      </>
                    ) : null}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2 self-start">
                  {activeHistoryRecord ? (
                    <button
                      onClick={() => openDeleteRecordConfirm(activeHistoryRecord.id)}
                      className="inline-flex h-8 items-center gap-1 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 text-[11px] font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                      title="Delete this record"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      Delete
                    </button>
                  ) : null}
                  <button
                    onClick={() => openAdjacentHistoryRecord("newer")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenNewerHistory}
                    title="Open newer record"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => openAdjacentHistoryRecord("older")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenOlderHistory}
                    title="Open older record"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                  <button
                    onClick={onCloseRecord}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                    title="Close"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="flex flex-1 flex-col overflow-y-auto">
                {selectedRecordLoading ? (
                  <div className="flex min-h-[220px] h-full items-center justify-center gap-2 text-sm text-[var(--text-muted)]">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading record...
                  </div>
                ) : selectedRecordError ? (
                  <div className="p-4 sm:p-5">
                    <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                      {selectedRecordError}
                    </div>
                  </div>
                ) : activeHistoryRecord ? (
                  <div className="p-3 sm:p-4">
                    <Tabs
                      value={recordWorkspaceTab}
                      onValueChange={setRecordWorkspaceTab}
                      className="space-y-3"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <TabsList className="inline-flex w-auto justify-start rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-0.5">
                          <TabsTrigger
                            value="transcript"
                            className="h-8 rounded-full px-3.5 text-[12px] font-medium"
                          >
                            Transcript
                          </TabsTrigger>
                          <TabsTrigger
                            value="speakers"
                            className="h-8 rounded-full px-3.5 text-[12px] font-medium"
                          >
                            Speakers
                          </TabsTrigger>
                          <TabsTrigger
                            value="quality"
                            className="h-8 rounded-full px-3.5 text-[12px] font-medium"
                          >
                            Quality
                          </TabsTrigger>
                        </TabsList>

                        {recordWorkspaceTab === "transcript" ? (
                          <div className="flex items-center gap-2">
                            <Button
                              onClick={() => void handleRegenerateSummary()}
                              variant="outline"
                              size="sm"
                              className="h-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-[12px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)]"
                              disabled={
                                summaryRefreshPendingId === activeHistoryRecord.id
                              }
                            >
                              {summaryRefreshPendingId ===
                              activeHistoryRecord.id ? (
                                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                              ) : (
                                <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
                              )}
                              Regenerate summary
                            </Button>
                            <button
                              onClick={() => void handleCopyHistoryTranscript()}
                              className="inline-flex h-8 items-center gap-1.5 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-[12px] font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)] disabled:opacity-45"
                              disabled={!normalizedActiveTranscript}
                            >
                              {historyTranscriptCopied ? (
                                <>
                                  <Check className="h-3.5 w-3.5" />
                                  Copied
                                </>
                              ) : (
                                <>
                                  <Copy className="h-3.5 w-3.5" />
                                  Copy
                                </>
                              )}
                            </button>
                            <DiarizationExportDialog record={activeHistoryRecord}>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="h-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-[12px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)]"
                                disabled={!normalizedActiveTranscript}
                              >
                                <Download className="mr-1.5 h-3.5 w-3.5" />
                                Export
                              </Button>
                            </DiarizationExportDialog>
                          </div>
                        ) : null}
                      </div>

                      {summaryRefreshError ? (
                        <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                          {summaryRefreshError}
                        </div>
                      ) : null}

                      <TabsContent value="transcript" className="mt-0 space-y-3">
                        <DiarizationReviewWorkspace
                          record={activeHistoryRecord}
                          audioUrl={selectedHistoryAudioUrl}
                          loading={selectedRecordLoading}
                          autoScrollActiveEntry={true}
                          summaryModelGuidance={summaryModelGuidance}
                          emptyMessage={
                            normalizedActiveTranscript ||
                            "No transcript text available for this record."
                          }
                        />
                      </TabsContent>

                      <TabsContent value="speakers" className="mt-0">
                        <DiarizationSpeakerManager
                          record={activeHistoryRecord}
                          isSaving={speakerUpdatePending}
                          error={speakerUpdateError}
                          onSave={handleSaveSpeakerCorrections}
                        />
                      </TabsContent>

                      <TabsContent value="quality" className="mt-0">
                        <DiarizationQualityPanel
                          record={activeHistoryRecord}
                          isRerunning={rerunPending}
                          error={rerunError}
                          onRerun={handleRerunRecord}
                        />
                      </TabsContent>
                    </Tabs>
                  </div>
                ) : (
                  <div className="flex min-h-[220px] h-full items-center justify-center text-center text-sm text-[var(--text-subtle)]">
                    Select a history record to inspect playback and transcript.
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <Dialog
        open={!!deleteTargetRecord}
        onOpenChange={(open) => {
          if (!open) {
            closeDeleteRecordConfirm();
          }
        }}
      >
        {deleteTargetRecord ? (
          <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
            <DialogTitle className="sr-only">
              Delete diarization record?
            </DialogTitle>
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Delete diarization record?
                </h3>
                <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                  This permanently removes the saved audio and transcript from
                  history.
                </DialogDescription>
                <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                  {deleteTargetRecord.audio_filename ||
                    deleteTargetRecord.model_id ||
                    deleteTargetRecord.id}
                </p>
              </div>
            </div>

            <AnimatePresence>
              {deleteRecordError ? (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]"
                >
                  {deleteRecordError}
                </motion.div>
              ) : null}
            </AnimatePresence>

            <div className="mt-5 flex items-center justify-end gap-2">
              <Button
                onClick={closeDeleteRecordConfirm}
                variant="outline"
                size="sm"
                className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
                disabled={deleteRecordPending}
              >
                Cancel
              </Button>
              <Button
                onClick={() => void confirmDeleteRecord()}
                variant="destructive"
                size="sm"
                className="h-8 gap-1.5"
                disabled={deleteRecordPending}
              >
                {deleteRecordPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Delete record
              </Button>
            </div>
          </DialogContent>
        ) : null}
      </Dialog>
    </>
  );
}
