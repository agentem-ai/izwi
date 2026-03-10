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
  History,
  Loader2,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  api,
  type TranscriptionRecord,
  type TranscriptionRecordSummary,
} from "@/api";
import {
  formatAudioDuration,
  formatCreatedAt,
  summarizeRecord,
} from "@/features/transcription/playground/support";
import { TranscriptionCorrectionsPanel } from "@/features/transcription/components/TranscriptionCorrectionsPanel";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import { TranscriptionQualityPanel } from "@/features/transcription/components/TranscriptionQualityPanel";
import { TranscriptionReviewWorkspace } from "@/features/transcription/components/TranscriptionReviewWorkspace";
import { formattedTranscriptFromRecord } from "@/features/transcription/utils/transcriptionTranscript";

interface TranscriptionHistoryPanelProps {
  latestRecord?: TranscriptionRecord | null;
  historyActionContainer?: HTMLElement | null;
}

export function TranscriptionHistoryPanel({
  latestRecord = null,
  historyActionContainer,
}: TranscriptionHistoryPanelProps) {
  const [historyRecords, setHistoryRecords] = useState<
    TranscriptionRecordSummary[]
  >([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedHistoryRecordId, setSelectedHistoryRecordId] = useState<
    string | null
  >(null);
  const [selectedHistoryRecord, setSelectedHistoryRecord] =
    useState<TranscriptionRecord | null>(null);
  const [selectedHistoryLoading, setSelectedHistoryLoading] = useState(false);
  const [selectedHistoryError, setSelectedHistoryError] = useState<
    string | null
  >(null);
  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState(false);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [historyTranscriptCopied, setHistoryTranscriptCopied] = useState(false);
  const [recordWorkspaceTab, setRecordWorkspaceTab] = useState("transcript");
  const [updatePending, setUpdatePending] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);
  const [deleteTargetRecordId, setDeleteTargetRecordId] = useState<
    string | null
  >(null);
  const [deleteRecordPending, setDeleteRecordPending] = useState(false);
  const [deleteRecordError, setDeleteRecordError] = useState<string | null>(
    null,
  );

  const mergeHistorySummary = useCallback(
    (summary: TranscriptionRecordSummary) => {
      setHistoryRecords((previous) => {
        const next = [
          summary,
          ...previous.filter((item) => item.id !== summary.id),
        ];
        next.sort((a, b) => b.created_at - a.created_at);
        return next;
      });
    },
    [],
  );

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const records = await api.listTranscriptionRecords();
      setHistoryRecords(records);
      setSelectedHistoryRecordId((current) => {
        if (current && records.some((item) => item.id === current)) {
          return current;
        }
        return records[0]?.id ?? null;
      });
    } catch (err) {
      setHistoryError(
        err instanceof Error
          ? err.message
          : "Failed to load transcription history.",
      );
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!selectedHistoryRecordId) {
      setSelectedHistoryRecord(null);
      setSelectedHistoryError(null);
      return;
    }

    if (selectedHistoryRecord?.id === selectedHistoryRecordId) {
      return;
    }

    let cancelled = false;
    setSelectedHistoryLoading(true);
    setSelectedHistoryError(null);

    api
      .getTranscriptionRecord(selectedHistoryRecordId)
      .then((record) => {
        if (cancelled) {
          return;
        }
        setSelectedHistoryRecord(record);
        mergeHistorySummary(summarizeRecord(record));
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSelectedHistoryError(
          err instanceof Error
            ? err.message
            : "Failed to load transcription record details.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setSelectedHistoryLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [mergeHistorySummary, selectedHistoryRecord, selectedHistoryRecordId]);

  useEffect(() => {
    if (!latestRecord) {
      return;
    }
    setSelectedHistoryRecord(latestRecord);
    setSelectedHistoryRecordId(latestRecord.id);
    setSelectedHistoryError(null);
    mergeHistorySummary(summarizeRecord(latestRecord));
  }, [latestRecord?.id, latestRecord, mergeHistorySummary]);

  const closeHistoryModal = useCallback(() => {
    setIsHistoryModalOpen(false);
  }, []);

  const openHistoryRecord = useCallback((recordId: string) => {
    setSelectedHistoryRecordId(recordId);
    setSelectedHistoryError(null);
    setIsHistoryModalOpen(true);
  }, []);

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

  const handleHistoryDrawerOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen && deleteTargetRecordId) {
        return;
      }
      setIsHistoryDrawerOpen(nextOpen);
    },
    [deleteTargetRecordId],
  );

  const confirmDeleteRecord = useCallback(async () => {
    if (!deleteTargetRecordId || deleteRecordPending) {
      return;
    }

    setDeleteRecordPending(true);
    setDeleteRecordError(null);

    try {
      await api.deleteTranscriptionRecord(deleteTargetRecordId);

      const previous = historyRecords;
      const deletedIndex = previous.findIndex(
        (record) => record.id === deleteTargetRecordId,
      );
      const remaining = previous.filter(
        (record) => record.id !== deleteTargetRecordId,
      );

      setHistoryRecords(remaining);

      if (selectedHistoryRecordId === deleteTargetRecordId) {
        const fallbackIndex =
          deletedIndex >= 0 ? Math.min(deletedIndex, remaining.length - 1) : 0;
        const fallbackId = remaining[fallbackIndex]?.id ?? null;
        setSelectedHistoryRecordId(fallbackId);
        if (!fallbackId) {
          setSelectedHistoryRecord(null);
          setIsHistoryModalOpen(false);
        }
      }

      if (selectedHistoryRecord?.id === deleteTargetRecordId) {
        setSelectedHistoryRecord(null);
      }

      setDeleteTargetRecordId(null);
      setDeleteRecordError(null);
    } catch (err) {
      setDeleteRecordError(
        err instanceof Error
          ? err.message
          : "Failed to delete transcription record.",
      );
    } finally {
      setDeleteRecordPending(false);
    }
  }, [
    deleteRecordPending,
    deleteTargetRecordId,
    historyRecords,
    selectedHistoryRecord,
    selectedHistoryRecordId,
  ]);

  useEffect(() => {
    if (!isHistoryModalOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeHistoryModal();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeHistoryModal, isHistoryModalOpen]);

  useEffect(() => {
    if (!isHistoryModalOpen) {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isHistoryModalOpen]);

  const selectedHistorySummary = useMemo(
    () =>
      selectedHistoryRecordId
        ? (historyRecords.find(
            (record) => record.id === selectedHistoryRecordId,
          ) ?? null)
        : null,
    [historyRecords, selectedHistoryRecordId],
  );
  const activeHistoryRecord =
    selectedHistoryRecord &&
    selectedHistoryRecord.id === selectedHistoryRecordId
      ? selectedHistoryRecord
      : null;
  const deleteTargetRecord = useMemo(() => {
    if (!deleteTargetRecordId) {
      return null;
    }
    const fromSummary = historyRecords.find(
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
  }, [activeHistoryRecord, deleteTargetRecordId, historyRecords]);
  const selectedHistoryAudioUrl = useMemo(
    () =>
      selectedHistoryRecordId
        ? api.transcriptionRecordAudioUrl(selectedHistoryRecordId)
        : null,
    [selectedHistoryRecordId],
  );
  const selectedHistoryIndex = useMemo(
    () =>
      selectedHistoryRecordId
        ? historyRecords.findIndex(
            (record) => record.id === selectedHistoryRecordId,
          )
        : -1,
    [historyRecords, selectedHistoryRecordId],
  );
  const canOpenNewerHistory = selectedHistoryIndex > 0;
  const canOpenOlderHistory =
    selectedHistoryIndex >= 0 &&
    selectedHistoryIndex < historyRecords.length - 1;

  const openAdjacentHistoryRecord = useCallback(
    (direction: "newer" | "older") => {
      if (selectedHistoryIndex < 0) {
        return;
      }
      const targetIndex =
        direction === "newer"
          ? selectedHistoryIndex - 1
          : selectedHistoryIndex + 1;
      if (targetIndex < 0 || targetIndex >= historyRecords.length) {
        return;
      }
      const target = historyRecords[targetIndex];
      if (!target) {
        return;
      }
      setSelectedHistoryRecordId(target.id);
      setSelectedHistoryError(null);
      setIsHistoryModalOpen(true);
    },
    [historyRecords, selectedHistoryIndex],
  );

  const normalizedTranscript = useMemo(
    () => formattedTranscriptFromRecord(activeHistoryRecord),
    [activeHistoryRecord],
  );

  const handleCopyHistoryTranscript = useCallback(async () => {
    if (!normalizedTranscript) {
      return;
    }
    await navigator.clipboard.writeText(normalizedTranscript);
    setHistoryTranscriptCopied(true);
    window.setTimeout(() => setHistoryTranscriptCopied(false), 1800);
  }, [normalizedTranscript]);

  const handleSaveCorrections = useCallback(
    async (segments: TranscriptionRecord["segments"]) => {
      if (!activeHistoryRecord || updatePending) {
        return;
      }

      setUpdatePending(true);
      setUpdateError(null);
      try {
        const transcription = segments
          .map((segment) => segment.text.trim())
          .filter(Boolean)
          .join("\n\n");
        const updatedRecord = await api.updateTranscriptionRecord(
          activeHistoryRecord.id,
          {
            transcription,
            segments,
          },
        );
        setSelectedHistoryRecord(updatedRecord);
        mergeHistorySummary(summarizeRecord(updatedRecord));
        setRecordWorkspaceTab("transcript");
      } catch (err) {
        setUpdateError(
          err instanceof Error
            ? err.message
            : "Failed to save transcription corrections.",
        );
      } finally {
        setUpdatePending(false);
      }
    },
    [activeHistoryRecord, mergeHistorySummary, updatePending],
  );

  useEffect(() => {
    setHistoryTranscriptCopied(false);
    setRecordWorkspaceTab("transcript");
    setUpdateError(null);
  }, [selectedHistoryRecordId]);

  const historyDrawer = (
    <RouteHistoryDrawer
      title="Transcriptions"
      countLabel={`${historyRecords.length} ${historyRecords.length === 1 ? "record" : "records"}`}
      triggerCount={historyRecords.length}
      open={isHistoryDrawerOpen}
      onOpenChange={handleHistoryDrawerOpenChange}
    >
      {({ close }) => (
        <>
          <div className="app-sidebar-list scrollbar-thin">
            {historyLoading ? (
              <div className="app-sidebar-loading">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Loading history...
              </div>
            ) : historyRecords.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-6 text-center opacity-60">
                <History className="mb-3 h-10 w-10 text-muted-foreground" />
                <p className="text-sm font-medium text-muted-foreground">
                  No history yet
                </p>
                <p className="mt-1 text-xs text-muted-foreground/70">
                  Transcriptions will appear here
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-2.5">
                {historyRecords.map((record) => {
                  const isActive = record.id === selectedHistoryRecordId;
                  return (
                    <div
                      key={record.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => {
                        openHistoryRecord(record.id);
                        close();
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          openHistoryRecord(record.id);
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
                            "Transcription run"}
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
                            aria-label={`Delete ${record.audio_filename || record.model_id || "transcript"}`}
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
                        {record.transcription_preview}
                      </p>
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
        {isHistoryModalOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-black/70 p-3 backdrop-blur-sm sm:p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeHistoryModal}
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
                    Transcription Record
                  </p>
                  <div className="mt-1 flex items-center gap-3">
                    <h2 className="truncate text-[1.95rem] font-semibold leading-none tracking-[-0.03em] text-[var(--text-primary)]">
                      {selectedHistorySummary?.audio_filename ||
                        selectedHistorySummary?.model_id ||
                        "Transcription transcript"}
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
                          {activeHistoryRecord.language || "Unknown language"}
                        </span>
                        <span className="inline-flex max-w-[220px] items-center truncate rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-0.5 text-[10px] font-medium tracking-[0.02em] text-[var(--text-secondary)]">
                          {activeHistoryRecord.model_id || "Unknown model"}
                        </span>
                      </>
                    ) : null}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2 self-start">
                  {activeHistoryRecord && (
                    <button
                      onClick={() =>
                        openDeleteRecordConfirm(activeHistoryRecord.id)
                      }
                      className="inline-flex h-8 items-center gap-1 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 text-[11px] font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                      title="Delete this record"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </button>
                  )}
                  <button
                    onClick={() => openAdjacentHistoryRecord("newer")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenNewerHistory}
                    title="Open newer record"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => openAdjacentHistoryRecord("older")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenOlderHistory}
                    title="Open older record"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                  <button
                    onClick={closeHistoryModal}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                    title="Close"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="flex flex-1 flex-col overflow-y-auto">
                {selectedHistoryLoading ? (
                  <div className="flex min-h-[220px] h-full items-center justify-center gap-2 text-sm text-[var(--text-muted)]">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading record...
                  </div>
                ) : selectedHistoryError ? (
                  <div className="p-4 sm:p-5">
                    <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                      {selectedHistoryError}
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
                            value="corrections"
                            className="h-8 rounded-full px-3.5 text-[12px] font-medium"
                          >
                            Corrections
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
                            <button
                              onClick={() => void handleCopyHistoryTranscript()}
                              className="inline-flex h-8 items-center gap-1.5 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-[12px] font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)] disabled:opacity-45"
                              disabled={!normalizedTranscript}
                            >
                              {historyTranscriptCopied ? (
                                <>
                                  <Check className="w-3.5 h-3.5" />
                                  Copied
                                </>
                              ) : (
                                <>
                                  <Copy className="w-3.5 h-3.5" />
                                  Copy
                                </>
                              )}
                            </button>
                            <TranscriptionExportDialog
                              record={activeHistoryRecord}
                            >
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="h-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-[12px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)]"
                              >
                                <Download className="mr-1.5 w-3.5 h-3.5" />
                                Export
                              </Button>
                            </TranscriptionExportDialog>
                          </div>
                        ) : null}
                      </div>

                      <TabsContent value="transcript" className="mt-0 space-y-3">
                        <TranscriptionReviewWorkspace
                          record={activeHistoryRecord}
                          audioUrl={selectedHistoryAudioUrl}
                          loading={selectedHistoryLoading}
                          emptyMessage={
                            normalizedTranscript ||
                            "No transcript text available for this record."
                          }
                        />
                      </TabsContent>

                      <TabsContent value="corrections" className="mt-0">
                        <TranscriptionCorrectionsPanel
                          record={activeHistoryRecord}
                          isSaving={updatePending}
                          error={updateError}
                          onSave={handleSaveCorrections}
                        />
                      </TabsContent>

                      <TabsContent value="quality" className="mt-0">
                        <TranscriptionQualityPanel
                          record={activeHistoryRecord}
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
        )}
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
            <DialogTitle className="sr-only">Delete transcription record?</DialogTitle>
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Delete transcription record?
                </h3>
                <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                  This permanently removes the saved audio and transcript from history.
                </DialogDescription>
                <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                  {deleteTargetRecord.audio_filename ||
                    deleteTargetRecord.model_id ||
                    deleteTargetRecord.id}
                </p>
              </div>
            </div>

            <AnimatePresence>
              {deleteRecordError && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]"
                >
                  {deleteRecordError}
                </motion.div>
              )}
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
