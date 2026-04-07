import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Copy,
  Download,
  ExternalLink,
  Loader2,
  MoreVertical,
  Trash2,
} from "lucide-react";

import { api, type DiarizationRecordSummary } from "@/api";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { DiarizationExportDialog } from "@/components/DiarizationExportDialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  diarizationSummaryStatusLabel,
  normalizeDiarizationSummaryStatus,
} from "@/utils/diarizationSummary";
import { formattedTranscriptFromRecord } from "@/utils/diarizationTranscript";

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

interface DiarizationHistoryTableProps {
  records: DiarizationRecordSummary[];
  loading?: boolean;
  error?: string | null;
  loadMore?: {
    canLoadMore: boolean;
    loading: boolean;
    onLoadMore: () => void;
  };
  onOpenRecord: (recordId: string) => void;
  onDeleteRecord?: (recordId: string) => Promise<void>;
  onRefresh?: () => void;
}

function rowLabel(record: DiarizationRecordSummary): string {
  return record.audio_filename || record.model_id || record.id;
}

export function DiarizationHistoryTable({
  records,
  loading = false,
  error = null,
  loadMore,
  onOpenRecord,
  onDeleteRecord,
  onRefresh,
}: DiarizationHistoryTableProps) {
  const { notify } = useNotifications();
  const [busyRecordId, setBusyRecordId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DiarizationRecordSummary | null>(
    null,
  );
  const [deletePending, setDeletePending] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportRecord, setExportRecord] = useState<Awaited<
    ReturnType<typeof api.getDiarizationRecord>
  > | null>(null);

  const deleteTargetLabel = useMemo(
    () => (deleteTarget ? rowLabel(deleteTarget) : "Diarization record"),
    [deleteTarget],
  );

  async function loadRecordForAction(recordId: string) {
    return api.getDiarizationRecord(recordId);
  }

  async function handleCopy(record: DiarizationRecordSummary): Promise<void> {
    if (busyRecordId || deletePending) {
      return;
    }

    setBusyRecordId(record.id);
    try {
      const fullRecord = await loadRecordForAction(record.id);
      const transcript = formattedTranscriptFromRecord(fullRecord);
      if (!transcript.trim()) {
        notify({
          title: "Nothing to copy",
          description: "This diarization does not have transcript text yet.",
          tone: "warning",
        });
        return;
      }

      await navigator.clipboard.writeText(transcript);
      notify({
        title: "Transcript copied",
        description: rowLabel(record),
        tone: "success",
      });
    } catch (err) {
      notify({
        title: "Could not copy transcript",
        description:
          err instanceof Error ? err.message : "Failed to load diarization record.",
        tone: "warning",
      });
    } finally {
      setBusyRecordId(null);
    }
  }

  async function handleExport(record: DiarizationRecordSummary): Promise<void> {
    if (busyRecordId || deletePending) {
      return;
    }

    setBusyRecordId(record.id);
    try {
      const fullRecord = await loadRecordForAction(record.id);
      setExportRecord(fullRecord);
      setExportDialogOpen(true);
    } catch (err) {
      notify({
        title: "Could not open export",
        description:
          err instanceof Error ? err.message : "Failed to load diarization record.",
        tone: "warning",
      });
    } finally {
      setBusyRecordId(null);
    }
  }

  async function handleConfirmDelete(): Promise<void> {
    if (!deleteTarget || !onDeleteRecord || deletePending) {
      return;
    }

    setDeletePending(true);
    setDeleteError(null);
    try {
      await onDeleteRecord(deleteTarget.id);
      notify({
        title: "Diarization deleted",
        description: deleteTargetLabel,
        tone: "success",
      });
      setDeleteTarget(null);
    } catch (err) {
      setDeleteError(
        err instanceof Error ? err.message : "Failed to delete diarization.",
      );
    } finally {
      setDeletePending(false);
    }
  }

  const activeBusyRecordId = busyRecordId;

  if (loading) {
    return (
      <div className="mb-6 flex min-h-[20rem] items-center justify-center rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm text-[var(--text-muted)]">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading diarization history...
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-6 rounded-2xl border border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{error}</p>
          </div>
          {onRefresh ? (
            <Button type="button" variant="outline" size="sm" onClick={onRefresh}>
              Retry
            </Button>
          ) : null}
        </div>
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="mb-6 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-10 text-center">
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          No diarization records yet
        </h3>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Saved speaker-separated transcripts will appear here.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="mb-6 overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-sm">
            <thead className="bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)]">
              <tr>
                <th className="px-4 py-3 font-semibold sm:px-5">Created</th>
                <th className="px-4 py-3 font-semibold">File</th>
                <th className="px-4 py-3 font-semibold">Speakers</th>
                <th className="px-4 py-3 font-semibold">Duration</th>
                <th className="px-4 py-3 font-semibold">Summary</th>
                <th className="w-[56px] px-3 py-3 text-right font-semibold sm:px-4">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {records.map((record) => {
                const summaryStatus = normalizeDiarizationSummaryStatus(
                  record.summary_status,
                  record.summary_preview,
                  null,
                );
                const summaryLabel =
                  summaryStatus === "not_requested"
                    ? "No summary yet"
                    : diarizationSummaryStatusLabel(summaryStatus);
                const isBusy = activeBusyRecordId === record.id;

                return (
                  <tr
                    key={record.id}
                    aria-label={`Open diarization ${rowLabel(record)}`}
                    className="cursor-pointer border-t border-[var(--border-muted)] transition-colors hover:bg-[var(--bg-surface-1)]"
                    onClick={(event) => {
                      if ((event.target as HTMLElement).closest("[data-row-action]")) {
                        return;
                      }
                      onOpenRecord(record.id);
                    }}
                    onKeyDown={(event) => {
                      if ((event.target as HTMLElement).closest("[data-row-action]")) {
                        return;
                      }
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onOpenRecord(record.id);
                      }
                    }}
                    tabIndex={0}
                  >
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)] sm:px-5">
                      {formatCreatedAt(record.created_at)}
                    </td>
                    <td className="px-4 py-3 align-top">
                      <div className="font-medium text-[var(--text-primary)]">
                        {record.audio_filename || "Audio input"}
                      </div>
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      {record.corrected_speaker_count ?? record.speaker_count}
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      {formatAudioDuration(record.duration_secs)}
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      <div className="max-w-[34rem]">
                        {record.summary_preview ? (
                          <div className="line-clamp-3 text-[var(--text-primary)]">
                            {record.summary_preview}
                          </div>
                        ) : (
                          <div className="text-xs font-medium uppercase tracking-[0.12em] text-[var(--text-muted)]">
                            {summaryLabel}
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-3 py-2 align-top text-right sm:px-4">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            data-row-action
                            className="h-8 w-8 rounded-full text-[var(--text-muted)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                            aria-label={`More actions for ${rowLabel(record)}`}
                            onClick={(event) => event.stopPropagation()}
                            onKeyDown={(event) => event.stopPropagation()}
                          >
                            {isBusy ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <MoreVertical className="h-4 w-4" />
                            )}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="w-48"
                          onClick={(event) => event.stopPropagation()}
                        >
                          <DropdownMenuItem onSelect={() => onOpenRecord(record.id)}>
                            <ExternalLink className="mr-2 h-4 w-4" />
                            Open record
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            disabled={isBusy || deletePending}
                            onSelect={() => void handleCopy(record)}
                          >
                            <Copy className="mr-2 h-4 w-4" />
                            Copy transcript
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            disabled={isBusy || deletePending}
                            onSelect={() => void handleExport(record)}
                          >
                            <Download className="mr-2 h-4 w-4" />
                            Export
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            disabled={!onDeleteRecord || deletePending}
                            onSelect={() => {
                              setDeleteError(null);
                              setDeleteTarget(record);
                            }}
                            className="text-[var(--danger-text)] focus:text-[var(--danger-text)]"
                          >
                            <Trash2 className="mr-2 h-4 w-4" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {loadMore?.canLoadMore ? (
          <div className="flex justify-center border-t border-[var(--border-muted)] px-4 py-3 sm:px-5">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              onClick={loadMore.onLoadMore}
              disabled={loadMore.loading}
            >
              {loadMore.loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Load more
            </Button>
          </div>
        ) : null}
      </div>

      <DiarizationExportDialog
        record={exportRecord}
        open={exportDialogOpen}
        onOpenChange={(open) => {
          setExportDialogOpen(open);
          if (!open) {
            setExportRecord(null);
          }
        }}
      />

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!deletePending) {
            setDeleteTarget(open ? deleteTarget : null);
            if (!open) {
              setDeleteError(null);
            }
          }
        }}
      >
        <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
          <DialogTitle className="sr-only">Delete diarization?</DialogTitle>
          {deleteTarget ? (
            <>
              <div className="flex items-start gap-3">
                <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                  <AlertTriangle className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                    Delete diarization?
                  </h3>
                  <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                    This permanently removes the saved audio and speaker-separated
                    transcript from history.
                  </DialogDescription>
                  <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                    {deleteTargetLabel}
                  </p>
                </div>
              </div>

              {deleteError ? (
                <div className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                  {deleteError}
                </div>
              ) : null}

              <div className="mt-5 flex items-center justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
                  onClick={() => {
                    setDeleteTarget(null);
                    setDeleteError(null);
                  }}
                  disabled={deletePending}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  className="h-8 gap-1.5"
                  onClick={() => void handleConfirmDelete()}
                  disabled={deletePending}
                >
                  {deletePending ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Deleting
                    </>
                  ) : (
                    "Delete diarization"
                  )}
                </Button>
              </div>
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  );
}
