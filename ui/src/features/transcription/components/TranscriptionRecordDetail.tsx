import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  Copy,
  Download,
  Loader2,
  RotateCcw,
  Trash2,
} from "lucide-react";

import { type TranscriptionRecord } from "@/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import { TranscriptionReviewWorkspace } from "@/features/transcription/components/TranscriptionReviewWorkspace";
import {
  formatAudioDuration,
  formatCreatedAt,
  normalizeProcessingStatus,
} from "@/features/transcription/playground/support";
import { formatTranscriptionText } from "@/features/transcription/transcript";

interface TranscriptionRecordDetailProps {
  record: TranscriptionRecord | null;
  audioUrl: string | null;
  loading?: boolean;
  error?: string | null;
  deleteError?: string | null;
  summaryModelGuidance?: string | null;
  onBack?: () => void;
  onDelete?: () => Promise<void> | void;
  onRegenerateSummary?: () => void;
  deletePending?: boolean;
  summaryRefreshPending?: boolean;
  summaryRefreshError?: string | null;
}

export function TranscriptionRecordDetail({
  record,
  audioUrl,
  loading = false,
  error = null,
  deleteError = null,
  summaryModelGuidance = null,
  onBack,
  onDelete,
  onRegenerateSummary,
  deletePending = false,
  summaryRefreshPending = false,
  summaryRefreshError = null,
}: TranscriptionRecordDetailProps) {
  const [copied, setCopied] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);

  const processingStatus = useMemo(
    () =>
      normalizeProcessingStatus(
        record?.processing_status,
        record?.processing_error,
      ),
    [record?.processing_error, record?.processing_status],
  );
  const exportText = useMemo(() => formatTranscriptionText(record), [record]);
  const hasTranscript = useMemo(
    () => (record?.transcription ?? "").trim().length > 0,
    [record?.transcription],
  );
  const statusMessage = useMemo(() => {
    switch (processingStatus) {
      case "pending":
        return "This transcription is queued and will begin processing shortly.";
      case "processing":
        return "This transcription is currently being processed. Results will appear here automatically.";
      case "failed":
        return record?.processing_error || "Transcription processing failed.";
      case "ready":
      default:
        return null;
    }
  }, [processingStatus, record?.processing_error]);

  async function handleCopy(): Promise<void> {
    if (!exportText) {
      return;
    }
    await navigator.clipboard.writeText(exportText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  async function handleConfirmDelete(): Promise<void> {
    if (!onDelete || deletePending) {
      return;
    }

    await onDelete();
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          {onBack ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mb-4 h-10 gap-2 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 text-sm font-medium text-[var(--text-secondary)] shadow-sm hover:bg-[var(--bg-surface-1)]"
              onClick={onBack}
            >
              <ArrowLeft className="h-4 w-4" />
              Back to transcriptions
            </Button>
          ) : null}
          <h2 className="truncate text-2xl font-semibold tracking-tight text-[var(--text-primary)]">
            {record?.audio_filename || record?.model_id || "Transcription record"}
          </h2>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-[var(--text-muted)]">
            {record ? <span>{formatCreatedAt(record.created_at)}</span> : null}
            {record?.duration_secs != null ? (
              <span>{formatAudioDuration(record.duration_secs)}</span>
            ) : null}
            {record?.language ? <span>{record.language}</span> : null}
            {record?.model_id ? <span>{record.model_id}</span> : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          {onRegenerateSummary ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              onClick={onRegenerateSummary}
              disabled={
                !record ||
                processingStatus !== "ready" ||
                summaryRefreshPending ||
                !hasTranscript
              }
            >
              {summaryRefreshPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              Regenerate summary
            </Button>
          ) : null}
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-9 gap-2"
            onClick={() => void handleCopy()}
            disabled={!hasTranscript}
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            Copy
          </Button>
          <TranscriptionExportDialog record={record}>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              disabled={!hasTranscript}
            >
              <Download className="h-4 w-4" />
              Export
            </Button>
          </TranscriptionExportDialog>
          {onDelete ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2 border-[var(--danger-border)] text-[var(--danger-text)] hover:bg-[var(--danger-bg)]"
              onClick={() => setDeleteConfirmOpen(true)}
              disabled={deletePending}
            >
              {deletePending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
              Delete
            </Button>
          ) : null}
        </div>
      </div>

      {error ? (
        <Card className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
          {error}
        </Card>
      ) : null}

      {statusMessage ? (
        <Card
          className={
            processingStatus === "failed"
              ? "border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
              : "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]"
          }
        >
          <div className="flex items-start gap-3">
            {processingStatus === "failed" ? (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            ) : (
              <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />
            )}
            <p>{statusMessage}</p>
          </div>
        </Card>
      ) : null}

      {summaryRefreshError ? (
        <Card className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
          {summaryRefreshError}
        </Card>
      ) : null}

      <TranscriptionReviewWorkspace
        record={record}
        audioUrl={audioUrl}
        loading={loading}
        autoScrollActiveEntry={true}
        fixedPlaybackFooter={true}
        summaryModelGuidance={summaryModelGuidance}
        emptyTitle="Transcription in progress"
        emptyMessage="The transcript will appear here as soon as this record is ready."
      />

      <Dialog
        open={deleteConfirmOpen}
        onOpenChange={(open) => {
          if (!deletePending) {
            setDeleteConfirmOpen(open);
          }
        }}
      >
        <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-6">
          <DialogTitle className="sr-only">Delete transcription?</DialogTitle>
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--danger-bg)] text-[var(--danger-text)]">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-base font-semibold text-[var(--text-primary)]">
                Delete transcription?
              </h3>
              <DialogDescription className="mt-1.5 text-sm leading-relaxed text-[var(--text-muted)]">
                This permanently removes the saved audio and transcript from
                history. This action cannot be undone.
              </DialogDescription>
              <div className="mt-4 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                <p className="truncate text-xs font-medium text-[var(--text-secondary)]">
                  {record?.audio_filename || record?.model_id || record?.id || "Transcription record"}
                </p>
              </div>
            </div>
          </div>

          {deleteError ? (
            <div className="mt-4 flex items-start gap-2 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger-text)]">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <p>{deleteError}</p>
            </div>
          ) : null}

          <div className="mt-6 flex items-center justify-end gap-3 border-t border-[var(--border-muted)] pt-5">
            <Button
              type="button"
              variant="outline"
              onClick={() => setDeleteConfirmOpen(false)}
              disabled={deletePending}
            >
              Cancel
            </Button>
            <Button
              type="button"
              className="bg-[var(--danger-text)] text-white hover:bg-[var(--danger-text)]/90"
              onClick={() => void handleConfirmDelete()}
              disabled={deletePending}
            >
              {deletePending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete transcription"
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
