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
import { StatusBadge } from "@/components/ui/status-badge";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import { TranscriptionReviewWorkspace } from "@/features/transcription/components/TranscriptionReviewWorkspace";
import {
  formatAudioDuration,
  formatCreatedAt,
  normalizeProcessingStatus,
  normalizeSummaryStatus,
  processingStatusLabel,
  processingStatusTone,
  summaryStatusLabel,
  summaryStatusTone,
} from "@/features/transcription/playground/support";
import { formatTranscriptionText } from "@/features/transcription/transcript";

interface TranscriptionRecordDetailProps {
  record: TranscriptionRecord | null;
  audioUrl: string | null;
  loading?: boolean;
  error?: string | null;
  summaryModelGuidance?: string | null;
  onBack?: () => void;
  onDelete?: () => void;
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
  summaryModelGuidance = null,
  onBack,
  onDelete,
  onRegenerateSummary,
  deletePending = false,
  summaryRefreshPending = false,
  summaryRefreshError = null,
}: TranscriptionRecordDetailProps) {
  const [copied, setCopied] = useState(false);

  const processingStatus = useMemo(
    () =>
      normalizeProcessingStatus(
        record?.processing_status,
        record?.processing_error,
      ),
    [record?.processing_error, record?.processing_status],
  );
  const summaryStatus = useMemo(
    () =>
      normalizeSummaryStatus(
        record?.summary_status,
        record?.summary_text,
        record?.summary_error,
      ),
    [record?.summary_error, record?.summary_status, record?.summary_text],
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

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          {onBack ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="mb-2 h-8 gap-1.5 px-2 text-xs text-[var(--text-muted)]"
              onClick={onBack}
            >
              <ArrowLeft className="h-3.5 w-3.5" />
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
          <StatusBadge tone={processingStatusTone(processingStatus)}>
            {processingStatusLabel(processingStatus)}
          </StatusBadge>
          <StatusBadge tone={summaryStatusTone(summaryStatus)}>
            {summaryStatusLabel(summaryStatus)}
          </StatusBadge>
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
              onClick={onDelete}
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
        stickyPlaybackFooter={true}
        summaryModelGuidance={summaryModelGuidance}
        emptyTitle="Transcription in progress"
        emptyMessage="The transcript will appear here as soon as this record is ready."
      />
    </div>
  );
}
