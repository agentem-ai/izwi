import { AlertTriangle, Loader2 } from "lucide-react";

import { type SpeechHistoryRecordSummary } from "@/api";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  formatSpeechCreatedAt,
  formatSpeechDuration,
  normalizeSpeechProcessingStatus,
  speechProcessingStatusLabel,
} from "@/features/text-to-speech/support";

interface TextToSpeechHistoryTableProps {
  records: SpeechHistoryRecordSummary[];
  loading?: boolean;
  error?: string | null;
  onOpenRecord: (recordId: string) => void;
  onRefresh?: () => void;
}

function statusToneFor(
  status: ReturnType<typeof normalizeSpeechProcessingStatus>,
): "neutral" | "warning" | "success" | "danger" {
  switch (status) {
    case "pending":
      return "neutral";
    case "processing":
      return "warning";
    case "failed":
      return "danger";
    case "ready":
    default:
      return "success";
  }
}

export function TextToSpeechHistoryTable({
  records,
  loading = false,
  error = null,
  onOpenRecord,
  onRefresh,
}: TextToSpeechHistoryTableProps) {
  if (loading) {
    return (
      <div className="mb-6 flex min-h-[20rem] items-center justify-center rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm text-[var(--text-muted)]">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading text-to-speech history...
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
          No text-to-speech jobs yet
        </h3>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Queued, processing, and completed generations will appear here.
        </p>
      </div>
    );
  }

  return (
    <div className="mb-6 overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)]">
            <tr>
              <th className="px-4 py-3 font-semibold sm:px-5">Created</th>
              <th className="px-4 py-3 font-semibold">Status</th>
              <th className="px-4 py-3 font-semibold">Voice</th>
              <th className="px-4 py-3 font-semibold">Duration</th>
              <th className="px-4 py-3 font-semibold">Preview</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => {
              const processingStatus = normalizeSpeechProcessingStatus(
                record.processing_status,
                record.processing_error,
              );
              const voiceLabel =
                record.saved_voice_id ||
                record.speaker ||
                record.model_id ||
                "Generated voice";

              return (
                <tr
                  key={record.id}
                  aria-label={`Open text-to-speech ${record.audio_filename || record.id}`}
                  className="cursor-pointer border-t border-[var(--border-muted)] transition-colors hover:bg-[var(--bg-surface-1)]"
                  onClick={() => onOpenRecord(record.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onOpenRecord(record.id);
                    }
                  }}
                  tabIndex={0}
                >
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)] sm:px-5">
                    {formatSpeechCreatedAt(record.created_at)}
                  </td>
                  <td className="px-4 py-3 align-top">
                    <StatusBadge tone={statusToneFor(processingStatus)}>
                      {speechProcessingStatusLabel(processingStatus)}
                    </StatusBadge>
                  </td>
                  <td className="px-4 py-3 align-top">
                    <div className="font-medium text-[var(--text-primary)]">
                      {voiceLabel}
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                    {formatSpeechDuration(record.audio_duration_secs)}
                  </td>
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                    <div className="max-w-[34rem] line-clamp-2 text-[var(--text-primary)]">
                      {record.input_preview}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
