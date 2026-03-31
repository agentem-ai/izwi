import { AlertTriangle, Loader2 } from "lucide-react";

import { type TranscriptionRecordSummary } from "@/api";
import { Button } from "@/components/ui/button";
import {
  formatAudioDuration,
  formatCreatedAt,
} from "@/features/transcription/playground/support";

interface TranscriptionHistoryTableProps {
  records: TranscriptionRecordSummary[];
  loading?: boolean;
  error?: string | null;
  onOpenRecord: (recordId: string) => void;
  onRefresh?: () => void;
}

export function TranscriptionHistoryTable({
  records,
  loading = false,
  error = null,
  onOpenRecord,
  onRefresh,
}: TranscriptionHistoryTableProps) {
  if (loading) {
    return (
      <div className="flex min-h-[20rem] items-center justify-center rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm text-[var(--text-muted)]">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading transcriptions...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
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
      <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-10 text-center">
          <h3 className="text-lg font-semibold text-[var(--text-primary)]">
            No transcription jobs yet
          </h3>
          <p className="mt-2 text-sm text-[var(--text-muted)]">
            Queued, processing, and completed jobs will appear here.
          </p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)]">
            <tr>
              <th className="px-4 py-3 font-semibold sm:px-5">Created</th>
              <th className="px-4 py-3 font-semibold">File</th>
              <th className="px-4 py-3 font-semibold">Duration</th>
              <th className="px-4 py-3 font-semibold">Preview</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => {
              return (
                <tr
                  key={record.id}
                  aria-label={`Open transcription ${record.audio_filename || record.id}`}
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
                    {formatCreatedAt(record.created_at)}
                  </td>
                  <td className="px-4 py-3 align-top">
                    <div className="font-medium text-[var(--text-primary)]">
                      {record.audio_filename || "Audio input"}
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                    {formatAudioDuration(record.duration_secs)}
                  </td>
                  <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                    <div className="max-w-[34rem]">
                      <div className="line-clamp-2 text-[var(--text-primary)]">
                        {record.transcription_preview}
                      </div>
                      {record.summary_preview ? (
                        <div className="mt-1 line-clamp-1 text-xs text-[var(--text-muted)]">
                          Summary: {record.summary_preview}
                        </div>
                      ) : null}
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
