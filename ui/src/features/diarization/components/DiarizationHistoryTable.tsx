import { AlertTriangle, Loader2 } from "lucide-react";

import { type DiarizationRecordSummary } from "@/api";
import { Button } from "@/components/ui/button";

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
  onOpenRecord: (recordId: string) => void;
  onRefresh?: () => void;
}

export function DiarizationHistoryTable({
  records,
  loading = false,
  error = null,
  onOpenRecord,
  onRefresh,
}: DiarizationHistoryTableProps) {
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
    <div className="mb-6 overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)]">
            <tr>
              <th className="px-4 py-3 font-semibold sm:px-5">Created</th>
              <th className="px-4 py-3 font-semibold">File</th>
              <th className="px-4 py-3 font-semibold">Speakers</th>
              <th className="px-4 py-3 font-semibold">Duration</th>
              <th className="px-4 py-3 font-semibold">Preview</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr
                key={record.id}
                aria-label={`Open diarization ${record.audio_filename || record.id}`}
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
                  {record.model_id ? (
                    <div className="mt-1 text-xs text-[var(--text-muted)]">
                      {record.model_id}
                    </div>
                  ) : null}
                </td>
                <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                  {record.corrected_speaker_count ?? record.speaker_count}
                </td>
                <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                  {formatAudioDuration(record.duration_secs)}
                </td>
                <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                  <div className="max-w-[34rem]">
                    <div className="line-clamp-2 text-[var(--text-primary)]">
                      {record.transcript_preview}
                    </div>
                    {record.summary_preview ? (
                      <div className="mt-1 line-clamp-1 text-xs text-[var(--text-muted)]">
                        Summary: {record.summary_preview}
                      </div>
                    ) : null}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
