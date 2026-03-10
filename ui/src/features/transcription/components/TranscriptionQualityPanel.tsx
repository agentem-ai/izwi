import { AlertTriangle, Clock3, FileText, Languages } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { TranscriptionRecord } from "@/shared/api/audio";
import { formatAudioDuration } from "@/features/transcription/playground/support";

interface TranscriptionQualityPanelProps {
  record: Pick<
    TranscriptionRecord,
    | "aligner_model_id"
    | "duration_secs"
    | "language"
    | "model_id"
    | "processing_time_ms"
    | "raw_transcription"
    | "rtf"
    | "segments"
    | "transcription"
    | "words"
  > | null;
  onExport?: () => void;
}

function qualityWarnings(
  record: NonNullable<TranscriptionQualityPanelProps["record"]>,
): Array<{ title: string; description: string }> {
  const warnings: Array<{ title: string; description: string }> = [];

  if (!record.aligner_model_id || record.words.length === 0) {
    warnings.push({
      title: "Timing metadata unavailable",
      description:
        "This transcript can still be reviewed and exported, but subtitle timing quality may be limited until forced alignment succeeds.",
    });
  }

  if (record.segments.length <= 1) {
    warnings.push({
      title: "Transcript structure is coarse",
      description:
        "The transcript is currently stored as a single block. Corrections and subtitle exports work best when the transcript is split into multiple timed segments.",
    });
  }

  if (
    record.raw_transcription.trim() &&
    record.transcription.trim() !== record.raw_transcription.trim()
  ) {
    warnings.push({
      title: "Corrections applied",
      description:
        "Saved transcript text differs from the raw ASR output. Exported files will use the corrected wording.",
    });
  }

  return warnings;
}

export function TranscriptionQualityPanel({
  record,
  onExport,
}: TranscriptionQualityPanelProps) {
  if (!record) {
    return (
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardContent className="py-12 text-center text-sm text-[var(--text-muted)]">
          No transcript quality data is available yet.
        </CardContent>
      </Card>
    );
  }

  const warnings = qualityWarnings(record);
  const runtimeMs = Math.max(0, Math.round(record.processing_time_ms));

  return (
    <div className="space-y-3">
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardHeader className="space-y-1 pb-3">
          <CardTitle className="text-[15px] font-semibold tracking-[-0.01em] text-[var(--text-primary)]">
            Transcript Quality
          </CardTitle>
          <CardDescription className="text-sm leading-6 text-[var(--text-muted)]">
            Review timing coverage, transcript structure, and export readiness before sharing.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                <Clock3 className="h-3.5 w-3.5" />
                Runtime
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                {runtimeMs} ms
              </div>
              <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                RTF {record.rtf ?? "Unknown"}
              </div>
            </div>
            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                <FileText className="h-3.5 w-3.5" />
                Structure
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                {record.segments.length} segments
              </div>
              <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                {record.words.length} timed words
              </div>
            </div>
            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                <Languages className="h-3.5 w-3.5" />
                Language
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                {record.language || "Unknown"}
              </div>
              <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                {formatAudioDuration(record.duration_secs)}
              </div>
            </div>
            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                Models
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                {record.model_id || "Unknown ASR"}
              </div>
              <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                {record.aligner_model_id || "No aligner metadata"}
              </div>
            </div>
          </div>

          {warnings.length > 0 ? (
            <div className="space-y-2">
              {warnings.map((warning) => (
                <div
                  key={warning.title}
                  className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-3"
                >
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[var(--danger-text)]" />
                    <div>
                      <div className="text-sm font-semibold text-[var(--danger-text)]">
                        {warning.title}
                      </div>
                      <div className="mt-1 text-xs leading-5 text-[var(--danger-text)]/90">
                        {warning.description}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          {onExport ? (
            <div className="flex justify-end">
              <Button
                type="button"
                size="sm"
                className="h-8 gap-1.5"
                onClick={onExport}
              >
                Export transcript
              </Button>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
