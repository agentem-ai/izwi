import { type FormEvent, useEffect, useMemo, useState } from "react";
import { Check, Loader2, RotateCcw, Save } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import type {
  TranscriptionRecord,
  TranscriptionSegment,
} from "@/shared/api/audio";
import { formatClockTime } from "@/features/transcription/playground/support";
import { transcriptEntriesFromRecord } from "@/features/transcription/utils/transcriptionTranscript";

interface TranscriptionCorrectionsPanelProps {
  record: Pick<
    TranscriptionRecord,
    "duration_secs" | "raw_transcription" | "segments" | "transcription"
  > | null;
  isSaving?: boolean;
  error?: string | null;
  onSave: (segments: TranscriptionSegment[]) => Promise<void> | void;
}

function normalizeSegmentText(value: string): string {
  return value
    .trim()
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/\s+/g, " "))
    .filter(Boolean)
    .join("\n");
}

export function TranscriptionCorrectionsPanel({
  record,
  isSaving = false,
  error = null,
  onSave,
}: TranscriptionCorrectionsPanelProps) {
  const baseSegments = useMemo(() => {
    if (!record) {
      return [];
    }

    if (record.segments.length > 0) {
      return record.segments;
    }

    return transcriptEntriesFromRecord(record).map((entry, index) => ({
      start: entry.start,
      end: entry.end,
      text: entry.text,
      word_start: index,
      word_end: index + 1,
    }));
  }, [record]);
  const [draftSegments, setDraftSegments] = useState<TranscriptionSegment[]>([]);
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    setDraftSegments(baseSegments);
    setSavedFlash(false);
  }, [baseSegments]);

  const normalizedSegments = useMemo(
    () =>
      draftSegments.map((segment) => ({
        ...segment,
        text: normalizeSegmentText(segment.text),
      })),
    [draftSegments],
  );

  const normalizedText = normalizedSegments
    .map((segment) => segment.text)
    .filter(Boolean)
    .join("\n\n");
  const hasDraftChanges =
    normalizedText !== (record?.transcription.trim() ?? "") ||
    normalizedSegments.some(
      (segment, index) => segment.text !== normalizeSegmentText(baseSegments[index]?.text ?? ""),
    );
  const editedSegmentCount = normalizedSegments.filter(
    (segment, index) =>
      segment.text !== normalizeSegmentText(baseSegments[index]?.text ?? ""),
  ).length;

  if (!record) {
    return (
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardContent className="py-12 text-center text-sm text-[var(--text-muted)]">
          No transcript is available yet.
        </CardContent>
      </Card>
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!hasDraftChanges || isSaving) {
      return;
    }

    await onSave(normalizedSegments);
    setSavedFlash(true);
    window.setTimeout(() => setSavedFlash(false), 1800);
  }

  return (
    <form onSubmit={(event) => void handleSubmit(event)} className="space-y-3">
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardHeader className="space-y-1 pb-3">
          <CardTitle className="text-[15px] font-semibold tracking-[-0.01em] text-[var(--text-primary)]">
            Transcript Corrections
          </CardTitle>
          <CardDescription className="text-sm leading-6 text-[var(--text-muted)]">
            Refine transcript segments while preserving timing boundaries for subtitle exports.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-2.5 sm:grid-cols-3">
            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-2.5">
              <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                Segments
              </div>
              <div className="mt-1 text-[1.5rem] font-semibold leading-none text-[var(--text-primary)]">
                {normalizedSegments.length}
              </div>
            </div>
            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-2.5">
              <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                Edited
              </div>
              <div className="mt-1 text-[1.5rem] font-semibold leading-none text-[var(--text-primary)]">
                {editedSegmentCount}
              </div>
            </div>
            <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-2.5">
              <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                Characters
              </div>
              <div className="mt-1 text-[1.5rem] font-semibold leading-none text-[var(--text-primary)]">
                {normalizedText.length}
              </div>
            </div>
          </div>

          <div className="space-y-2.5">
            {draftSegments.map((segment, index) => (
              <div
                key={`${segment.start}-${segment.end}-${index}`}
                className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2.5 py-0.5 text-[10px] font-semibold tracking-[0.14em] text-[var(--text-secondary)]">
                    {formatClockTime(segment.start)} to {formatClockTime(segment.end)}
                  </div>
                  <div className="text-[11px] text-[var(--text-muted)]">
                    Segment {index + 1}
                  </div>
                </div>
                <Textarea
                  value={segment.text}
                  onChange={(event) =>
                    setDraftSegments((current) =>
                      current.map((item, itemIndex) =>
                        itemIndex === index
                          ? { ...item, text: event.target.value }
                          : item,
                      ),
                    )
                  }
                  className="mt-3 min-h-[104px] border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-sm leading-6"
                />
              </div>
            ))}
          </div>

          {error ? (
            <div className="rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
              {error}
            </div>
          ) : null}

          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)]"
              onClick={() => setDraftSegments(baseSegments)}
              disabled={!hasDraftChanges || isSaving}
            >
              <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
              Reset
            </Button>
            <Button
              type="submit"
              size="sm"
              className="h-8 gap-1.5"
              disabled={!hasDraftChanges || isSaving}
            >
              {isSaving ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : savedFlash ? (
                <Check className="h-3.5 w-3.5" />
              ) : (
                <Save className="h-3.5 w-3.5" />
              )}
              {savedFlash ? "Saved" : "Save corrections"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </form>
  );
}
