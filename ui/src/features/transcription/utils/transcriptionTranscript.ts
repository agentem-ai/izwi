import type {
  TranscriptionRecord,
  TranscriptionSegment,
} from "@/shared/api/audio";

export interface TranscriptEntry {
  start: number;
  end: number;
  text: string;
}

type ExportableRecord = Pick<
  TranscriptionRecord,
  "duration_secs" | "raw_transcription" | "segments" | "transcription"
>;

export function transcriptEntriesFromRecord(
  record: ExportableRecord | null | undefined,
): TranscriptEntry[] {
  if (!record) {
    return [];
  }

  const segments = Array.isArray(record.segments) ? record.segments : [];
  const transcription =
    typeof record.transcription === "string" ? record.transcription : "";
  const rawTranscription =
    typeof record.raw_transcription === "string" ? record.raw_transcription : "";

  if (segments.length > 0) {
    return segments
      .filter(
        (segment): segment is TranscriptionSegment =>
          segment.text.trim().length > 0 && segment.end > segment.start,
      )
      .map((segment) => ({
        start: segment.start,
        end: segment.end,
        text: segment.text.trim(),
      }));
  }

  const fallbackText = transcription.trim() || rawTranscription.trim();
  if (!fallbackText) {
    return [];
  }

  return [
    {
      start: 0,
      end: Math.max(record.duration_secs ?? 0, 0.1),
      text: fallbackText,
    },
  ];
}

export function formattedTranscriptFromRecord(
  record: ExportableRecord | null | undefined,
): string {
  if (!record) {
    return "";
  }

  const entries = transcriptEntriesFromRecord(record);
  if (entries.length > 0) {
    return entries.map((entry) => entry.text).join("\n\n");
  }

  const transcription =
    typeof record.transcription === "string" ? record.transcription : "";
  const rawTranscription =
    typeof record.raw_transcription === "string" ? record.raw_transcription : "";
  return transcription.trim() || rawTranscription.trim();
}
