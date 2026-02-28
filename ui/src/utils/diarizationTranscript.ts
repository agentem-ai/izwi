import type {
  DiarizationRecord,
  DiarizationResponse,
  DiarizationUtterance,
} from "../api";

export interface TranscriptEntry {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

function coerceUtteranceEntries(
  utterances: DiarizationUtterance[] | null | undefined,
): TranscriptEntry[] {
  if (!Array.isArray(utterances)) {
    return [];
  }

  return utterances
    .map((utterance) => ({
      speaker: String(utterance.speaker ?? "UNKNOWN"),
      start: Number(utterance.start ?? 0),
      end: Number(utterance.end ?? 0),
      text: String(utterance.text ?? "").trim(),
    }))
    .filter(
      (entry) =>
        entry.text.length > 0 &&
        Number.isFinite(entry.start) &&
        Number.isFinite(entry.end) &&
        entry.end > entry.start,
    );
}

function parseTranscriptEntriesFromText(text: string): TranscriptEntry[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^[-*]\s+/, "").replace(/^\d+\.\s+/, ""))
    .map((line): TranscriptEntry | null => {
      const match = line.match(
        /^([A-Za-z0-9_]+)\s+\[([0-9]+(?:\.[0-9]+)?)s\s*-\s*([0-9]+(?:\.[0-9]+)?)s\]:\s*(.*)$/,
      );
      if (!match) {
        return null;
      }
      const start = Number(match[2]);
      const end = Number(match[3]);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        return null;
      }
      return {
        speaker: match[1],
        start,
        end,
        text: match[4].trim(),
      };
    })
    .filter((entry): entry is TranscriptEntry => entry !== null);
}

function sanitizeTranscriptText(transcript: string, rawTranscript: string): string {
  const source = (transcript || rawTranscript || "").trim();
  if (!source) {
    return "";
  }

  return source
    .replace(/<think>[\s\S]*?<\/think>/gi, " ")
    .replace(/```text/gi, "")
    .replace(/```/g, "")
    .trim();
}

export function transcriptEntriesFromUtterances(
  utterances: DiarizationUtterance[] | null | undefined,
): TranscriptEntry[] {
  return coerceUtteranceEntries(utterances);
}

export function transcriptEntriesFromResult(
  result: Pick<DiarizationResponse, "utterances" | "transcript" | "raw_transcript">,
): TranscriptEntry[] {
  const entries = coerceUtteranceEntries(result.utterances);
  if (entries.length > 0) {
    return entries;
  }
  return parseTranscriptEntriesFromText(
    sanitizeTranscriptText(result.transcript, result.raw_transcript),
  );
}

export function formattedTranscriptFromResult(
  result: Pick<DiarizationResponse, "utterances" | "transcript" | "raw_transcript">,
): string {
  const entries = transcriptEntriesFromResult(result);
  if (entries.length > 0) {
    return formatTranscriptFromEntries(entries);
  }
  return sanitizeTranscriptText(result.transcript, result.raw_transcript);
}

export function transcriptEntriesFromRecord(
  record: Pick<DiarizationRecord, "utterances" | "transcript" | "raw_transcript">,
): TranscriptEntry[] {
  const entries = coerceUtteranceEntries(record.utterances);
  if (entries.length > 0) {
    return entries;
  }
  return parseTranscriptEntriesFromText(
    sanitizeTranscriptText(record.transcript, record.raw_transcript),
  );
}

export function formattedTranscriptFromRecord(
  record: Pick<DiarizationRecord, "utterances" | "transcript" | "raw_transcript">,
): string {
  const entries = transcriptEntriesFromRecord(record);
  if (entries.length > 0) {
    return formatTranscriptFromEntries(entries);
  }
  return sanitizeTranscriptText(record.transcript, record.raw_transcript);
}

export function formatTranscriptFromEntries(entries: TranscriptEntry[]): string {
  return entries
    .map(
      (entry) =>
        `${entry.speaker} [${entry.start.toFixed(2)}s - ${entry.end.toFixed(2)}s]: ${entry.text}`,
    )
    .join("\n");
}

export function previewTranscript(
  entries: TranscriptEntry[],
  transcript: string,
  rawTranscript: string,
  maxChars = 180,
): string {
  const formatted =
    entries.length > 0
      ? formatTranscriptFromEntries(entries)
      : sanitizeTranscriptText(transcript, rawTranscript);
  if (formatted.length <= maxChars) {
    return formatted;
  }
  return `${formatted.slice(0, maxChars)}...`;
}
