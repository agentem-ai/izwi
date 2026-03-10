import type { TranscriptionRecord } from "@/shared/api/audio";
import {
  formattedTranscriptFromRecord,
  transcriptEntriesFromRecord,
} from "./transcriptionTranscript";

export type TranscriptionExportFormat = "txt" | "json" | "srt" | "vtt";

export interface TranscriptionExportOptions {
  includeMetadata?: boolean;
}

export interface TranscriptionExportPayload {
  content: string;
  extension: string;
  filename: string;
  mimeType: string;
}

type ExportableRecord = Pick<
  TranscriptionRecord,
  | "id"
  | "created_at"
  | "model_id"
  | "aligner_model_id"
  | "language"
  | "duration_secs"
  | "audio_filename"
  | "raw_transcription"
  | "transcription"
  | "segments"
  | "words"
>;

function stripExtension(filename: string): string {
  return filename.replace(/\.[^.]+$/, "");
}

function baseFilename(record: ExportableRecord): string {
  const source = record.audio_filename?.trim() || `transcription-${record.id}`;
  return stripExtension(source);
}

function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0.00";
  }
  return seconds.toFixed(2);
}

function formatSrtTimestamp(totalSeconds: number): string {
  const clamped = Math.max(0, totalSeconds);
  const hours = Math.floor(clamped / 3600);
  const minutes = Math.floor((clamped % 3600) / 60);
  const seconds = Math.floor(clamped % 60);
  const milliseconds = Math.round((clamped - Math.floor(clamped)) * 1000);
  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")},${milliseconds
    .toString()
    .padStart(3, "0")}`;
}

function formatVttTimestamp(totalSeconds: number): string {
  return formatSrtTimestamp(totalSeconds).replace(",", ".");
}

function metadataBlock(record: ExportableRecord): string[] {
  return [
    `File: ${record.audio_filename ?? `${record.id}.wav`}`,
    `ASR Model: ${record.model_id ?? "Unknown model"}`,
    `Aligner Model: ${record.aligner_model_id ?? "Unavailable"}`,
    `Language: ${record.language ?? "Unknown"}`,
    `Duration: ${formatSeconds(record.duration_secs ?? 0)}s`,
    `Created At: ${new Date(record.created_at).toISOString()}`,
  ];
}

function txtContent(
  record: ExportableRecord,
  options: TranscriptionExportOptions,
): string {
  const transcript = formattedTranscriptFromRecord(record);
  if (!options.includeMetadata) {
    return transcript;
  }
  return `${metadataBlock(record).join("\n")}\n\n${transcript}`;
}

function jsonContent(
  record: ExportableRecord,
  options: TranscriptionExportOptions,
): string {
  const entries = transcriptEntriesFromRecord(record);
  const payload = {
    metadata: options.includeMetadata
      ? {
          id: record.id,
          created_at: record.created_at,
          model_id: record.model_id,
          aligner_model_id: record.aligner_model_id,
          language: record.language,
          duration_secs: record.duration_secs,
          audio_filename: record.audio_filename,
        }
      : undefined,
    raw_transcription: record.raw_transcription,
    transcription: record.transcription,
    segments: entries,
    words: record.words,
  };

  return JSON.stringify(payload, null, 2);
}

function subtitleContent(
  record: ExportableRecord,
  format: "srt" | "vtt",
): string {
  const entries = transcriptEntriesFromRecord(record);

  const lines = entries.map((entry, index) => {
    const timestamp =
      format === "srt"
        ? `${formatSrtTimestamp(entry.start)} --> ${formatSrtTimestamp(entry.end)}`
        : `${formatVttTimestamp(entry.start)} --> ${formatVttTimestamp(entry.end)}`;

    if (format === "srt") {
      return `${index + 1}\n${timestamp}\n${entry.text}`;
    }

    return `${timestamp}\n${entry.text}`;
  });

  if (format === "vtt") {
    return `WEBVTT\n\n${lines.join("\n\n")}`;
  }

  return lines.join("\n\n");
}

export function buildTranscriptionExport(
  record: ExportableRecord,
  format: TranscriptionExportFormat,
  options: TranscriptionExportOptions = {},
): TranscriptionExportPayload {
  const includeMetadata = options.includeMetadata ?? format === "json";
  const normalizedOptions = { includeMetadata };

  switch (format) {
    case "txt":
      return {
        content: txtContent(record, normalizedOptions),
        extension: "txt",
        filename: `${baseFilename(record)}.txt`,
        mimeType: "text/plain; charset=utf-8",
      };
    case "json":
      return {
        content: jsonContent(record, normalizedOptions),
        extension: "json",
        filename: `${baseFilename(record)}.json`,
        mimeType: "application/json; charset=utf-8",
      };
    case "srt":
      return {
        content: subtitleContent(record, "srt"),
        extension: "srt",
        filename: `${baseFilename(record)}.srt`,
        mimeType: "application/x-subrip; charset=utf-8",
      };
    case "vtt":
      return {
        content: subtitleContent(record, "vtt"),
        extension: "vtt",
        filename: `${baseFilename(record)}.vtt`,
        mimeType: "text/vtt; charset=utf-8",
      };
    default:
      return {
        content: txtContent(record, normalizedOptions),
        extension: "txt",
        filename: `${baseFilename(record)}.txt`,
        mimeType: "text/plain; charset=utf-8",
      };
  }
}
