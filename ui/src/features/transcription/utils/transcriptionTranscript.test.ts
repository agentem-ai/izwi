import { describe, expect, it } from "vitest";

import type { TranscriptionRecord } from "@/shared/api/audio";
import {
  formattedTranscriptFromRecord,
  transcriptEntriesFromRecord,
} from "./transcriptionTranscript";

function buildRecord(
  overrides: Partial<TranscriptionRecord> = {},
): TranscriptionRecord {
  return {
    id: "txr_1",
    created_at: 1,
    model_id: "Qwen3-ASR-0.6B",
    aligner_model_id: "ctc_forced_aligner",
    language: "English",
    duration_secs: 12.3,
    processing_time_ms: 400,
    rtf: 0.2,
    audio_mime_type: "audio/wav",
    audio_filename: "meeting.wav",
    raw_transcription: "hello world",
    transcription: "hello world",
    words: [],
    segments: [],
    ...overrides,
  };
}

describe("transcriptionTranscript", () => {
  it("prefers timed segments when they are available", () => {
    const record = buildRecord({
      transcription: "Corrected final text",
      segments: [
        {
          start: 0,
          end: 1.2,
          text: "Hello world",
          word_start: 0,
          word_end: 2,
        },
        {
          start: 1.2,
          end: 2.8,
          text: "Second line",
          word_start: 2,
          word_end: 4,
        },
      ],
    });

    expect(transcriptEntriesFromRecord(record)).toEqual([
      { start: 0, end: 1.2, text: "Hello world" },
      { start: 1.2, end: 2.8, text: "Second line" },
    ]);
    expect(formattedTranscriptFromRecord(record)).toBe(
      "Hello world\n\nSecond line",
    );
  });

  it("falls back to the flat transcript when no segments exist", () => {
    const record = buildRecord({
      duration_secs: 8.5,
      raw_transcription: "Raw fallback",
      transcription: "Corrected fallback",
    });

    expect(transcriptEntriesFromRecord(record)).toEqual([
      { start: 0, end: 8.5, text: "Corrected fallback" },
    ]);
    expect(formattedTranscriptFromRecord(record)).toBe("Corrected fallback");
  });
});
