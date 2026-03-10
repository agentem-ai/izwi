import { describe, expect, it } from "vitest";

import type { TranscriptionRecord } from "@/shared/api/audio";
import { buildTranscriptionExport } from "./transcriptionExport";

function buildRecord(
  overrides: Partial<TranscriptionRecord> = {},
): TranscriptionRecord {
  return {
    id: "txr_export",
    created_at: 1_700_000_000_000,
    model_id: "Qwen3-ASR-0.6B",
    aligner_model_id: "ctc_forced_aligner",
    language: "English",
    duration_secs: 5.2,
    processing_time_ms: 250,
    rtf: 0.15,
    audio_mime_type: "audio/wav",
    audio_filename: "board-review.wav",
    raw_transcription: "raw transcript",
    transcription: "Corrected transcript",
    words: [
      { word: "Corrected", start: 0, end: 0.8 },
      { word: "transcript", start: 0.8, end: 1.6 },
    ],
    segments: [
      {
        start: 0,
        end: 1.6,
        text: "Corrected transcript",
        word_start: 0,
        word_end: 2,
      },
    ],
    ...overrides,
  };
}

describe("transcriptionExport", () => {
  it("builds subtitle exports from timed segments", () => {
    const record = buildRecord();

    const srt = buildTranscriptionExport(record, "srt");
    const vtt = buildTranscriptionExport(record, "vtt");

    expect(srt.filename).toBe("board-review.srt");
    expect(srt.content).toContain("1\n00:00:00,000 --> 00:00:01,600");
    expect(srt.content).toContain("Corrected transcript");

    expect(vtt.filename).toBe("board-review.vtt");
    expect(vtt.content.startsWith("WEBVTT")).toBe(true);
    expect(vtt.content).toContain("00:00:00.000 --> 00:00:01.600");
  });

  it("includes metadata in JSON exports when requested", () => {
    const record = buildRecord();
    const payload = buildTranscriptionExport(record, "json", {
      includeMetadata: true,
    });

    expect(payload.filename).toBe("board-review.json");
    expect(payload.content).toContain('"audio_filename": "board-review.wav"');
    expect(payload.content).toContain('"aligner_model_id": "ctc_forced_aligner"');
    expect(payload.content).toContain('"text": "Corrected transcript"');
  });
});
