import { afterEach, describe, expect, it, vi } from "vitest";

import { AudioApiClient, type DiarizationRecord } from "@/shared/api/audio";
import { ApiHttpClient } from "@/shared/api/http";

const updatedRecord = {
  id: "diar-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Qwen3-ASR-0.6B",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: null,
  min_speakers: 1,
  max_speakers: 4,
  min_speech_duration_ms: 240,
  min_silence_duration_ms: 200,
  enable_llm_refinement: false,
  processing_time_ms: 120,
  duration_secs: 6,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 1,
  unattributed_words: 0,
  llm_refined: false,
  asr_text: "Hello there. Hi back.",
  raw_transcript: "",
  transcript: "",
  segments: [],
  words: [],
  utterances: [],
  speaker_name_overrides: {
    SPEAKER_00: "Alice",
  },
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
} satisfies DiarizationRecord;

describe("AudioApiClient.updateDiarizationRecord", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("retries with PUT when PATCH is rejected", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            error: { message: "Method Not Allowed" },
          }),
          {
            status: 405,
            headers: {
              "Content-Type": "application/json",
            },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(updatedRecord), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const result = await client.updateDiarizationRecord("diar-1", {
      speaker_name_overrides: {
        SPEAKER_00: "Alice",
      },
    });

    expect(result).toEqual(updatedRecord);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost/v1/diarizations/diar-1",
      expect.objectContaining({ method: "PATCH" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost/v1/diarizations/diar-1",
      expect.objectContaining({ method: "PUT" }),
    );
  });
});
