import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SpeechHistoryPanel } from "./SpeechHistoryPanel";

const apiMocks = vi.hoisted(() => ({
  listSpeechHistoryRecords: vi.fn(),
  getSpeechHistoryRecord: vi.fn(),
  deleteSpeechHistoryRecord: vi.fn(),
  speechHistoryRecordAudioUrl: vi.fn(),
}));

vi.mock("../api", () => ({
  api: {
    listSpeechHistoryRecords: apiMocks.listSpeechHistoryRecords,
    getSpeechHistoryRecord: apiMocks.getSpeechHistoryRecord,
    deleteSpeechHistoryRecord: apiMocks.deleteSpeechHistoryRecord,
    speechHistoryRecordAudioUrl: apiMocks.speechHistoryRecordAudioUrl,
  },
}));

vi.mock("@/api", () => ({
  api: {
    listSpeechHistoryRecords: apiMocks.listSpeechHistoryRecords,
    getSpeechHistoryRecord: apiMocks.getSpeechHistoryRecord,
    deleteSpeechHistoryRecord: apiMocks.deleteSpeechHistoryRecord,
    speechHistoryRecordAudioUrl: apiMocks.speechHistoryRecordAudioUrl,
  },
}));

describe("History panels", () => {
  beforeEach(() => {
    apiMocks.listSpeechHistoryRecords.mockReset();
    apiMocks.getSpeechHistoryRecord.mockReset();
    apiMocks.deleteSpeechHistoryRecord.mockReset();
    apiMocks.speechHistoryRecordAudioUrl.mockReset();

    apiMocks.speechHistoryRecordAudioUrl.mockReturnValue("/audio/speech.wav");

    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("keeps the speech history drawer open while confirming a delete", async () => {
    apiMocks.listSpeechHistoryRecords.mockResolvedValue([
      {
        id: "speech-1",
        created_at: 1,
        route_kind: "text_to_speech",
        model_id: "Kokoro-82M",
        speaker: "Vivian",
        language: "en",
        input_preview: "Hello from saved speech history.",
        input_chars: 31,
        generation_time_ms: 80,
        audio_duration_secs: 1.2,
        rtf: 0.4,
        tokens_generated: 12,
        audio_mime_type: "audio/wav",
        audio_filename: "speech.wav",
      },
    ]);
    apiMocks.getSpeechHistoryRecord.mockResolvedValue({
      id: "speech-1",
      created_at: 1,
      model_id: "Kokoro-82M",
      speaker: "Vivian",
      language: "en",
      input_text: "Hello from saved speech history.",
      generation_time_ms: 80,
      audio_duration_secs: 1.2,
      rtf: 0.4,
      tokens_generated: 12,
      audio_mime_type: "audio/wav",
      audio_filename: "speech.wav",
    });
    apiMocks.deleteSpeechHistoryRecord.mockResolvedValue(undefined);

    render(
      <SpeechHistoryPanel
        route="text-to-speech"
        title="Speech History"
        emptyMessage="No speech history yet."
      />,
    );

    await waitFor(() =>
      expect(apiMocks.listSpeechHistoryRecords).toHaveBeenCalledWith(
        "text-to-speech",
      ),
    );

    const historyButton = screen.getByRole("button", { name: /History/i });
    expect(historyButton).not.toHaveClass("fixed");
    fireEvent.click(historyButton);

    expect(await screen.findByText("Speech History")).toBeInTheDocument();
    expect(screen.queryByTitle("Refresh history")).not.toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Delete speech.wav" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete speech.wav" }));

    expect(
      await screen.findByRole("button", { name: "Delete record" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Speech History")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete record" }));

    await waitFor(() =>
      expect(apiMocks.deleteSpeechHistoryRecord).toHaveBeenCalledWith(
        "text-to-speech",
        "speech-1",
      ),
    );
  });
});
