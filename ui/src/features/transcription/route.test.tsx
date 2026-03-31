import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TranscriptionPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listTranscriptionRecords: vi.fn(),
  getTranscriptionRecord: vi.fn(),
  transcriptionRecordAudioUrl: vi.fn(),
  deleteTranscriptionRecord: vi.fn(),
  regenerateTranscriptionSummary: vi.fn(),
}));

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    listTranscriptionRecords: apiMocks.listTranscriptionRecords,
    getTranscriptionRecord: apiMocks.getTranscriptionRecord,
    transcriptionRecordAudioUrl: apiMocks.transcriptionRecordAudioUrl,
    deleteTranscriptionRecord: apiMocks.deleteTranscriptionRecord,
    regenerateTranscriptionSummary: apiMocks.regenerateTranscriptionSummary,
  },
}));

vi.mock("@/features/models/hooks/useRouteModelSelection", () => ({
  useRouteModelSelection: hookMocks.useRouteModelSelection,
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: () => null,
}));

const baseProps = {
  models: [],
  selectedModel: null,
  loading: false,
  downloadProgress: {},
  onDownload: vi.fn(),
  onCancelDownload: vi.fn(),
  onLoad: vi.fn(),
  onUnload: vi.fn(),
  onDelete: vi.fn(),
  onSelect: vi.fn(),
  onError: vi.fn(),
};

function renderRoute(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route
          path="/transcription"
          element={<TranscriptionPage {...baseProps} />}
        />
        <Route
          path="/transcription/:recordId"
          element={<TranscriptionPage {...baseProps} />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe("TranscriptionPage detail route", () => {
  beforeEach(() => {
    apiMocks.getTranscriptionRecord.mockReset();
    apiMocks.listTranscriptionRecords.mockReset();
    apiMocks.transcriptionRecordAudioUrl.mockReset();
    apiMocks.deleteTranscriptionRecord.mockReset();
    apiMocks.regenerateTranscriptionSummary.mockReset();
    hookMocks.useRouteModelSelection.mockReset();

    apiMocks.transcriptionRecordAudioUrl.mockReturnValue("/audio/transcription.wav");
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: null,
      selectedModelReady: false,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
      handleModelSelect: vi.fn(),
      modelOptions: [],
    });

    HTMLElement.prototype.scrollIntoView = vi.fn();
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(
      () => {},
    );
  });

  it("renders the transcription history table on /transcription", async () => {
    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription history" }),
    ).toBeInTheDocument();
  });

  it("loads an existing record directly from /transcription/:id", async () => {
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-route-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "ready",
      processing_error: null,
      duration_secs: 4,
      processing_time_ms: 120,
      rtf: 0.5,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcription: "Hello there.",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription/txr-route-1");

    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith(
        "txr-route-1",
      ),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "meeting.wav" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Hello there.")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Back to transcriptions/i }),
    ).toBeInTheDocument();
  });

  it("shows route-level load errors for missing records", async () => {
    apiMocks.getTranscriptionRecord.mockRejectedValue(
      new Error("Transcription record not found"),
    );

    renderRoute("/transcription/missing");

    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith("missing"),
    );

    expect(
      await screen.findByText("Transcription record not found"),
    ).toBeInTheDocument();
  });
});
