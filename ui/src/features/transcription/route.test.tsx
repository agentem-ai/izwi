import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TranscriptionPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listTranscriptionRecords: vi.fn(),
  getTranscriptionRecord: vi.fn(),
  transcriptionRecordAudioUrl: vi.fn(),
  deleteTranscriptionRecord: vi.fn(),
  regenerateTranscriptionSummary: vi.fn(),
  createTranscriptionRecord: vi.fn(),
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
    createTranscriptionRecord: apiMocks.createTranscriptionRecord,
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

function deferredPromise<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("TranscriptionPage detail route", () => {
  beforeEach(() => {
    apiMocks.getTranscriptionRecord.mockReset();
    apiMocks.listTranscriptionRecords.mockReset();
    apiMocks.transcriptionRecordAudioUrl.mockReset();
    apiMocks.deleteTranscriptionRecord.mockReset();
    apiMocks.regenerateTranscriptionSummary.mockReset();
    apiMocks.createTranscriptionRecord.mockReset();
    hookMocks.useRouteModelSelection.mockReset();

    apiMocks.transcriptionRecordAudioUrl.mockReturnValue("/audio/transcription.wav");
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);
    apiMocks.createTranscriptionRecord.mockResolvedValue({
      id: "txr-created-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "pending",
      processing_error: null,
      duration_secs: null,
      processing_time_ms: 0,
      rtf: null,
      audio_mime_type: "audio/wav",
      audio_filename: "clip.wav",
      transcription: "",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Parakeet-TDT-0.6B-v3",
      selectedModelReady: true,
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
      await screen.findByRole("heading", { name: "Transcription" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /New transcript/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Models/i })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "No transcription jobs yet" }),
    ).toBeInTheDocument();
  });

  it("opens the new transcript modal from the header action", async () => {
    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    expect(
      await screen.findByRole("heading", { name: "New transcript" }),
    ).toBeInTheDocument();
  });

  it("redirects to /transcription/:id after an upload creates a record", async () => {
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-created-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "processing",
      processing_error: null,
      duration_secs: null,
      processing_time_ms: 0,
      rtf: null,
      audio_mime_type: "audio/wav",
      audio_filename: "clip.wav",
      transcription: "",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["audio"], "clip.wav", { type: "audio/wav" })],
      },
    });

    await waitFor(() =>
      expect(apiMocks.createTranscriptionRecord).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith(
        "txr-created-1",
      ),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
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

  it("keeps the current detail view visible while polling in the background", async () => {
    const backgroundRefresh =
      deferredPromise<Awaited<ReturnType<typeof apiMocks.getTranscriptionRecord>>>();
    const intervalCallbacks: Array<() => void> = [];
    const setIntervalSpy = vi
      .spyOn(window, "setInterval")
      .mockImplementation((handler) => {
        if (typeof handler === "function") {
          intervalCallbacks.push(handler);
        }
        return 1 as unknown as ReturnType<typeof window.setInterval>;
      });
    const clearIntervalSpy = vi
      .spyOn(window, "clearInterval")
      .mockImplementation(() => {});

    try {
      apiMocks.getTranscriptionRecord
        .mockResolvedValueOnce({
          id: "txr-polling-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: null,
          language: "English",
          processing_status: "processing",
          processing_error: null,
          duration_secs: 4,
          processing_time_ms: 120,
          rtf: 0.5,
          audio_mime_type: "audio/wav",
          audio_filename: "meeting.wav",
          transcription: "",
          segments: [],
          words: [],
          summary_status: "not_requested",
          summary_model_id: null,
          summary_text: null,
          summary_error: null,
          summary_updated_at: null,
        })
        .mockImplementationOnce(() => backgroundRefresh.promise);

      renderRoute("/transcription/txr-polling-1");

      expect(
        await screen.findByRole("heading", { name: "meeting.wav" }),
      ).toBeInTheDocument();
      expect(screen.getByText("Transcription in progress")).toBeInTheDocument();
      expect(screen.queryByText("Loading transcript...")).not.toBeInTheDocument();

      await waitFor(() => expect(setIntervalSpy).toHaveBeenCalled());
      if (intervalCallbacks.length === 0) {
        throw new Error("Expected transcription polling to register an interval.");
      }

      intervalCallbacks.forEach((callback) => callback());

      await waitFor(() =>
        expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledTimes(2),
      );

      expect(screen.getByText("Transcription in progress")).toBeInTheDocument();
      expect(screen.queryByText("Loading transcript...")).not.toBeInTheDocument();

      backgroundRefresh.resolve({
        id: "txr-polling-1",
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

      expect(await screen.findByText("Hello there.")).toBeInTheDocument();
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
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
