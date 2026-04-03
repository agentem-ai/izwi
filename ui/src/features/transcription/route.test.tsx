import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { TranscriptionPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listTranscriptionRecords: vi.fn(),
  getTranscriptionRecord: vi.fn(),
  transcriptionRecordAudioUrl: vi.fn(),
  deleteTranscriptionRecord: vi.fn(),
  regenerateTranscriptionSummary: vi.fn(),
  createTranscriptionRecord: vi.fn(),
  createTranscriptionRecordStream: vi.fn(),
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
    createTranscriptionRecordStream: apiMocks.createTranscriptionRecordStream,
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
    <NotificationProvider>
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
      </MemoryRouter>
    </NotificationProvider>,
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
    apiMocks.createTranscriptionRecordStream.mockReset();
    hookMocks.useRouteModelSelection.mockReset();

    apiMocks.transcriptionRecordAudioUrl.mockReturnValue("/audio/transcription.wav");
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);
    apiMocks.deleteTranscriptionRecord.mockResolvedValue(undefined);
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
    apiMocks.createTranscriptionRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onCreated?.({
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
        return new AbortController();
      },
    );
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

  it("shows standard row actions from the history menu", async () => {
    apiMocks.listTranscriptionRecords.mockResolvedValue([
      {
        id: "txr-history-1",
        created_at: 1,
        model_id: "Parakeet-TDT-0.6B-v3",
        language: "English",
        duration_secs: 4,
        processing_status: "ready",
        processing_error: null,
        processing_time_ms: 120,
        rtf: 0.5,
        audio_mime_type: "audio/wav",
        audio_filename: "meeting.wav",
        transcription_preview: "Hello there.",
        transcription_chars: 12,
        summary_status: "ready",
        summary_preview: "Short summary",
        summary_chars: 13,
      },
    ]);

    renderRoute("/transcription");

    expect(await screen.findByText("meeting.wav")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: /More actions for meeting\.wav/i }),
      { button: 0, ctrlKey: false },
    );

    expect(await screen.findByRole("menuitem", { name: /Open record/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /Copy transcript/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /^Export$/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /^Delete$/i })).toBeVisible();
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
    expect(screen.getByText("Bring in a recording")).toBeInTheDocument();
    expect(screen.getByText("Review job settings")).toBeInTheDocument();
    expect(screen.getByText("Upload audio")).toBeInTheDocument();
    expect(screen.getByText("Stream results")).toBeInTheDocument();
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
      expect(apiMocks.createTranscriptionRecordStream).toHaveBeenCalled(),
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

  it("shows streamed transcript deltas on the detail page while processing", async () => {
    let streamCallbacks:
      | {
          onStart?: () => void;
          onDelta?: (delta: string) => void;
          onFinal?: (record: unknown) => void;
        }
      | undefined;

    apiMocks.createTranscriptionRecordStream.mockImplementationOnce(
      (_request, callbacks) => {
        streamCallbacks = callbacks;
        callbacks.onCreated?.({
          id: "txr-stream-1",
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
          audio_filename: "streamed.wav",
          transcription: "",
          segments: [],
          words: [],
          summary_status: "not_requested",
          summary_model_id: null,
          summary_text: null,
          summary_error: null,
          summary_updated_at: null,
        });
        return new AbortController();
      },
    );
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-stream-1",
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
      audio_filename: "streamed.wav",
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
        files: [new File(["audio"], "streamed.wav", { type: "audio/wav" })],
      },
    });

    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
    ).toBeInTheDocument();

    await act(async () => {
      streamCallbacks?.onStart?.();
      streamCallbacks?.onDelta?.("Hello ");
      streamCallbacks?.onDelta?.("world");
    });

    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("refreshes transcription history after creating a record", async () => {
    apiMocks.listTranscriptionRecords
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: "txr-created-1",
          created_at: 1,
          audio_filename: "clip.wav",
          duration_secs: null,
          processing_status: "pending",
          processing_error: null,
          transcription_preview: "",
          summary_status: "not_requested",
          summary_preview: null,
        },
      ]);
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
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(1),
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
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(2),
    );
    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /Back to transcriptions/i }),
    );

    expect(await screen.findByText("clip.wav")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "No transcription jobs yet" }),
    ).not.toBeInTheDocument();
  });

  it("accepts drag and drop on the upload area", async () => {
    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    const uploadArea = await screen.findByRole("button", {
      name: "Upload audio file",
    });
    const file = new File(["audio"], "dragged-clip.wav", {
      type: "audio/wav",
    });

    fireEvent.dragOver(uploadArea, {
      dataTransfer: { files: [file] },
    });
    fireEvent.drop(uploadArea, {
      dataTransfer: { files: [file] },
    });

    await waitFor(() =>
      expect(apiMocks.createTranscriptionRecordStream).toHaveBeenCalledWith(
        expect.objectContaining({
          audio_file: file,
          audio_filename: "dragged-clip.wav",
        }),
        expect.any(Object),
      ),
    );
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
    expect(screen.queryByText(/^Ready$/)).not.toBeInTheDocument();
    expect(screen.getByTestId("transcription-review-player")).toHaveClass(
      "fixed",
    );
  });

  it("confirms deletion before removing a transcription record", async () => {
    apiMocks.listTranscriptionRecords
      .mockResolvedValueOnce([
        {
          id: "txr-delete-1",
          created_at: 1,
          audio_filename: "meeting.wav",
          duration_secs: 4,
          processing_status: "ready",
          processing_error: null,
          transcription_preview: "Hello there.",
          summary_status: "not_requested",
          summary_preview: null,
        },
      ])
      .mockResolvedValueOnce([]);
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-delete-1",
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

    renderRoute("/transcription/txr-delete-1");

    expect(
      await screen.findByRole("heading", { name: "meeting.wav" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^Delete$/i }));

    expect(
      await screen.findByText(
        /This permanently removes the saved audio and transcript from history\./i,
      ),
    ).toBeInTheDocument();
    expect(screen.getAllByText("meeting.wav").length).toBeGreaterThan(0);
    expect(apiMocks.deleteTranscriptionRecord).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("button", { name: "Delete transcription" }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteTranscriptionRecord).toHaveBeenCalledWith(
        "txr-delete-1",
      ),
    );
    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(2),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Open transcription meeting.wav"),
    ).not.toBeInTheDocument();
  });

  it("confirms deletion from the history menu and refreshes the table", async () => {
    apiMocks.listTranscriptionRecords
      .mockResolvedValueOnce([
        {
          id: "txr-history-delete-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          language: "English",
          duration_secs: 4,
          processing_status: "ready",
          processing_error: null,
          processing_time_ms: 120,
          rtf: 0.5,
          audio_mime_type: "audio/wav",
          audio_filename: "meeting.wav",
          transcription_preview: "Hello there.",
          transcription_chars: 12,
          summary_status: "not_requested",
          summary_preview: null,
          summary_chars: 0,
        },
      ])
      .mockResolvedValueOnce([]);

    renderRoute("/transcription");

    expect(await screen.findByText("meeting.wav")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: /More actions for meeting\.wav/i }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Delete$/i }));

    expect(
      await screen.findByText(
        /This permanently removes the saved audio and transcript from history\./i,
      ),
    ).toBeInTheDocument();
    expect(apiMocks.deleteTranscriptionRecord).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("button", { name: "Delete transcription" }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteTranscriptionRecord).toHaveBeenCalledWith(
        "txr-history-delete-1",
      ),
    );
    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(2),
    );

    expect(
      screen.queryByLabelText("Open transcription meeting.wav"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "No transcription jobs yet" }),
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
