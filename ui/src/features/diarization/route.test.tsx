import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelInfo } from "@/api";

import { DiarizationPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listDiarizationRecords: vi.fn(),
  getDiarizationRecord: vi.fn(),
  updateDiarizationRecord: vi.fn(),
  rerunDiarizationRecord: vi.fn(),
  regenerateDiarizationSummary: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  createDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    updateDiarizationRecord: apiMocks.updateDiarizationRecord,
    rerunDiarizationRecord: apiMocks.rerunDiarizationRecord,
    regenerateDiarizationSummary: apiMocks.regenerateDiarizationSummary,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    createDiarizationRecord: apiMocks.createDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
  },
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: () => null,
}));

const baseModels: ModelInfo[] = [
  {
    variant: "diar_streaming_sortformer_4spk-v2.1",
    status: "ready" as const,
    local_path: "/models/diar",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Parakeet-TDT-0.6B-v3",
    status: "ready" as const,
    local_path: "/models/asr",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Qwen3-ForcedAligner-0.6B",
    status: "ready" as const,
    local_path: "/models/aligner",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Qwen3.5-4B",
    status: "ready" as const,
    local_path: "/models/llm",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
];

const baseProps = {
  models: baseModels,
  selectedModel: "diar_streaming_sortformer_4spk-v2.1",
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

function createRouteProps(
  overrides: Partial<typeof baseProps> = {},
): typeof baseProps {
  return {
    ...baseProps,
    onDownload: vi.fn(),
    onCancelDownload: vi.fn(),
    onLoad: vi.fn(),
    onUnload: vi.fn(),
    onDelete: vi.fn(),
    onSelect: vi.fn(),
    onError: vi.fn(),
    ...overrides,
  };
}

function renderRoute(
  initialEntry: string,
  props: typeof baseProps = createRouteProps(),
) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/diarization" element={<DiarizationPage {...props} />} />
        <Route
          path="/diarization/:recordId"
          element={<DiarizationPage {...props} />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

const pendingSummaryRecord = {
  id: "diar-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  speaker_count: 2,
  corrected_speaker_count: 2,
  duration_secs: 42,
  processing_time_ms: 120,
  rtf: 0.5,
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
  transcript_preview: "Hello there.",
  transcript_chars: 12,
  summary_status: "pending",
  summary_preview: null,
  summary_chars: 0,
};

const readySummaryRecord = {
  ...pendingSummaryRecord,
  id: "diar-2",
  audio_filename: "board-call.wav",
  summary_status: "ready",
  summary_preview: "Board sync covered runway, launch timing, and next hiring steps.",
  summary_chars: 63,
};

const fullRecord = {
  id: "diar-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: "Qwen3.5-4B",
  min_speakers: 1,
  max_speakers: 4,
  min_speech_duration_ms: 240,
  min_silence_duration_ms: 200,
  enable_llm_refinement: true,
  processing_time_ms: 120,
  duration_secs: 42,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 0.82,
  unattributed_words: 0,
  llm_refined: true,
  asr_text: "Hello there.",
  raw_transcript: "Speaker 1: Hello there.",
  transcript: "Speaker 1: Hello there.",
  summary_status: "pending",
  summary_model_id: "Qwen3.5-4B",
  summary_text: null,
  summary_error: null,
  summary_updated_at: null,
  segments: [],
  words: [],
  utterances: [
    {
      speaker: "SPEAKER_00",
      start: 0,
      end: 1,
      text: "Hello there.",
    },
  ],
  speaker_name_overrides: {},
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
};

describe("DiarizationPage routes", () => {
  beforeEach(() => {
    vi.useRealTimers();
    apiMocks.listDiarizationRecords.mockReset();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.updateDiarizationRecord.mockReset();
    apiMocks.rerunDiarizationRecord.mockReset();
    apiMocks.regenerateDiarizationSummary.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.createDiarizationRecord.mockReset();
    apiMocks.diarizationRecordAudioUrl.mockReset();

    apiMocks.listDiarizationRecords.mockResolvedValue([]);
    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);
    apiMocks.createDiarizationRecord.mockResolvedValue(fullRecord);
    apiMocks.diarizationRecordAudioUrl.mockReturnValue("/audio/meeting.wav");
  });

  it("renders the diarization route and loads history from the route hook", async () => {
    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    expect(
      await screen.findByRole("heading", { name: "Diarization" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /New diarization/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "History" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("No diarization records yet")).toBeInTheDocument();
  });

  it("shows summaries in the diarization history table", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([readySummaryRecord]);

    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    expect(screen.getByRole("columnheader", { name: "Summary" })).toBeInTheDocument();
    expect(
      screen.getByText(
        "Board sync covered runway, launch timing, and next hiring steps.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("diar_streaming_sortformer_4spk-v2.1"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Hello there.")).not.toBeInTheDocument();
  });

  it("opens the creation modal and routes new diarization runs to their detail page", async () => {
    let resolveRefreshHistory: ((records: unknown[]) => void) | null = null;
    apiMocks.listDiarizationRecords
      .mockResolvedValueOnce([])
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRefreshHistory = resolve;
          }),
      );

    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New diarization/i }));

    expect(
      await screen.findByRole("heading", { name: "New diarization" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Choose how to start")).toBeInTheDocument();
    expect(screen.getByText("Review run settings")).toBeInTheDocument();

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["audio"], "meeting.wav", { type: "audio/wav" })],
      },
    });

    await waitFor(() =>
      expect(apiMocks.createDiarizationRecord).toHaveBeenCalledTimes(1),
    );
    expect(apiMocks.createDiarizationRecord).toHaveBeenCalledWith(
      expect.objectContaining({
        audio_filename: "meeting.wav",
      }),
    );
    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    expect(
      await screen.findByRole("heading", { name: "Diarization Record" }),
    ).toBeInTheDocument();

    await act(async () => {
      if (resolveRefreshHistory) {
        resolveRefreshHistory([]);
      }
    });
  });

  it("loads all diarization stack models from the modal readiness controls", async () => {
    const props = createRouteProps({
      models: [
        {
          variant: "diar_streaming_sortformer_4spk-v2.1",
          status: "downloaded" as const,
          local_path: "/models/diar",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Parakeet-TDT-0.6B-v3",
          status: "not_downloaded" as const,
          local_path: "/models/asr",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Qwen3-ForcedAligner-0.6B",
          status: "ready" as const,
          local_path: "/models/aligner",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Qwen3.5-4B",
          status: "downloaded" as const,
          local_path: "/models/llm",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
      ],
      selectedModel: "diar_streaming_sortformer_4spk-v2.1",
    });

    renderRoute("/diarization", props);

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New diarization/i }));
    fireEvent.click(await screen.findByRole("button", { name: "Load all models" }));

    expect(props.onLoad).toHaveBeenCalledWith("diar_streaming_sortformer_4spk-v2.1");
    expect(props.onLoad).toHaveBeenCalledWith("Qwen3.5-4B");
    expect(props.onDownload).toHaveBeenCalledWith("Parakeet-TDT-0.6B-v3");
    expect(props.onUnload).not.toHaveBeenCalled();
  });

  it("unloads all ready diarization stack models from the modal readiness controls", async () => {
    const props = createRouteProps();

    renderRoute("/diarization", props);

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New diarization/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Unload all models" }),
    );

    expect(props.onUnload).toHaveBeenCalledWith("diar_streaming_sortformer_4spk-v2.1");
    expect(props.onUnload).toHaveBeenCalledWith("Parakeet-TDT-0.6B-v3");
    expect(props.onUnload).toHaveBeenCalledWith("Qwen3-ForcedAligner-0.6B");
    expect(props.onUnload).toHaveBeenCalledWith("Qwen3.5-4B");
  });

  it("loads the selected diarization record on /diarization/:recordId", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);
    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);

    renderRoute("/diarization/diar-1");

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    expect(await screen.findByText("meeting.wav")).toBeInTheDocument();
    expect(screen.getByText("Diarization Record")).toBeInTheDocument();
  });

  it("opens saved diarization records from the history table", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);

    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByText("meeting.wav"));

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
    expect(
      await screen.findByRole("heading", { name: "Diarization Record" }),
    ).toBeInTheDocument();
  });

  it("navigates back to the diarization index from a record page", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);

    renderRoute("/diarization/diar-1");

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /Back to diarization/i }),
    );

    expect(
      await screen.findByRole("heading", { name: "Diarization" }),
    ).toBeInTheDocument();
  });

  it("deletes a diarization record from the detail page and returns to history", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);
    apiMocks.deleteDiarizationRecord.mockResolvedValue({
      id: "diar-1",
      deleted: true,
    });

    renderRoute("/diarization/diar-1");

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    fireEvent.click(await screen.findByRole("button", { name: /^Delete$/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /Delete record/i }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
    expect(
      await screen.findByRole("heading", { name: "Diarization" }),
    ).toBeInTheDocument();
  });

  it("does not keep polling diarization history while viewing the route", async () => {
    vi.useFakeTimers();
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);

    renderRoute("/diarization");

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(2600);
    });

    expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1);
  });

  it("polls the selected record while its summary is pending", async () => {
    vi.useFakeTimers();
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);
    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);

    renderRoute("/diarization/diar-1");

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.getDiarizationRecord).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(2600);
    });

    expect(apiMocks.getDiarizationRecord).toHaveBeenCalledTimes(2);
  });
});
