import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TextToSpeechPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listTextToSpeechRecords: vi.fn(),
  getTextToSpeechRecord: vi.fn(),
  textToSpeechRecordAudioUrl: vi.fn(),
  deleteTextToSpeechRecord: vi.fn(),
  createTextToSpeechRecord: vi.fn(),
  createTextToSpeechRecordStream: vi.fn(),
  listSavedVoices: vi.fn(),
  downloadAudioFile: vi.fn(),
}));

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    listTextToSpeechRecords: apiMocks.listTextToSpeechRecords,
    getTextToSpeechRecord: apiMocks.getTextToSpeechRecord,
    textToSpeechRecordAudioUrl: apiMocks.textToSpeechRecordAudioUrl,
    deleteTextToSpeechRecord: apiMocks.deleteTextToSpeechRecord,
    createTextToSpeechRecord: apiMocks.createTextToSpeechRecord,
    createTextToSpeechRecordStream: apiMocks.createTextToSpeechRecordStream,
    listSavedVoices: apiMocks.listSavedVoices,
    downloadAudioFile: apiMocks.downloadAudioFile,
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

function buildRecord(
  overrides: Partial<Record<string, unknown>> = {},
) {
  return {
    id: "tts-1",
    created_at: 1,
    route_kind: "text_to_speech",
    processing_status: "pending",
    processing_error: null,
    model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
    speaker: "Vivian",
    language: null,
    saved_voice_id: null,
    speed: 1,
    input_text: "Hello world",
    voice_description: null,
    reference_text: null,
    generation_time_ms: 0,
    audio_duration_secs: null,
    rtf: null,
    tokens_generated: null,
    audio_mime_type: "audio/wav",
    audio_filename: "tts.wav",
    ...overrides,
  };
}

function buildSummary(
  overrides: Partial<Record<string, unknown>> = {},
) {
  return {
    id: "tts-1",
    created_at: 1,
    route_kind: "text_to_speech",
    processing_status: "pending",
    processing_error: null,
    model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
    speaker: "Vivian",
    language: null,
    input_preview: "Hello world",
    input_chars: 11,
    generation_time_ms: 0,
    audio_duration_secs: null,
    rtf: null,
    tokens_generated: null,
    audio_mime_type: "audio/wav",
    audio_filename: "tts.wav",
    ...overrides,
  };
}

function buildSavedVoice(id: string, name: string) {
  return {
    id,
    created_at: 1,
    updated_at: 1,
    name,
    reference_text_preview: "preview",
    reference_text_chars: 7,
    audio_mime_type: "audio/wav",
    audio_filename: "voice.wav",
    source_route_kind: null,
    source_record_id: null,
  };
}

function renderRoute(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route
          path="/text-to-speech"
          element={<TextToSpeechPage {...baseProps} />}
        />
        <Route
          path="/text-to-speech/:recordId"
          element={<TextToSpeechPage {...baseProps} />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe("TextToSpeechPage", () => {
  beforeEach(() => {
    apiMocks.listTextToSpeechRecords.mockReset();
    apiMocks.getTextToSpeechRecord.mockReset();
    apiMocks.textToSpeechRecordAudioUrl.mockReset();
    apiMocks.deleteTextToSpeechRecord.mockReset();
    apiMocks.createTextToSpeechRecord.mockReset();
    apiMocks.createTextToSpeechRecordStream.mockReset();
    apiMocks.listSavedVoices.mockReset();
    apiMocks.downloadAudioFile.mockReset();
    hookMocks.useRouteModelSelection.mockReset();

    apiMocks.listTextToSpeechRecords.mockResolvedValue([]);
    apiMocks.getTextToSpeechRecord.mockResolvedValue(buildRecord());
    apiMocks.deleteTextToSpeechRecord.mockResolvedValue({
      id: "tts-1",
      deleted: true,
    });
    apiMocks.textToSpeechRecordAudioUrl.mockReturnValue("/audio/tts.wav");
    apiMocks.createTextToSpeechRecord.mockResolvedValue(buildRecord());
    apiMocks.listSavedVoices.mockResolvedValue([
      buildSavedVoice("voice-1", "Narrator"),
    ]);
    apiMocks.downloadAudioFile.mockResolvedValue(undefined);
    apiMocks.createTextToSpeechRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onCreated?.(buildRecord());
        return new AbortController();
      },
    );

    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-Chat",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Chat",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: true,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });
  });

  it("renders the text-to-speech history table on /text-to-speech", async () => {
    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    expect(
      await screen.findByRole("heading", { name: "Text to Speech" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /New generation/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Models/i })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "No text-to-speech jobs yet" }),
    ).toBeInTheDocument();
  });

  it("hides status column and uses saved voice names in history rows", async () => {
    apiMocks.listTextToSpeechRecords.mockResolvedValue([
      buildSummary({
        saved_voice_id: "voice-1",
        speaker: null,
      }),
    ]);
    apiMocks.listSavedVoices.mockResolvedValue([
      buildSavedVoice("voice-1", "Narrator Prime"),
    ]);

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    expect(
      screen.queryByRole("columnheader", { name: /Status/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Narrator Prime")).toBeInTheDocument();
    expect(screen.queryByText("voice-1")).not.toBeInTheDocument();
  });

  it("uses built-in speaker display names in history rows", async () => {
    apiMocks.listTextToSpeechRecords.mockResolvedValue([
      buildSummary({
        saved_voice_id: null,
        speaker: "Ono_anna",
        model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
      }),
    ]);

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    expect(screen.getByText("Anna")).toBeInTheDocument();
    expect(screen.queryByText("Ono_anna")).not.toBeInTheDocument();
  });

  it("opens the new text-to-speech modal from the header action", async () => {
    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Enter text for generation")).toBeInTheDocument();
    expect(screen.getByText("Review settings")).toBeInTheDocument();
    expect(screen.getByText("Built-in voice")).toBeInTheDocument();
    expect(screen.getByText("Vivian")).toBeInTheDocument();
    expect(screen.queryByText("Saved voice")).not.toBeInTheDocument();
    expect(screen.queryByText("Voice direction")).not.toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText("Optional style guidance"),
    ).not.toBeInTheDocument();
  });

  it("resets modal state after close and reopen", async () => {
    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));
    expect(screen.getByText("Enter text to generate speech.")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Write the text to speak..."), {
      target: { value: "Temporary draft text" },
    });

    const streamToggle = screen.getByRole("checkbox");
    expect(streamToggle).toBeChecked();
    fireEvent.click(streamToggle);
    expect(streamToggle).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: /^Close$/i }));

    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: "New text-to-speech job" }),
      ).not.toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();

    expect(screen.getByPlaceholderText("Write the text to speak...")).toHaveValue("");
    expect(screen.getByRole("checkbox")).toBeChecked();
    expect(
      screen.queryByText("Enter text to generate speech."),
    ).not.toBeInTheDocument();
  });

  it("shows only saved voice controls for clone-capable models", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-Base",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Base",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: false,
          supports_reference_voice: true,
          supports_voice_description: false,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Saved voice")).toBeInTheDocument();
    expect(screen.getByText("Select saved voice")).toBeInTheDocument();
    expect(screen.queryByText("Built-in voice")).not.toBeInTheDocument();
    expect(screen.queryByText("Voice direction")).not.toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText("Optional style guidance"),
    ).not.toBeInTheDocument();
  });

  it("shows kokoro built-in voices with display names in the modal", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Kokoro-82M",
      selectedModelInfo: {
        variant: "Kokoro-82M",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: false,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech?speaker=bf_alice");

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();

    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.queryByText("bf_alice")).not.toBeInTheDocument();
  });

  it("navigates to /text-to-speech/:id after stream created event", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        id: "tts-created-1",
        processing_status: "processing",
      }),
    );
    apiMocks.createTextToSpeechRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onCreated?.(
          buildRecord({
            id: "tts-created-1",
            processing_status: "pending",
          }),
        );
        return new AbortController();
      },
    );

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    fireEvent.change(screen.getByPlaceholderText("Write the text to speak..."), {
      target: { value: "Hello from modal" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));

    await waitFor(() =>
      expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-created-1"),
    );

    expect(
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();
  });

  it("navigates to /text-to-speech/:id when stream emits final without created", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        id: "tts-created-2",
        processing_status: "processing",
      }),
    );
    apiMocks.createTextToSpeechRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onStart?.({
          requestId: "req-1",
          sampleRate: 24000,
          audioFormat: "pcm_i16",
        });
        callbacks.onFinal?.({
          record: buildRecord({
            id: "tts-created-2",
            processing_status: "processing",
          }),
          stats: {
            generation_time_ms: 0,
            audio_duration_secs: 0,
            rtf: 0,
            tokens_generated: 0,
          },
        });
        callbacks.onDone?.();
        return new AbortController();
      },
    );

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    fireEvent.change(screen.getByPlaceholderText("Write the text to speak..."), {
      target: { value: "Hello from final event" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));

    await waitFor(() =>
      expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-created-2"),
    );

    expect(
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();
  });

  it("deletes from the record detail page and navigates back to history", async () => {
    apiMocks.listTextToSpeechRecords.mockResolvedValue([buildSummary()]);
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        processing_status: "ready",
        generation_time_ms: 120,
        audio_duration_secs: 2.5,
        rtf: 0.4,
        tokens_generated: 120,
      }),
    );

    renderRoute("/text-to-speech/tts-1");

    expect(
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^Delete$/i }));
    fireEvent.click(screen.getByRole("button", { name: /Delete generation/i }));

    await waitFor(() =>
      expect(apiMocks.deleteTextToSpeechRecord).toHaveBeenCalledWith("tts-1"),
    );

    expect(
      await screen.findByRole("heading", { name: "Text to Speech" }),
    ).toBeInTheDocument();
  });

  it("uses saved voice names on detail headers and removes header status badges", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        saved_voice_id: "voice-1",
        speaker: null,
        processing_status: "ready",
        generation_time_ms: 120,
        audio_duration_secs: 2.5,
        rtf: 0.4,
        tokens_generated: 120,
      }),
    );
    apiMocks.listSavedVoices.mockResolvedValue([
      buildSavedVoice("voice-1", "Narrator Prime"),
    ]);

    renderRoute("/text-to-speech/tts-1");

    expect(
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();

    expect(await screen.findByText("Voice: Narrator Prime")).toBeInTheDocument();
    expect(screen.queryByText(/^READY$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Saved voice:\s*voice-1/i)).not.toBeInTheDocument();
  });

  it("uses built-in speaker display names on detail headers", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        saved_voice_id: null,
        speaker: "Ono_anna",
        model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
        processing_status: "ready",
      }),
    );

    renderRoute("/text-to-speech/tts-1");

    expect(
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();

    expect(await screen.findByText("Voice: Anna")).toBeInTheDocument();
    expect(screen.queryByText("Voice: Ono_anna")).not.toBeInTheDocument();
  });
});
