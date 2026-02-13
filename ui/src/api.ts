const tauriServerUrl =
  typeof window !== "undefined"
    ? ((window as { __IZWI_SERVER_URL__?: string }).__IZWI_SERVER_URL__ ?? null)
    : null;
const API_BASE = tauriServerUrl
  ? `${tauriServerUrl.replace(/\/$/, "")}/v1`
  : "/v1";

// ============================================================================
// Types
// ============================================================================

export interface ModelInfo {
  variant: string;
  status:
    | "not_downloaded"
    | "downloading"
    | "downloaded"
    | "loading"
    | "ready"
    | "error";
  local_path: string | null;
  size_bytes: number | null;
  download_progress: number | null;
  error_message: string | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

// ============================================================================
// Chat Types
// ============================================================================

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionRequest {
  model_id?: string;
  messages: ChatMessage[];
  max_tokens?: number;
}

export interface ChatCompletionResponse {
  model_id: string;
  message: ChatMessage;
  stats: {
    tokens_generated: number;
    generation_time_ms: number;
  };
}

export type ChatStreamEvent =
  | { event: "start"; model_id: string }
  | { event: "delta"; delta: string }
  | {
      event: "done";
      model_id: string;
      message: string;
      stats: {
        tokens_generated: number;
        generation_time_ms: number;
      };
    }
  | { event: "error"; error: string };

export interface ChatStreamCallbacks {
  onStart?: (modelId: string) => void;
  onDelta?: (delta: string) => void;
  onDone?: (
    message: string,
    stats: { tokens_generated: number; generation_time_ms: number },
  ) => void;
  onError?: (error: string) => void;
}

// ============================================================================
// Unified TTS Types
// ============================================================================

export interface TTSRequest {
  text: string;
  model_id: string;
  language?: string;
  speaker?: string;
  voice_description?: string;
  reference_audio?: string;
  reference_text?: string;
  max_tokens?: number;
  format?: "wav" | "pcm" | "raw_f32" | "raw_i16";
  temperature?: number;
  speed?: number;
}

export interface TTSGenerationStats {
  generation_time_ms: number;
  audio_duration_secs: number;
  rtf: number;
  tokens_generated: number;
}

export interface TTSGenerateResult {
  audioBlob: Blob;
  stats: TTSGenerationStats | null;
}

export type TTSStreamEvent =
  | {
      event: "start";
      request_id: string;
      sample_rate: number;
      audio_format: "wav" | "pcm_i16" | "pcm_f32";
    }
  | {
      event: "chunk";
      request_id: string;
      sequence: number;
      audio_base64: string;
      sample_count: number;
      is_final: boolean;
    }
  | {
      event: "final";
      request_id: string;
      tokens_generated: number;
      generation_time_ms: number;
      audio_duration_secs: number;
      rtf: number;
    }
  | { event: "error"; request_id?: string; error: string }
  | { event: "done"; request_id?: string };

export interface TTSStreamCallbacks {
  onStart?: (event: {
    requestId: string;
    sampleRate: number;
    audioFormat: "wav" | "pcm_i16" | "pcm_f32";
  }) => void;
  onChunk?: (event: {
    requestId: string;
    sequence: number;
    audioBase64: string;
    sampleCount: number;
    isFinal: boolean;
  }) => void;
  onFinal?: (stats: TTSGenerationStats) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

// ============================================================================
// Unified STT (ASR) Types
// ============================================================================

export interface STTRequest {
  audio_base64: string;
  model_id?: string;
  language?: string;
}

export interface STTResponse {
  transcription: string;
  language: string | null;
}

export interface ASRTranscribeRequest {
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
  model_id?: string;
  language?: string;
}

export interface ASRTranscribeResponse {
  transcription: string;
  language: string | null;
  stats?: {
    processing_time_ms: number;
    audio_duration_secs: number | null;
    rtf: number | null;
  };
}

export type ASRStreamEvent =
  | { event: "start"; audio_duration_secs: number | null }
  | { event: "delta"; delta: string }
  | { event: "partial"; text: string; is_final: boolean }
  | {
      event: "final";
      text: string;
      language: string | null;
      audio_duration_secs: number | null;
    }
  | { event: "error"; error: string }
  | { event: "done" };

export interface ASRStreamCallbacks {
  onStart?: (audioDuration: number | null) => void;
  onDelta?: (delta: string) => void;
  onPartial?: (text: string) => void;
  onFinal?: (
    text: string,
    language: string | null,
    audioDuration: number | null,
  ) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

export interface ASRStatusResponse {
  running: boolean;
  status: string;
  device: string | null;
  cached_models: string[];
}

interface OpenAiChatCompletion {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: "assistant" | "system" | "user";
      content: string;
    };
    finish_reason: string;
  }>;
  usage?: {
    completion_tokens?: number;
  };
  izwi_generation_time_ms?: number;
}

interface OpenAiChatChunk {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant" | "system" | "user";
      content?: string;
    };
    finish_reason: string | null;
  }>;
}

type AsrResponseFormat = "json" | "verbose_json";

class ApiClient {
  readonly baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Request failed" } }));
      throw new Error(error.error?.message || "Request failed");
    }

    return response.json();
  }

  // ========================================================================
  // Admin Model Management
  // ========================================================================

  async listModels(): Promise<ModelsResponse> {
    return this.request("/admin/models");
  }

  async getModelInfo(variant: string): Promise<ModelInfo> {
    return this.request(`/admin/models/${variant}`);
  }

  async downloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/download`, { method: "POST" });
  }

  async loadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/load`, { method: "POST" });
  }

  async unloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/unload`, { method: "POST" });
  }

  async deleteModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}`, { method: "DELETE" });
  }

  async cancelDownload(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/download/cancel`, {
      method: "POST",
    });
  }

  // ========================================================================
  // OpenAI-compatible TTS API
  // ========================================================================

  async generateTTS(request: TTSRequest): Promise<Blob> {
    const result = await this.generateTTSWithStats(request);
    return result.audioBlob;
  }

  async generateTTSWithStats(request: TTSRequest): Promise<TTSGenerateResult> {
    const response = await fetch(`${this.baseUrl}/audio/speech`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: request.model_id,
        input: request.text,
        language: request.language,
        voice: request.speaker,
        instructions: request.voice_description,
        reference_audio: request.reference_audio,
        reference_text: request.reference_text,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        speed: request.speed,
        response_format: request.format ?? "wav",
      }),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "TTS generation failed" } }));
      throw new Error(error.error?.message || "TTS generation failed");
    }

    const generationTimeMs = response.headers.get("X-Generation-Time-Ms");
    const audioDurationSecs = response.headers.get("X-Audio-Duration-Secs");
    const rtf = response.headers.get("X-RTF");
    const tokensGenerated = response.headers.get("X-Tokens-Generated");

    const stats: TTSGenerationStats | null =
      generationTimeMs && audioDurationSecs && rtf && tokensGenerated
        ? {
            generation_time_ms: parseFloat(generationTimeMs),
            audio_duration_secs: parseFloat(audioDurationSecs),
            rtf: parseFloat(rtf),
            tokens_generated: parseInt(tokensGenerated, 10),
          }
        : null;

    const audioBlob = await response.blob();
    return { audioBlob, stats };
  }

  generateTTSStream(
    request: TTSRequest,
    callbacks: TTSStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/audio/speech`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: request.model_id,
            input: request.text,
            language: request.language,
            voice: request.speaker,
            instructions: request.voice_description,
            reference_audio: request.reference_audio,
            reference_text: request.reference_text,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            speed: request.speed,
            response_format: request.format ?? "pcm",
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "TTS streaming failed" } }));
          callbacks.onError?.(error.error?.message || "TTS streaming failed");
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            try {
              const event = JSON.parse(data) as TTSStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.({
                    requestId: event.request_id,
                    sampleRate: event.sample_rate,
                    audioFormat: event.audio_format,
                  });
                  break;
                case "chunk":
                  callbacks.onChunk?.({
                    requestId: event.request_id,
                    sequence: event.sequence,
                    audioBase64: event.audio_base64,
                    sampleCount: event.sample_count,
                    isFinal: event.is_final,
                  });
                  break;
                case "final":
                  callbacks.onFinal?.({
                    generation_time_ms: event.generation_time_ms,
                    audio_duration_secs: event.audio_duration_secs,
                    rtf: event.rtf,
                    tokens_generated: event.tokens_generated,
                  });
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
                case "done":
                  callbacks.onDone?.();
                  return;
              }
            } catch {
              // Skip malformed payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "TTS stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  // ========================================================================
  // OpenAI-compatible ASR API
  // ========================================================================

  async asrStatus(): Promise<ASRStatusResponse> {
    // Legacy method retained for UI compatibility.
    return {
      running: false,
      status: "unknown",
      device: null,
      cached_models: [],
    };
  }

  async asrTranscribe(
    request: ASRTranscribeRequest,
  ): Promise<ASRTranscribeResponse> {
    const response = await fetch(
      `${this.baseUrl}/audio/transcriptions`,
      this.buildAsrRequestInit(request, "verbose_json", false),
    );

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Transcription failed" } }));
      throw new Error(error.error?.message || "Transcription failed");
    }

    const payload = await response.json();
    const transcription = payload.text ?? "";

    return {
      transcription,
      language: payload.language ?? null,
      stats:
        typeof payload.processing_time_ms === "number"
          ? {
              processing_time_ms: payload.processing_time_ms,
              audio_duration_secs:
                typeof payload.duration === "number" ? payload.duration : null,
              rtf: typeof payload.rtf === "number" ? payload.rtf : null,
            }
          : undefined,
    };
  }

  asrTranscribeStream(
    request: ASRTranscribeRequest,
    callbacks: ASRStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/audio/transcriptions`, {
          ...this.buildAsrRequestInit(request, "json", true),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response.json().catch(() => ({
            error: { message: "Streaming transcription failed" },
          }));
          callbacks.onError?.(
            error.error?.message || "Streaming transcription failed",
          );
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";
        let assembledText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            if (data === "[DONE]") {
              callbacks.onDone?.();
              return;
            }

            try {
              const event = JSON.parse(data) as ASRStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.(event.audio_duration_secs);
                  break;
                case "delta":
                  assembledText += event.delta;
                  callbacks.onDelta?.(event.delta);
                  callbacks.onPartial?.(assembledText);
                  break;
                case "partial":
                  if (event.text.startsWith(assembledText)) {
                    const delta = event.text.slice(assembledText.length);
                    if (delta) callbacks.onDelta?.(delta);
                  } else if (event.text !== assembledText) {
                    callbacks.onDelta?.(event.text);
                  }
                  assembledText = event.text;
                  callbacks.onPartial?.(event.text);
                  break;
                case "final":
                  assembledText = event.text;
                  callbacks.onFinal?.(
                    event.text,
                    event.language,
                    event.audio_duration_secs,
                  );
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
                case "done":
                  callbacks.onDone?.();
                  return;
              }
            } catch {
              // Skip malformed payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  private buildAsrRequestInit(
    request: ASRTranscribeRequest,
    responseFormat: AsrResponseFormat,
    stream: boolean,
  ): RequestInit {
    if (request.audio_file) {
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      if (request.model_id) form.append("model", request.model_id);
      if (request.language) form.append("language", request.language);
      form.append("response_format", responseFormat);
      if (stream) form.append("stream", "true");
      return {
        method: "POST",
        body: form,
      };
    }

    if (!request.audio_base64) {
      throw new Error("Missing audio input");
    }

    return {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio_base64: request.audio_base64,
        model: request.model_id,
        language: request.language,
        response_format: responseFormat,
        stream,
      }),
    };
  }

  // ========================================================================
  // OpenAI-compatible Chat API
  // ========================================================================

  async chatCompletions(
    request: ChatCompletionRequest,
  ): Promise<ChatCompletionResponse> {
    const response = await this.request<OpenAiChatCompletion>(
      "/chat/completions",
      {
        method: "POST",
        body: JSON.stringify({
          model: request.model_id ?? "Qwen3-0.6B-4bit",
          messages: request.messages,
          max_tokens: request.max_tokens,
          stream: false,
        }),
      },
    );

    const firstChoice = response.choices[0];
    if (!firstChoice) {
      throw new Error("Missing assistant response");
    }

    return {
      model_id: response.model,
      message: {
        role: firstChoice.message.role,
        content: firstChoice.message.content,
      },
      stats: {
        tokens_generated: response.usage?.completion_tokens ?? 0,
        generation_time_ms: response.izwi_generation_time_ms ?? 0,
      },
    };
  }

  chatCompletionsStream(
    request: ChatCompletionRequest,
    callbacks: ChatStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: request.model_id ?? "Qwen3-0.6B-4bit",
            messages: request.messages,
            max_tokens: request.max_tokens,
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "Chat streaming failed" } }));
          callbacks.onError?.(error.error?.message || "Chat streaming failed");
          return;
        }

        callbacks.onStart?.(request.model_id ?? "Qwen3-0.6B-4bit");

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";
        let fullText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            if (data === "[DONE]") {
              callbacks.onDone?.(fullText, {
                tokens_generated: Math.max(1, Math.floor(fullText.length / 4)),
                generation_time_ms: 0,
              });
              return;
            }

            try {
              const payload = JSON.parse(data) as OpenAiChatChunk;
              const choice = payload.choices?.[0];
              const delta = choice?.delta?.content;
              if (delta) {
                fullText += delta;
                callbacks.onDelta?.(delta);
              }
            } catch {
              // Skip malformed SSE payloads.
            }
          }
        }
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Chat stream error",
          );
        }
      }
    };

    startStream();
    return abortController;
  }

  // ========================================================================
  // Convenience aliases
  // ========================================================================

  async synthesize(request: TTSRequest): Promise<Blob> {
    return this.generateTTS(request);
  }

  async transcribe(request: STTRequest): Promise<STTResponse> {
    const result = await this.asrTranscribe({
      audio_base64: request.audio_base64,
      model_id: request.model_id,
      language: request.language,
    });

    return {
      transcription: result.transcription,
      language: result.language,
    };
  }
}

export const api = new ApiClient();
