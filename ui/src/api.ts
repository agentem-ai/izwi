const API_BASE = "/api/v1";

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
  model_id?: string;
  speaker?: string;
  voice_description?: string;
  reference_audio?: string;
  reference_text?: string;
  format?: "wav" | "raw_f32" | "raw_i16";
  temperature?: number;
  speed?: number;
}

export interface TTSResponse {
  request_id: string;
  audio: string;
  format: string;
  sample_rate: number;
  duration_secs: number;
  stats: {
    tokens_generated: number;
    generation_time_ms: number;
    rtf: number;
  };
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
  audio_base64: string;
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

// Streaming transcription event types
export type ASRStreamEvent =
  | { event: "start"; audio_duration_secs: number | null }
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
  onPartial?: (text: string) => void;
  onFinal?: (
    text: string,
    language: string | null,
    audioDuration: number | null,
  ) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
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

export interface ASRStatusResponse {
  running: boolean;
  status: string;
  device: string | null;
  cached_models: string[];
}

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

  async listModels(): Promise<ModelsResponse> {
    return this.request("/models");
  }

  async getModelInfo(variant: string): Promise<ModelInfo> {
    return this.request(`/models/${variant}`);
  }

  async downloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}/download`, { method: "POST" });
  }

  async loadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}/load`, { method: "POST" });
  }

  async unloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}/unload`, { method: "POST" });
  }

  async generateTTS(request: TTSRequest): Promise<Blob> {
    const result = await this.generateTTSWithStats(request);
    return result.audioBlob;
  }

  async generateTTSWithStats(request: TTSRequest): Promise<TTSGenerateResult> {
    const response = await fetch(`${this.baseUrl}/tts/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ...request, format: "wav" }),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "TTS generation failed" } }));
      throw new Error(error.error?.message || "TTS generation failed");
    }

    // Extract timing stats from headers
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

  async generateTTSStream(request: TTSRequest): Promise<Response> {
    const response = await fetch(`${this.baseUrl}/tts/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "TTS streaming failed" } }));
      throw new Error(error.error?.message || "TTS streaming failed");
    }

    return response;
  }

  async asrStatus(): Promise<ASRStatusResponse> {
    return this.request("/asr/status");
  }

  async asrStartDaemon(): Promise<{ success: boolean; message: string }> {
    return this.request("/asr/start", { method: "POST" });
  }

  async asrStopDaemon(): Promise<{ success: boolean; message: string }> {
    return this.request("/asr/stop", { method: "POST" });
  }

  async asrTranscribe(
    request: ASRTranscribeRequest,
  ): Promise<ASRTranscribeResponse> {
    return this.request("/asr/transcribe", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async chatCompletions(
    request: ChatCompletionRequest,
  ): Promise<ChatCompletionResponse> {
    return this.request("/chat/completions", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  chatCompletionsStream(
    request: ChatCompletionRequest,
    callbacks: ChatStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/chat/completions/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(request),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "Chat streaming failed" } }));
          callbacks.onError?.(error.error?.message || "Chat streaming failed");
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
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
              const event = JSON.parse(data) as ChatStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.(event.model_id);
                  break;
                case "delta":
                  callbacks.onDelta?.(event.delta);
                  break;
                case "done":
                  callbacks.onDone?.(event.message, event.stats);
                  return;
                case "error":
                  callbacks.onError?.(event.error);
                  return;
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

  /**
   * Stream transcription with SSE - sends partial results as text is decoded.
   * Returns an AbortController that can be used to cancel the stream.
   */
  asrTranscribeStream(
    request: ASRTranscribeRequest,
    callbacks: ASRStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/asr/transcribe/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(request),
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

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events from buffer
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith("data:")) {
              const data = line.slice(5).trim();
              if (data) {
                try {
                  const event = JSON.parse(data) as ASRStreamEvent;

                  switch (event.event) {
                    case "start":
                      callbacks.onStart?.(event.audio_duration_secs);
                      break;
                    case "partial":
                      callbacks.onPartial?.(event.text);
                      break;
                    case "final":
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
                  // Skip malformed JSON
                }
              }
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

  // ==========================================================================
  // Unified TTS API
  // ==========================================================================

  /**
   * Unified TTS endpoint that works with any TTS model.
   * Automatically routes to the correct backend based on input.
   */
  async synthesize(request: TTSRequest): Promise<Blob> {
    return this.generateTTS(request);
  }

  // ==========================================================================
  // Unified STT API
  // ==========================================================================

  /**
   * Unified STT endpoint that works with any ASR model.
   * Automatically routes to the correct backend based on model.
   */
  async transcribe(request: STTRequest): Promise<STTResponse> {
    return this.asrTranscribe({
      audio_base64: request.audio_base64,
      model_id: request.model_id,
      language: request.language,
    });
  }

  // ==========================================================================
  // Delete Model
  // ==========================================================================

  async deleteModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}`, { method: "DELETE" });
  }

  async cancelDownload(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}/download/cancel`, {
      method: "POST",
    });
  }
}

export const api = new ApiClient();
