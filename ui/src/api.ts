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
// Unified TTS Types
// ============================================================================

export interface TTSRequest {
  text: string;
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

// ============================================================================
// Legacy Types (for backward compatibility)
// ============================================================================

export interface LFM2TTSRequest {
  text: string;
  voice?: "us_male" | "us_female" | "uk_male" | "uk_female";
  max_new_tokens?: number;
  audio_temperature?: number;
  audio_top_k?: number;
}

export interface LFM2ASRRequest {
  audio_base64: string;
  max_new_tokens?: number;
}

export interface LFM2AudioChatRequest {
  audio_base64?: string;
  text?: string;
  max_new_tokens?: number;
  audio_temperature?: number;
  audio_top_k?: number;
}

export interface LFM2TTSResponse {
  audio_base64: string;
  sample_rate: number;
  format: string;
}

export interface LFM2ASRResponse {
  transcription: string;
}

export interface LFM2AudioChatResponse {
  text: string;
  audio_base64: string | null;
  sample_rate: number;
  format: string;
}

export interface ASRTranscribeRequest {
  audio_base64: string;
  model_id?: string;
  language?: string;
}

export interface ASRTranscribeResponse {
  transcription: string;
  language: string | null;
}

export interface ASRStatusResponse {
  running: boolean;
  status: string;
  device: string | null;
  cached_models: string[];
}

class ApiClient {
  private baseUrl: string;

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

    return response.blob();
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

  async lfm2GenerateTTS(request: LFM2TTSRequest): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/lfm2/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "LFM2 TTS generation failed" } }));
      throw new Error(error.error?.message || "LFM2 TTS generation failed");
    }

    const data: LFM2TTSResponse = await response.json();
    const binaryString = atob(data.audio_base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type: "audio/wav" });
  }

  async lfm2TranscribeAudio(request: LFM2ASRRequest): Promise<LFM2ASRResponse> {
    return this.request("/lfm2/asr", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async lfm2AudioChat(
    request: LFM2AudioChatRequest,
  ): Promise<LFM2AudioChatResponse> {
    return this.request("/lfm2/chat", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async lfm2Status(): Promise<{
    running: boolean;
    status: string;
    device: string | null;
    cached_models: string[];
    voices: string[];
  }> {
    return this.request("/lfm2/status");
  }

  async lfm2StartDaemon(): Promise<{ success: boolean; message: string }> {
    return this.request("/lfm2/start", { method: "POST" });
  }

  async lfm2StopDaemon(): Promise<{ success: boolean; message: string }> {
    return this.request("/lfm2/stop", { method: "POST" });
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
}

export const api = new ApiClient();
