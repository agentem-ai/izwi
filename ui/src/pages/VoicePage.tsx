import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  MicOff,
  Volume2,
  Loader2,
  PhoneOff,
  AudioLines,
  Settings2,
  Download,
  Play,
  Square,
  Trash2,
  X,
} from "lucide-react";
import clsx from "clsx";

import { api, ModelInfo } from "../api";
import {
  getSpeakerProfilesForVariant,
  isKokoroVariant,
  isLfmAudioVariant,
  VIEW_CONFIGS,
} from "../types";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { Slider } from "../components/ui/slider";
import { Button } from "../components/ui/button";
import { cn } from "@/lib/utils";

type RuntimeStatus =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "assistant_speaking";
type PipelineMode = "s2s" | "stt_chat_tts";

type VoiceRealtimeServerEvent =
  | { type: "connected"; protocol: string; server_time_ms?: number }
  | { type: "session_ready"; protocol: string }
  | {
      type: "input_stream_ready";
      vad?: {
        threshold?: number;
        min_speech_ms?: number;
        silence_duration_ms?: number;
      };
    }
  | { type: "input_stream_stopped" }
  | { type: "listening"; utterance_id: string; utterance_seq: number }
  | { type: "user_speech_start"; utterance_id: string; utterance_seq: number }
  | {
      type: "user_speech_end";
      utterance_id: string;
      utterance_seq: number;
      reason?: "silence" | "max_duration" | "stream_stopped";
    }
  | { type: "turn_processing"; utterance_id: string; utterance_seq: number }
  | {
      type: "user_transcript_start";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "user_transcript_delta";
      utterance_id: string;
      utterance_seq: number;
      delta: string;
    }
  | {
      type: "user_transcript_final";
      utterance_id: string;
      utterance_seq: number;
      text: string;
      language?: string | null;
      audio_duration_secs?: number;
    }
  | {
      type: "assistant_text_start";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "assistant_text_final";
      utterance_id: string;
      utterance_seq: number;
      text: string;
      raw_text?: string;
    }
  | {
      type: "assistant_audio_start";
      utterance_id: string;
      utterance_seq: number;
      sample_rate: number;
      audio_format: "pcm_i16" | "pcm_f32" | "wav";
    }
  | {
      type: "assistant_audio_done";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "turn_done";
      utterance_id: string;
      utterance_seq: number;
      status: "ok" | "error" | "timeout" | "interrupted" | "no_input";
      reason?: string;
    }
  | {
      type: "error";
      utterance_id?: string | null;
      utterance_seq?: number | null;
      message: string;
    }
  | { type: "pong"; timestamp_ms?: number; server_time_ms?: number };

type VoiceRealtimeClientMessage =
  | { type: "session_start"; system_prompt?: string }
  | {
      type: "input_stream_start";
      asr_model_id: string;
      text_model_id: string;
      tts_model_id: string;
      speaker?: string;
      asr_language?: string;
      max_output_tokens?: number;
      vad_threshold?: number;
      min_speech_ms?: number;
      silence_duration_ms?: number;
      max_utterance_ms?: number;
      pre_roll_ms?: number;
      input_sample_rate?: number;
    }
  | { type: "input_stream_stop" }
  | { type: "interrupt"; reason?: string }
  | { type: "ping"; timestamp_ms?: number };

const VOICE_WS_BIN_MAGIC = "IVWS";
const VOICE_WS_BIN_VERSION = 1;
const VOICE_WS_BIN_KIND_CLIENT_PCM16 = 1;
const VOICE_WS_BIN_KIND_ASSISTANT_PCM16 = 2;
const VOICE_WS_BIN_CLIENT_HEADER_LEN = 16;
const VOICE_WS_BIN_ASSISTANT_HEADER_LEN = 24;

interface VoiceRealtimeAssistantAudioBinaryChunk {
  utteranceSeq: number;
  sequence: number;
  sampleRate: number;
  isFinal: boolean;
  pcm16Bytes: Uint8Array;
}

interface TranscriptEntry {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
}

interface VoicePageProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onError?: (message: string) => void;
}

const VOICE_AGENT_SYSTEM_PROMPT =
  "You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Do not output <think> tags or internal reasoning. Return only the final spoken answer. Keep responses brief unless asked for details.";

const PIPELINE_LABELS: Record<PipelineMode, string> = {
  s2s: "Speech-to-Speech (S2S)",
  stt_chat_tts: "STT -> Chat -> TTS",
};

function parseFinalAnswer(content: string): string {
  const openTag = "<think>";
  const closeTag = "</think>";
  let out = content;

  while (true) {
    const start = out.indexOf(openTag);
    if (start === -1) break;
    const end = out.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      out = out.slice(0, start);
      break;
    }
    out = `${out.slice(0, start)}${out.slice(end + closeTag.length)}`;
  }

  return out.trim();
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function isAsrVariant(variant: string): boolean {
  return (
    variant.includes("Qwen3-ASR") ||
    variant.includes("Parakeet-TDT") ||
    variant.includes("Voxtral") ||
    isLfmAudioVariant(variant)
  );
}

function isLfm2Variant(variant: string): boolean {
  return isLfmAudioVariant(variant);
}

function isTextVariant(variant: string): boolean {
  return VIEW_CONFIGS.chat.modelFilter(variant);
}

function isTtsVariant(variant: string): boolean {
  return (
    (variant.includes("Qwen3-TTS") && !variant.includes("Tokenizer")) ||
    isKokoroVariant(variant)
  );
}

function isCustomVoiceTtsVariant(variant: string): boolean {
  return (
    (isTtsVariant(variant) && variant.includes("CustomVoice")) ||
    isKokoroVariant(variant)
  );
}

function formatModelVariantLabel(variant: string): string {
  const normalized = variant
    .replace(/-4bit\b/g, "-4-bit")
    .replace(/-8bit\b/g, "-8-bit");

  if (normalized.startsWith("Qwen3-ASR-")) {
    return normalized.replace("Qwen3-ASR-", "ASR ");
  }

  if (normalized.startsWith("Parakeet-TDT-")) {
    return normalized.replace("Parakeet-TDT-", "Parakeet ");
  }

  if (normalized.startsWith("Qwen3-TTS-12Hz-")) {
    return normalized.replace("Qwen3-TTS-12Hz-", "TTS ");
  }

  if (normalized.startsWith("Qwen3-ForcedAligner-")) {
    return normalized.replace("Qwen3-ForcedAligner-", "ForcedAligner ");
  }

  if (normalized.startsWith("Qwen3-")) {
    return normalized.replace("Qwen3-", "Qwen3 ");
  }

  if (normalized.startsWith("Gemma-3-")) {
    return normalized
      .replace("Gemma-3-1b-it", "Gemma 3 1B Instruct")
      .replace("Gemma-3-4b-it", "Gemma 3 4B Instruct");
  }

  if (isLfmAudioVariant(normalized)) {
    return normalized
      .replace("LFM2.5-Audio-", "LFM2.5 Audio ")
      .replace("LFM2-Audio-", "LFM2 Audio ");
  }

  if (isKokoroVariant(normalized)) {
    return "Kokoro 82M";
  }

  return normalized.replace(/-/g, " ");
}

function isRunnableModelStatus(status: ModelInfo["status"]): boolean {
  return status === "ready";
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function decodePcmI16Base64(base64Data: string): Float32Array {
  const binary = atob(base64Data);
  const sampleCount = Math.floor(binary.length / 2);
  const out = new Float32Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const lo = binary.charCodeAt(i * 2);
    const hi = binary.charCodeAt(i * 2 + 1);
    let value = (hi << 8) | lo;
    if (value & 0x8000) {
      value -= 0x10000;
    }
    out[i] = value / 0x8000;
  }

  return out;
}

function mergeSampleChunks(chunks: Float32Array[]): Float32Array {
  const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeFloat32ToPcm16Bytes(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(i * 2, int16, true);
  }
  return out;
}

function encodeVoiceRealtimeClientPcm16Frame(
  pcm16Bytes: Uint8Array,
  sampleRate: number,
  frameSeq: number,
): Uint8Array {
  const frame = new Uint8Array(
    VOICE_WS_BIN_CLIENT_HEADER_LEN + pcm16Bytes.length,
  );
  frame[0] = VOICE_WS_BIN_MAGIC.charCodeAt(0);
  frame[1] = VOICE_WS_BIN_MAGIC.charCodeAt(1);
  frame[2] = VOICE_WS_BIN_MAGIC.charCodeAt(2);
  frame[3] = VOICE_WS_BIN_MAGIC.charCodeAt(3);
  frame[4] = VOICE_WS_BIN_VERSION;
  frame[5] = VOICE_WS_BIN_KIND_CLIENT_PCM16;
  frame[6] = 0;
  frame[7] = 0;
  const view = new DataView(frame.buffer);
  view.setUint32(8, sampleRate >>> 0, true);
  view.setUint32(12, frameSeq >>> 0, true);
  frame.set(pcm16Bytes, VOICE_WS_BIN_CLIENT_HEADER_LEN);
  return frame;
}

function parseVoiceRealtimeAssistantAudioBinaryChunk(
  data: ArrayBuffer,
): VoiceRealtimeAssistantAudioBinaryChunk | null {
  if (data.byteLength < VOICE_WS_BIN_ASSISTANT_HEADER_LEN) {
    return null;
  }
  const bytes = new Uint8Array(data);
  if (
    String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !==
    VOICE_WS_BIN_MAGIC
  ) {
    return null;
  }
  const view = new DataView(data);
  const version = view.getUint8(4);
  const kind = view.getUint8(5);
  if (
    version !== VOICE_WS_BIN_VERSION ||
    kind !== VOICE_WS_BIN_KIND_ASSISTANT_PCM16
  ) {
    return null;
  }
  const flags = view.getUint16(6, true);
  const utteranceSeq = Number(view.getBigUint64(8, true));
  const sequence = view.getUint32(16, true);
  const sampleRate = view.getUint32(20, true);
  const pcm16Bytes = bytes.slice(VOICE_WS_BIN_ASSISTANT_HEADER_LEN);
  return {
    utteranceSeq,
    sequence,
    sampleRate,
    isFinal: (flags & 1) === 1,
    pcm16Bytes,
  };
}

function decodePcmI16Bytes(pcm16Bytes: Uint8Array): Float32Array {
  const sampleCount = Math.floor(pcm16Bytes.length / 2);
  const out = new Float32Array(sampleCount);
  const view = new DataView(
    pcm16Bytes.buffer,
    pcm16Bytes.byteOffset,
    pcm16Bytes.byteLength,
  );
  for (let i = 0; i < sampleCount; i += 1) {
    out[i] = view.getInt16(i * 2, true) / 0x8000;
  }
  return out;
}

function buildVoiceRealtimeWebSocketUrl(apiBaseUrl: string): string {
  const base = new URL(apiBaseUrl, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/voice/realtime/ws`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

function isVoiceRealtimeServerEvent(
  value: unknown,
): value is VoiceRealtimeServerEvent {
  return (
    !!value &&
    typeof value === "object" &&
    "type" in value &&
    typeof (value as { type?: unknown }).type === "string"
  );
}

function makeTranscriptEntryId(role: "user" | "assistant"): string {
  return `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function transcodeToWav(
  inputBlob: Blob,
  targetSampleRate = 16000,
): Promise<Blob> {
  if (inputBlob.type === "audio/wav" || inputBlob.type === "audio/x-wav") {
    return inputBlob;
  }

  const decodeContext = new AudioContext();
  try {
    const sourceBytes = await inputBlob.arrayBuffer();
    const decoded = await decodeContext.decodeAudioData(sourceBytes.slice(0));

    const monoBuffer = decodeContext.createBuffer(
      1,
      decoded.length,
      decoded.sampleRate,
    );
    const mono = monoBuffer.getChannelData(0);

    for (let i = 0; i < decoded.length; i += 1) {
      let sum = 0;
      for (let ch = 0; ch < decoded.numberOfChannels; ch += 1) {
        sum += decoded.getChannelData(ch)[i] ?? 0;
      }
      mono[i] = sum / decoded.numberOfChannels;
    }

    const rendered = await (() => {
      if (decoded.sampleRate === targetSampleRate) {
        return Promise.resolve(monoBuffer);
      }

      const targetLength = Math.ceil(
        (monoBuffer.length * targetSampleRate) / monoBuffer.sampleRate,
      );
      const offline = new OfflineAudioContext(
        1,
        targetLength,
        targetSampleRate,
      );
      const source = offline.createBufferSource();
      source.buffer = monoBuffer;
      source.connect(offline.destination);
      source.start(0);
      return offline.startRendering();
    })();

    return encodeWavPcm16(rendered.getChannelData(0), targetSampleRate);
  } finally {
    decodeContext.close().catch(() => {});
  }
}

export function VoicePage({
  models,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onError,
}: VoicePageProps) {
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);

  const [pipelineMode, setPipelineMode] =
    useState<PipelineMode>("stt_chat_tts");
  const [selectedS2sModel, setSelectedS2sModel] = useState<string | null>(null);
  const [selectedAsrModel, setSelectedAsrModel] = useState<string | null>(null);
  const [selectedTextModel, setSelectedTextModel] = useState<string | null>(
    null,
  );
  const [selectedTtsModel, setSelectedTtsModel] = useState<string | null>(null);
  const [selectedSpeaker, setSelectedSpeaker] = useState("Serena");

  const [vadThreshold, setVadThreshold] = useState(0.02);
  const [silenceDurationMs, setSilenceDurationMs] = useState(900);
  const [minSpeechMs, setMinSpeechMs] = useState(300);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [pendingDeleteVariant, setPendingDeleteVariant] = useState<
    string | null
  >(null);

  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamingProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const streamingProcessorSinkRef = useRef<GainNode | null>(null);
  const vadTimerRef = useRef<number | null>(null);
  const speechStartRef = useRef<number | null>(null);
  const silenceMsRef = useRef(0);
  const processingRef = useRef(false);
  const runtimeStatusRef = useRef<RuntimeStatus>("idle");
  const isSessionActiveRef = useRef(false);
  const turnIdRef = useRef(0);
  const agentSessionIdRef = useRef<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const asrStreamAbortRef = useRef<AbortController | null>(null);
  const chatStreamAbortRef = useRef<AbortController | null>(null);
  const ttsStreamAbortRef = useRef<AbortController | null>(null);
  const ttsPlaybackContextRef = useRef<AudioContext | null>(null);
  const ttsPlaybackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const ttsNextPlaybackTimeRef = useRef(0);
  const ttsSampleRateRef = useRef(24000);
  const ttsSamplesRef = useRef<Float32Array[]>([]);
  const ttsStreamSessionRef = useRef(0);
  const voiceWsRef = useRef<WebSocket | null>(null);
  const voiceWsConnectingRef = useRef<Promise<WebSocket> | null>(null);
  const voiceWsSessionReadyRef = useRef(false);
  const voiceWsInputStreamStartedRef = useRef(false);
  const voiceWsInputStreamStartingRef = useRef<Promise<void> | null>(null);
  const voiceWsInputFrameSeqRef = useRef(0);
  const voiceMinAcceptedAssistantSeqRef = useRef(0);
  const voiceUserEntryIdsRef = useRef<Map<string, string>>(new Map());
  const voiceAssistantEntryIdsRef = useRef<Map<string, string>>(new Map());
  const voiceWsPlaybackRef = useRef<{
    utteranceId: string;
    utteranceSeq: number;
    streamSession: number;
    streamDone: boolean;
    playbackStarted: boolean;
  } | null>(null);

  useEffect(() => {
    if (!isConfigOpen) {
      setPendingDeleteVariant(null);
    }
  }, [isConfigOpen]);

  const handleConfigDelete = useCallback(
    (variant: string) => {
      setPendingDeleteVariant(null);
      onDelete(variant);
      if (selectedS2sModel === variant) {
        setSelectedS2sModel(null);
      }
      if (selectedAsrModel === variant) {
        setSelectedAsrModel(null);
      }
      if (selectedTextModel === variant) {
        setSelectedTextModel(null);
      }
      if (selectedTtsModel === variant) {
        setSelectedTtsModel(null);
      }
    },
    [
      onDelete,
      selectedS2sModel,
      selectedAsrModel,
      selectedTextModel,
      selectedTtsModel,
    ],
  );

  const sortedModels = useMemo(() => {
    return [...models]
      .filter((m) => !m.variant.includes("Tokenizer"))
      .sort((a, b) => a.variant.localeCompare(b.variant));
  }, [models]);

  const asrModels = useMemo(
    () => sortedModels.filter((m) => isAsrVariant(m.variant)),
    [sortedModels],
  );
  const s2sModels = useMemo(
    () => asrModels.filter((m) => isLfm2Variant(m.variant)),
    [asrModels],
  );
  const sttAsrModels = useMemo(
    () => asrModels.filter((m) => !isLfm2Variant(m.variant)),
    [asrModels],
  );
  const textModels = useMemo(
    () => sortedModels.filter((m) => isTextVariant(m.variant)),
    [sortedModels],
  );
  const ttsModels = useMemo(
    () => sortedModels.filter((m) => isTtsVariant(m.variant)),
    [sortedModels],
  );
  const ttsConfigModels = useMemo(
    () => ttsModels.filter((m) => isCustomVoiceTtsVariant(m.variant)),
    [ttsModels],
  );
  const assistantSpeakers = useMemo(
    () => getSpeakerProfilesForVariant(selectedTtsModel),
    [selectedTtsModel],
  );
  const voiceRouteModels = useMemo(
    () =>
      sortedModels.filter(
        (m) =>
          isAsrVariant(m.variant) ||
          isTextVariant(m.variant) ||
          isCustomVoiceTtsVariant(m.variant),
      ),
    [sortedModels],
  );

  useEffect(() => {
    if (!assistantSpeakers.some((speaker) => speaker.id === selectedSpeaker)) {
      setSelectedSpeaker(assistantSpeakers[0]?.id ?? "Serena");
    }
  }, [assistantSpeakers, selectedSpeaker]);

  useEffect(() => {
    runtimeStatusRef.current = runtimeStatus;
  }, [runtimeStatus]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, runtimeStatus]);

  useEffect(() => {
    if (pipelineMode === "s2s" && s2sModels.length === 0) {
      setPipelineMode("stt_chat_tts");
    }
  }, [pipelineMode, s2sModels.length]);

  useEffect(() => {
    if (
      !selectedS2sModel ||
      !s2sModels.some((m) => m.variant === selectedS2sModel)
    ) {
      const preferredS2s =
        s2sModels.find(
          (m) => m.variant === "LFM2.5-Audio-1.5B-4bit" && m.status === "ready",
        ) ||
        s2sModels.find(
          (m) => m.variant === "LFM2.5-Audio-1.5B" && m.status === "ready",
        ) ||
        s2sModels.find((m) => m.status === "ready") ||
        s2sModels.find((m) => m.variant === "LFM2.5-Audio-1.5B-4bit") ||
        s2sModels.find((m) => m.variant === "LFM2.5-Audio-1.5B") ||
        s2sModels[0];
      setSelectedS2sModel(preferredS2s?.variant ?? null);
    }
  }, [s2sModels, selectedS2sModel]);

  useEffect(() => {
    if (
      !selectedAsrModel ||
      !sttAsrModels.some((m) => m.variant === selectedAsrModel)
    ) {
      const preferredAsr =
        sttAsrModels.find(
          (m) => m.variant === "Qwen3-ASR-0.6B" && m.status === "ready",
        ) ||
        sttAsrModels.find(
          (m) => m.variant.includes("Qwen3-ASR-0.6B") && m.status === "ready",
        ) ||
        sttAsrModels.find((m) => m.status === "ready") ||
        sttAsrModels.find((m) => m.variant === "Qwen3-ASR-0.6B") ||
        sttAsrModels.find((m) => m.variant.includes("Qwen3-ASR-0.6B")) ||
        sttAsrModels[0];
      setSelectedAsrModel(preferredAsr?.variant ?? null);
    }
  }, [sttAsrModels, selectedAsrModel]);

  useEffect(() => {
    if (
      !selectedTextModel ||
      !textModels.some((m) => m.variant === selectedTextModel)
    ) {
      const preferredText =
        textModels.find(
          (m) => m.variant === "Qwen3-0.6B-4bit" && m.status === "ready",
        ) ||
        textModels.find(
          (m) => m.variant === "Qwen3-0.6B" && m.status === "ready",
        ) ||
        textModels.find(
          (m) => m.variant === "Qwen3-1.7B-4bit" && m.status === "ready",
        ) ||
        textModels.find((m) => m.status === "ready") ||
        textModels.find((m) => m.variant === "Qwen3-1.7B-4bit") ||
        textModels.find((m) => m.variant === "Qwen3-0.6B-4bit") ||
        textModels.find((m) => m.variant === "Qwen3-0.6B") ||
        textModels[0];
      setSelectedTextModel(preferredText?.variant ?? null);
    }
  }, [textModels, selectedTextModel]);

  useEffect(() => {
    if (
      !selectedTtsModel ||
      !ttsConfigModels.some((m) => m.variant === selectedTtsModel)
    ) {
      const preferredTts =
        ttsConfigModels.find(
          (m) =>
            m.variant === "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit" &&
            m.status === "ready",
        ) ||
        ttsConfigModels.find(
          (m) =>
            m.variant === "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit" &&
            m.status === "ready",
        ) ||
        ttsConfigModels.find(
          (m) =>
            m.variant.includes("0.6B") &&
            m.variant.includes("4bit") &&
            m.status === "ready",
        ) ||
        ttsConfigModels.find((m) => m.status === "ready") ||
        ttsConfigModels.find(
          (m) => m.variant === "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
        ) ||
        ttsConfigModels.find(
          (m) => m.variant === "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        ) ||
        ttsConfigModels.find(
          (m) => m.variant.includes("0.6B") && m.variant.includes("4bit"),
        ) ||
        ttsConfigModels[0];
      setSelectedTtsModel(preferredTts?.variant ?? null);
    }
  }, [ttsConfigModels, selectedTtsModel]);

  const selectedS2sInfo = useMemo(
    () => s2sModels.find((m) => m.variant === selectedS2sModel) ?? null,
    [s2sModels, selectedS2sModel],
  );
  const selectedAsrInfo = useMemo(
    () => sttAsrModels.find((m) => m.variant === selectedAsrModel) ?? null,
    [sttAsrModels, selectedAsrModel],
  );
  const selectedTextInfo = useMemo(
    () => textModels.find((m) => m.variant === selectedTextModel) ?? null,
    [textModels, selectedTextModel],
  );
  const selectedTtsInfo = useMemo(
    () => ttsConfigModels.find((m) => m.variant === selectedTtsModel) ?? null,
    [ttsConfigModels, selectedTtsModel],
  );

  const lfm2DirectMode = pipelineMode === "s2s";
  const currentPipelineLabel = PIPELINE_LABELS[pipelineMode];

  const hasRunnableConfig = useMemo(() => {
    if (lfm2DirectMode) {
      return !!selectedS2sInfo && isRunnableModelStatus(selectedS2sInfo.status);
    }

    if (!selectedAsrInfo || !isRunnableModelStatus(selectedAsrInfo.status)) {
      return false;
    }

    return (
      !!selectedTextInfo &&
      !!selectedTtsInfo &&
      isRunnableModelStatus(selectedTextInfo.status) &&
      isRunnableModelStatus(selectedTtsInfo.status)
    );
  }, [
    lfm2DirectMode,
    selectedAsrInfo,
    selectedS2sInfo,
    selectedTextInfo,
    selectedTtsInfo,
  ]);

  const stopTtsStreamingPlayback = useCallback(() => {
    ttsStreamSessionRef.current += 1;

    if (ttsStreamAbortRef.current) {
      ttsStreamAbortRef.current.abort();
      ttsStreamAbortRef.current = null;
    }

    for (const source of ttsPlaybackSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // Ignore already-stopped sources.
      }
    }
    ttsPlaybackSourcesRef.current.clear();

    if (ttsPlaybackContextRef.current) {
      ttsPlaybackContextRef.current.close().catch(() => {});
      ttsPlaybackContextRef.current = null;
    }

    ttsNextPlaybackTimeRef.current = 0;
    ttsSampleRateRef.current = 24000;
    ttsSamplesRef.current = [];
  }, []);

  const clearAudioPlayback = useCallback(() => {
    stopTtsStreamingPlayback();

    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.src = "";
    }

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }, [stopTtsStreamingPlayback]);

  const closeVoiceRealtimeSocket = useCallback((reason?: string) => {
    voiceWsSessionReadyRef.current = false;
    voiceWsInputStreamStartedRef.current = false;
    voiceWsInputFrameSeqRef.current = 0;
    voiceWsInputStreamStartingRef.current = null;
    voiceWsConnectingRef.current = null;

    const socket = voiceWsRef.current;
    voiceWsRef.current = null;
    if (socket) {
      try {
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING
        ) {
          socket.close(1000, reason || "session_closed");
        }
      } catch {
        // Ignore close failures.
      }
    }
  }, []);

  const stopSession = useCallback(() => {
    isSessionActiveRef.current = false;
    turnIdRef.current += 1;
    processingRef.current = false;
    silenceMsRef.current = 0;
    speechStartRef.current = null;
    setRuntimeStatus("idle");
    setAudioLevel(0);

    if (vadTimerRef.current != null) {
      window.clearInterval(vadTimerRef.current);
      vadTimerRef.current = null;
    }

    if (
      voiceWsRef.current &&
      voiceWsRef.current.readyState === WebSocket.OPEN
    ) {
      try {
        voiceWsRef.current.send(
          JSON.stringify({
            type: "input_stream_stop",
          } satisfies VoiceRealtimeClientMessage),
        );
      } catch {
        // Best-effort during shutdown.
      }
    }

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;

    if (streamingProcessorRef.current) {
      try {
        streamingProcessorRef.current.disconnect();
      } catch {
        // Ignore.
      }
      streamingProcessorRef.current.onaudioprocess = null;
      streamingProcessorRef.current = null;
    }
    if (streamingProcessorSinkRef.current) {
      try {
        streamingProcessorSinkRef.current.disconnect();
      } catch {
        // Ignore.
      }
      streamingProcessorSinkRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    if (asrStreamAbortRef.current) {
      asrStreamAbortRef.current.abort();
      asrStreamAbortRef.current = null;
    }

    if (chatStreamAbortRef.current) {
      chatStreamAbortRef.current.abort();
      chatStreamAbortRef.current = null;
    }

    closeVoiceRealtimeSocket("session_stopped");
    voiceWsPlaybackRef.current = null;
    voiceMinAcceptedAssistantSeqRef.current = 0;
    voiceUserEntryIdsRef.current.clear();
    voiceAssistantEntryIdsRef.current.clear();

    analyserRef.current = null;
    clearAudioPlayback();
  }, [clearAudioPlayback, closeVoiceRealtimeSocket]);

  useEffect(() => {
    return () => stopSession();
  }, [stopSession]);

  const appendTranscriptEntry = useCallback((entry: TranscriptEntry) => {
    setTranscript((prev) => [...prev, entry]);
  }, []);

  const setTranscriptEntryText = useCallback(
    (entryId: string, text: string) => {
      setTranscript((prev) => {
        const index = prev.findIndex((entry) => entry.id === entryId);
        if (index === -1) {
          return prev;
        }
        const next = [...prev];
        next[index] = {
          ...next[index],
          text,
        };
        return next;
      });
    },
    [],
  );

  const removeTranscriptEntry = useCallback((entryId: string) => {
    setTranscript((prev) => prev.filter((entry) => entry.id !== entryId));
  }, []);

  const sendVoiceRealtimeJson = useCallback(
    (message: VoiceRealtimeClientMessage) => {
      const socket = voiceWsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        throw new Error("Voice realtime websocket is not connected");
      }
      socket.send(JSON.stringify(message));
    },
    [],
  );

  const sendVoiceRealtimeBinary = useCallback((data: Uint8Array) => {
    const socket = voiceWsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error("Voice realtime websocket is not connected");
    }
    socket.send(data);
  }, []);

  const finalizeVoiceWsPlaybackIfComplete = useCallback(
    (utteranceSeq: number, streamSession: number) => {
      const active = voiceWsPlaybackRef.current;
      if (!active) return;
      if (active.utteranceSeq !== utteranceSeq) return;
      if (active.streamSession !== streamSession) return;
      if (!active.streamDone || ttsPlaybackSourcesRef.current.size > 0) {
        return;
      }

      if (ttsStreamSessionRef.current === streamSession) {
        const merged = mergeSampleChunks(ttsSamplesRef.current);
        if (merged.length > 0) {
          const wavBlob = encodeWavPcm16(merged, ttsSampleRateRef.current);
          const nextUrl = URL.createObjectURL(wavBlob);
          if (audioUrlRef.current) {
            URL.revokeObjectURL(audioUrlRef.current);
          }
          audioUrlRef.current = nextUrl;
        }

        if (ttsPlaybackContextRef.current) {
          ttsPlaybackContextRef.current.close().catch(() => {});
          ttsPlaybackContextRef.current = null;
        }

        ttsPlaybackSourcesRef.current.clear();
        ttsNextPlaybackTimeRef.current = 0;
        ttsSamplesRef.current = [];
        ttsStreamAbortRef.current = null;
      }

      if (
        voiceWsPlaybackRef.current &&
        voiceWsPlaybackRef.current.streamSession === streamSession
      ) {
        voiceWsPlaybackRef.current = null;
      }

      processingRef.current = false;
      if (
        isSessionActiveRef.current &&
        runtimeStatusRef.current !== "user_speaking"
      ) {
        setRuntimeStatus("listening");
      }
    },
    [],
  );

  const handleVoiceRealtimeAssistantAudioBinaryChunk = useCallback(
    (chunk: VoiceRealtimeAssistantAudioBinaryChunk) => {
      const playback = voiceWsPlaybackRef.current;
      if (!playback) return;
      if (playback.utteranceSeq !== chunk.utteranceSeq) {
        return;
      }

      const context = ttsPlaybackContextRef.current;
      if (!context) return;

      if (
        chunk.sampleRate > 0 &&
        ttsSampleRateRef.current !== chunk.sampleRate
      ) {
        ttsSampleRateRef.current = chunk.sampleRate;
      }

      const samples = decodePcmI16Bytes(chunk.pcm16Bytes);
      if (samples.length === 0) return;

      if (!playback.playbackStarted) {
        playback.playbackStarted = true;
        processingRef.current = false;
        if (isSessionActiveRef.current) {
          setRuntimeStatus("assistant_speaking");
        }
      }

      ttsSamplesRef.current.push(samples);

      const buffer = context.createBuffer(
        1,
        samples.length,
        ttsSampleRateRef.current,
      );
      const samplesForPlayback = new Float32Array(samples.length);
      samplesForPlayback.set(samples);
      buffer.copyToChannel(samplesForPlayback, 0);

      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);

      const scheduledAt = Math.max(
        context.currentTime + 0.02,
        ttsNextPlaybackTimeRef.current,
      );
      source.start(scheduledAt);
      ttsNextPlaybackTimeRef.current = scheduledAt + buffer.duration;

      const streamSession = playback.streamSession;
      const utteranceSeq = playback.utteranceSeq;
      ttsPlaybackSourcesRef.current.add(source);
      source.onended = () => {
        ttsPlaybackSourcesRef.current.delete(source);
        finalizeVoiceWsPlaybackIfComplete(utteranceSeq, streamSession);
      };

      if (chunk.isFinal) {
        playback.streamDone = true;
      }

      if (context.state === "suspended") {
        context.resume().catch(() => {});
      }
    },
    [finalizeVoiceWsPlaybackIfComplete],
  );

  const handleVoiceRealtimeServerEvent = useCallback(
    (event: VoiceRealtimeServerEvent) => {
      const eventSeq =
        "utterance_seq" in event && typeof event.utterance_seq === "number"
          ? event.utterance_seq
          : null;
      const ignoreAssistantEvent =
        eventSeq != null &&
        eventSeq < voiceMinAcceptedAssistantSeqRef.current &&
        (event.type.startsWith("assistant_") ||
          (event.type === "turn_done" && event.status === "interrupted"));

      if (ignoreAssistantEvent) {
        return;
      }

      switch (event.type) {
        case "connected":
          return;
        case "session_ready":
          voiceWsSessionReadyRef.current = true;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current === "idle"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        case "input_stream_ready":
          voiceWsInputStreamStartedRef.current = true;
          return;
        case "input_stream_stopped":
          voiceWsInputStreamStartedRef.current = false;
          return;
        case "listening":
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        case "user_speech_start":
          voiceMinAcceptedAssistantSeqRef.current = Math.max(
            voiceMinAcceptedAssistantSeqRef.current,
            event.utterance_seq,
          );
          if (isSessionActiveRef.current) {
            setRuntimeStatus("user_speaking");
          }
          return;
        case "user_speech_end":
          processingRef.current = true;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("processing");
          }
          return;
        case "turn_processing":
          processingRef.current = true;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking"
          ) {
            setRuntimeStatus("processing");
          }
          return;
        case "user_transcript_start": {
          const existingId = voiceUserEntryIdsRef.current.get(
            event.utterance_id,
          );
          if (!existingId) {
            const entryId = makeTranscriptEntryId("user");
            voiceUserEntryIdsRef.current.set(event.utterance_id, entryId);
            appendTranscriptEntry({
              id: entryId,
              role: "user",
              text: "",
              timestamp: Date.now(),
            });
          }
          return;
        }
        case "user_transcript_delta": {
          const entryId = voiceUserEntryIdsRef.current.get(event.utterance_id);
          if (!entryId) return;
          setTranscript((prev) => {
            const index = prev.findIndex((entry) => entry.id === entryId);
            if (index === -1) return prev;
            const next = [...prev];
            next[index] = {
              ...next[index],
              text: `${next[index].text}${event.delta}`,
            };
            return next;
          });
          return;
        }
        case "user_transcript_final": {
          const entryId = voiceUserEntryIdsRef.current.get(event.utterance_id);
          if (!entryId) return;
          const finalText = (event.text ?? "").trim();
          if (finalText) {
            setTranscriptEntryText(entryId, finalText);
          } else {
            removeTranscriptEntry(entryId);
            voiceUserEntryIdsRef.current.delete(event.utterance_id);
          }
          return;
        }
        case "assistant_text_start": {
          const existingId = voiceAssistantEntryIdsRef.current.get(
            event.utterance_id,
          );
          if (!existingId) {
            const entryId = makeTranscriptEntryId("assistant");
            voiceAssistantEntryIdsRef.current.set(event.utterance_id, entryId);
            appendTranscriptEntry({
              id: entryId,
              role: "assistant",
              text: "",
              timestamp: Date.now(),
            });
          }
          return;
        }
        case "assistant_text_final": {
          const entryId = voiceAssistantEntryIdsRef.current.get(
            event.utterance_id,
          );
          const finalText = parseFinalAnswer((event.text ?? "").trim());
          if (entryId) {
            if (finalText) {
              setTranscriptEntryText(entryId, finalText);
            } else {
              removeTranscriptEntry(entryId);
              voiceAssistantEntryIdsRef.current.delete(event.utterance_id);
            }
          }
          return;
        }
        case "assistant_audio_start": {
          if (event.audio_format !== "pcm_i16") {
            const message = `Unsupported streamed audio format '${event.audio_format}'. Expected pcm_i16.`;
            setError(message);
            onError?.(message);
            return;
          }

          clearAudioPlayback();

          const playbackContext = new AudioContext();
          ttsPlaybackContextRef.current = playbackContext;
          ttsNextPlaybackTimeRef.current = playbackContext.currentTime + 0.05;
          ttsSampleRateRef.current = event.sample_rate || 24000;
          ttsSamplesRef.current = [];

          const streamSession = ++ttsStreamSessionRef.current;
          voiceWsPlaybackRef.current = {
            utteranceId: event.utterance_id,
            utteranceSeq: event.utterance_seq,
            streamSession,
            streamDone: false,
            playbackStarted: false,
          };
          return;
        }
        case "assistant_audio_done": {
          const playback = voiceWsPlaybackRef.current;
          if (!playback) {
            if (
              isSessionActiveRef.current &&
              runtimeStatusRef.current !== "user_speaking"
            ) {
              setRuntimeStatus("listening");
            }
            processingRef.current = false;
            return;
          }
          if (
            playback.utteranceId !== event.utterance_id ||
            playback.utteranceSeq !== event.utterance_seq
          ) {
            return;
          }
          playback.streamDone = true;
          finalizeVoiceWsPlaybackIfComplete(
            playback.utteranceSeq,
            playback.streamSession,
          );
          return;
        }
        case "turn_done": {
          if (event.status !== "ok" && event.status !== "interrupted") {
            processingRef.current = false;
          }

          if (event.status === "interrupted") {
            if (
              voiceWsPlaybackRef.current?.utteranceSeq === event.utterance_seq
            ) {
              voiceWsPlaybackRef.current = null;
            }
            return;
          }

          if (!voiceWsPlaybackRef.current && isSessionActiveRef.current) {
            if (runtimeStatusRef.current !== "user_speaking") {
              setRuntimeStatus("listening");
            }
            processingRef.current = false;
          }
          return;
        }
        case "error": {
          const message = event.message || "Voice realtime error";
          setError(message);
          onError?.(message);
          processingRef.current = false;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking" &&
            runtimeStatusRef.current !== "assistant_speaking"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        }
        case "pong":
          return;
      }
    },
    [
      appendTranscriptEntry,
      clearAudioPlayback,
      finalizeVoiceWsPlaybackIfComplete,
      onError,
      removeTranscriptEntry,
      setTranscriptEntryText,
    ],
  );

  const ensureVoiceRealtimeSocket =
    useCallback(async (): Promise<WebSocket> => {
      const existing = voiceWsRef.current;
      if (existing && existing.readyState === WebSocket.OPEN) {
        return existing;
      }
      if (voiceWsConnectingRef.current) {
        return voiceWsConnectingRef.current;
      }

      const url = buildVoiceRealtimeWebSocketUrl(api.baseUrl);
      const promise = new Promise<WebSocket>((resolve, reject) => {
        let settled = false;
        const ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        ws.onopen = () => {
          voiceWsRef.current = ws;
          voiceWsSessionReadyRef.current = false;
          voiceWsInputStreamStartedRef.current = false;
          voiceWsInputFrameSeqRef.current = 0;
          try {
            ws.send(
              JSON.stringify({
                type: "session_start",
                system_prompt: VOICE_AGENT_SYSTEM_PROMPT,
              } satisfies VoiceRealtimeClientMessage),
            );
          } catch (error) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(
                error instanceof Error
                  ? error
                  : new Error("Failed to initialize voice realtime websocket"),
              );
            });
            return;
          }
          settle(() => resolve(ws));
        };

        ws.onmessage = (messageEvent) => {
          if (messageEvent.data instanceof ArrayBuffer) {
            const chunk = parseVoiceRealtimeAssistantAudioBinaryChunk(
              messageEvent.data,
            );
            if (chunk) {
              handleVoiceRealtimeAssistantAudioBinaryChunk(chunk);
            }
            return;
          }
          if (messageEvent.data instanceof Blob) {
            void messageEvent.data.arrayBuffer().then((buffer) => {
              const chunk = parseVoiceRealtimeAssistantAudioBinaryChunk(buffer);
              if (chunk) {
                handleVoiceRealtimeAssistantAudioBinaryChunk(chunk);
              }
            });
            return;
          }
          if (typeof messageEvent.data !== "string") return;
          try {
            const parsed: unknown = JSON.parse(messageEvent.data);
            if (!isVoiceRealtimeServerEvent(parsed)) {
              return;
            }
            handleVoiceRealtimeServerEvent(parsed);
          } catch {
            // Ignore malformed events.
          }
        };

        ws.onerror = () => {
          const message = "Voice realtime websocket error";
          if (!settled) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(new Error(message));
            });
          }
        };

        ws.onclose = () => {
          if (!settled) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(
                new Error("Voice realtime connection closed during setup"),
              );
            });
          }
          const wasActive = isSessionActiveRef.current;
          const wasCurrent = voiceWsRef.current === ws;
          if (wasCurrent) {
            voiceWsRef.current = null;
          }
          voiceWsSessionReadyRef.current = false;
          voiceWsInputStreamStartedRef.current = false;
          voiceWsInputFrameSeqRef.current = 0;
          voiceWsInputStreamStartingRef.current = null;
          voiceWsConnectingRef.current = null;
          if (wasActive && wasCurrent) {
            processingRef.current = false;
            if (runtimeStatusRef.current !== "idle") {
              setRuntimeStatus("idle");
            }
            const message = "Voice realtime connection closed";
            setError(message);
            onError?.(message);
          }
        };
      });

      voiceWsConnectingRef.current = promise;
      try {
        return await promise;
      } finally {
        if (voiceWsConnectingRef.current === promise) {
          voiceWsConnectingRef.current = null;
        }
      }
    }, [
      handleVoiceRealtimeAssistantAudioBinaryChunk,
      handleVoiceRealtimeServerEvent,
      onError,
    ]);

  const ensureVoiceRealtimeInputStreamStarted = useCallback(
    async (inputSampleRate: number) => {
      if (voiceWsInputStreamStartedRef.current) {
        return;
      }
      if (voiceWsInputStreamStartingRef.current) {
        return voiceWsInputStreamStartingRef.current;
      }
      if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
        throw new Error(
          "Select ASR, text, and TTS models before starting voice mode.",
        );
      }

      const startPromise = (async () => {
        const socket = await ensureVoiceRealtimeSocket();
        if (socket.readyState !== WebSocket.OPEN) {
          throw new Error("Voice realtime websocket is not connected");
        }
        sendVoiceRealtimeJson({
          type: "input_stream_start",
          asr_model_id: selectedAsrModel,
          text_model_id: selectedTextModel,
          tts_model_id: selectedTtsModel,
          speaker: selectedSpeaker,
          asr_language: "Auto",
          max_output_tokens: 1536,
          vad_threshold: vadThreshold,
          min_speech_ms: minSpeechMs,
          silence_duration_ms: silenceDurationMs,
          max_utterance_ms: 20_000,
          pre_roll_ms: 160,
          input_sample_rate: Math.round(inputSampleRate),
        });
      })();

      voiceWsInputStreamStartingRef.current = startPromise.finally(() => {
        if (voiceWsInputStreamStartingRef.current === startPromise) {
          voiceWsInputStreamStartingRef.current = null;
        }
      });

      return voiceWsInputStreamStartingRef.current;
    },
    [
      ensureVoiceRealtimeSocket,
      minSpeechMs,
      selectedAsrModel,
      selectedSpeaker,
      selectedTextModel,
      selectedTtsModel,
      sendVoiceRealtimeJson,
      silenceDurationMs,
      vadThreshold,
    ],
  );

  const streamUserTranscription = useCallback(
    (audioBlob: Blob, modelId: string): Promise<string> =>
      new Promise((resolve, reject) => {
        const entryId = makeTranscriptEntryId("user");
        let assembledText = "";
        let settled = false;

        appendTranscriptEntry({
          id: entryId,
          role: "user",
          text: "",
          timestamp: Date.now(),
        });

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        asrStreamAbortRef.current = api.asrTranscribeStream(
          {
            audio_file: audioBlob,
            audio_filename: "voice-turn.wav",
            model_id: modelId,
            language: "Auto",
          },
          {
            onDelta: (delta) => {
              assembledText += delta;
              setTranscriptEntryText(entryId, assembledText);
            },
            onPartial: (text) => {
              assembledText = text;
              setTranscriptEntryText(entryId, assembledText);
            },
            onFinal: (text) => {
              assembledText = text;
              setTranscriptEntryText(entryId, assembledText);
            },
            onError: (errorMessage) => {
              settle(() => {
                asrStreamAbortRef.current = null;
                const finalText = assembledText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                reject(new Error(errorMessage));
              });
            },
            onDone: () => {
              settle(() => {
                asrStreamAbortRef.current = null;
                const finalText = assembledText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                resolve(finalText);
              });
            },
          },
        );
      }),
    [appendTranscriptEntry, removeTranscriptEntry, setTranscriptEntryText],
  );

  const ensureAgentSession = useCallback(async (modelId: string) => {
    if (agentSessionIdRef.current) {
      return agentSessionIdRef.current;
    }

    const session = await api.createAgentSession({
      agent_id: "voice-agent",
      model_id: modelId,
      system_prompt: VOICE_AGENT_SYSTEM_PROMPT,
      planning_mode: "auto",
      title: "Voice Session",
    });

    agentSessionIdRef.current = session.id;
    return session.id;
  }, []);

  const streamAssistantResponse = useCallback(
    (userText: string, modelId: string): Promise<string> =>
      new Promise((resolve, reject) => {
        const entryId = makeTranscriptEntryId("assistant");
        let rawText = "";
        let settled = false;

        appendTranscriptEntry({
          id: entryId,
          role: "assistant",
          text: "",
          timestamp: Date.now(),
        });

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        const updateVisibleText = () => {
          const visible = parseFinalAnswer(rawText);
          setTranscriptEntryText(entryId, visible);
        };

        const abortController = new AbortController();
        chatStreamAbortRef.current = abortController;

        const run = async () => {
          try {
            const sessionId = await ensureAgentSession(modelId);
            const response = await api.createAgentTurn(
              sessionId,
              {
                input: userText,
                model_id: modelId,
                max_output_tokens: 1536,
              },
              abortController.signal,
            );

            rawText = response.assistant_text ?? "";
            updateVisibleText();
            settle(() => {
              chatStreamAbortRef.current = null;
              const finalText = parseFinalAnswer(rawText) || rawText.trim();
              if (finalText) {
                setTranscriptEntryText(entryId, finalText);
              } else {
                removeTranscriptEntry(entryId);
              }
              resolve(finalText);
            });
          } catch (error) {
            if ((error as Error).name === "AbortError") {
              settle(() => {
                chatStreamAbortRef.current = null;
                const finalText = parseFinalAnswer(rawText) || rawText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                reject(error as Error);
              });
              return;
            }

            settle(() => {
              chatStreamAbortRef.current = null;
              const finalText = parseFinalAnswer(rawText) || rawText.trim();
              if (finalText) {
                setTranscriptEntryText(entryId, finalText);
              } else {
                removeTranscriptEntry(entryId);
              }
              reject(
                error instanceof Error
                  ? error
                  : new Error("Agent response failed"),
              );
            });
          }
        };

        run();
      }),
    [
      appendTranscriptEntry,
      ensureAgentSession,
      removeTranscriptEntry,
      setTranscriptEntryText,
    ],
  );

  const streamAssistantSpeech = useCallback(
    (text: string, modelId: string, speaker: string, turnId: number) =>
      new Promise<void>((resolve, reject) => {
        clearAudioPlayback();

        const playbackContext = new AudioContext();
        ttsPlaybackContextRef.current = playbackContext;
        ttsNextPlaybackTimeRef.current = playbackContext.currentTime + 0.05;
        ttsSampleRateRef.current = 24000;
        ttsSamplesRef.current = [];

        const streamSession = ++ttsStreamSessionRef.current;
        let settled = false;
        let streamDone = false;
        let playbackStarted = false;

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        const finalizeIfComplete = () => {
          if (!streamDone || ttsPlaybackSourcesRef.current.size > 0) {
            return;
          }

          if (ttsStreamSessionRef.current === streamSession) {
            const merged = mergeSampleChunks(ttsSamplesRef.current);
            if (merged.length > 0) {
              const wavBlob = encodeWavPcm16(merged, ttsSampleRateRef.current);
              const nextUrl = URL.createObjectURL(wavBlob);
              if (audioUrlRef.current) {
                URL.revokeObjectURL(audioUrlRef.current);
              }
              audioUrlRef.current = nextUrl;
            }

            if (ttsPlaybackContextRef.current) {
              ttsPlaybackContextRef.current.close().catch(() => {});
              ttsPlaybackContextRef.current = null;
            }

            ttsPlaybackSourcesRef.current.clear();
            ttsNextPlaybackTimeRef.current = 0;
            ttsSamplesRef.current = [];
            ttsStreamAbortRef.current = null;

            if (turnId === turnIdRef.current) {
              if (isSessionActiveRef.current) {
                setRuntimeStatus("listening");
              } else {
                setRuntimeStatus("idle");
              }
            }
          }

          settle(() => resolve());
        };

        ttsStreamAbortRef.current = api.generateTTSStream(
          {
            text,
            model_id: modelId,
            speaker,
            max_tokens: 0,
            format: "pcm",
          },
          {
            onStart: ({ sampleRate, audioFormat }) => {
              if (ttsStreamSessionRef.current !== streamSession) return;
              ttsSampleRateRef.current = sampleRate;

              if (audioFormat !== "pcm_i16") {
                stopTtsStreamingPlayback();
                settle(() => {
                  reject(
                    new Error(
                      `Unsupported streamed audio format '${audioFormat}'. Expected pcm_i16.`,
                    ),
                  );
                });
              }
            },
            onChunk: ({ audioBase64 }) => {
              if (ttsStreamSessionRef.current !== streamSession) return;

              const context = ttsPlaybackContextRef.current;
              if (!context) return;

              const samples = decodePcmI16Base64(audioBase64);
              if (samples.length === 0) return;

              if (!playbackStarted) {
                playbackStarted = true;
                processingRef.current = false;
                if (turnId === turnIdRef.current) {
                  setRuntimeStatus("assistant_speaking");
                }
              }

              ttsSamplesRef.current.push(samples);

              const buffer = context.createBuffer(
                1,
                samples.length,
                ttsSampleRateRef.current,
              );
              const samplesForPlayback = new Float32Array(samples.length);
              samplesForPlayback.set(samples);
              buffer.copyToChannel(samplesForPlayback, 0);

              const source = context.createBufferSource();
              source.buffer = buffer;
              source.connect(context.destination);

              const scheduledAt = Math.max(
                context.currentTime + 0.02,
                ttsNextPlaybackTimeRef.current,
              );
              source.start(scheduledAt);
              ttsNextPlaybackTimeRef.current = scheduledAt + buffer.duration;

              ttsPlaybackSourcesRef.current.add(source);
              source.onended = () => {
                ttsPlaybackSourcesRef.current.delete(source);
                finalizeIfComplete();
              };

              if (context.state === "suspended") {
                context.resume().catch(() => {});
              }
            },
            onError: (errorMessage) => {
              if (ttsStreamSessionRef.current !== streamSession) {
                settle(() => resolve());
                return;
              }

              stopTtsStreamingPlayback();
              settle(() => reject(new Error(errorMessage)));
            },
            onDone: () => {
              if (ttsStreamSessionRef.current !== streamSession) {
                settle(() => resolve());
                return;
              }

              streamDone = true;
              if (!playbackStarted) {
                processingRef.current = false;
              }
              finalizeIfComplete();
            },
          },
        );
      }),
    [clearAudioPlayback, stopTtsStreamingPlayback],
  );

  const playAssistantBlob = useCallback(
    (audioBlob: Blob, turnId: number) =>
      new Promise<void>((resolve, reject) => {
        clearAudioPlayback();

        const nextUrl = URL.createObjectURL(audioBlob);
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current);
        }
        audioUrlRef.current = nextUrl;

        let audio = audioRef.current;
        if (!audio) {
          audio = new Audio();
          audioRef.current = audio;
        }

        const finalize = (error?: Error) => {
          audio!.onended = null;
          audio!.onerror = null;

          if (turnId === turnIdRef.current) {
            if (isSessionActiveRef.current) {
              setRuntimeStatus("listening");
            } else {
              setRuntimeStatus("idle");
            }
          }

          if (error) {
            reject(error);
          } else {
            resolve();
          }
        };

        audio.src = nextUrl;
        audio.onended = () => finalize();
        audio.onerror = () =>
          finalize(new Error("Failed to play assistant audio"));

        if (turnId === turnIdRef.current) {
          setRuntimeStatus("assistant_speaking");
        }

        audio.play().catch((error) => {
          finalize(
            error instanceof Error
              ? error
              : new Error("Failed to start assistant audio playback"),
          );
        });
      }),
    [clearAudioPlayback],
  );

  const processUtterance = useCallback(
    async (audioBlob: Blob) => {
      if (!isSessionActiveRef.current) {
        return;
      }

      if (
        (lfm2DirectMode && !selectedS2sModel) ||
        (!lfm2DirectMode && !selectedAsrModel) ||
        (!lfm2DirectMode && (!selectedTextModel || !selectedTtsModel))
      ) {
        setError(
          lfm2DirectMode
            ? "Select a speech-to-speech model before starting voice mode."
            : "Select ASR, text, and TTS models before starting voice mode.",
        );
        setIsConfigOpen(true);
        setRuntimeStatus("listening");
        processingRef.current = false;
        return;
      }

      if (!hasRunnableConfig) {
        setError(
          "Selected models must be loaded. Open Config to manage models.",
        );
        setIsConfigOpen(true);
        setRuntimeStatus("listening");
        processingRef.current = false;
        return;
      }

      const turnId = turnIdRef.current + 1;
      turnIdRef.current = turnId;

      try {
        setRuntimeStatus("processing");

        if (lfm2DirectMode) {
          const wavBlob = await transcodeToWav(audioBlob, 24000);
          if (turnId !== turnIdRef.current || !isSessionActiveRef.current)
            return;

          const response = await api.speechToSpeech({
            audio_file: wavBlob,
            audio_filename: "voice-turn.wav",
            model_id: selectedS2sModel!,
            language: "English",
          });

          if (turnId !== turnIdRef.current || !isSessionActiveRef.current)
            return;

          let userText = response.transcription?.trim() || "";
          if (!userText) {
            try {
              const fallback = await api.asrTranscribe({
                audio_file: wavBlob,
                audio_filename: "voice-turn.wav",
                model_id: selectedS2sModel!,
                language: "English",
              });
              userText = fallback.transcription.trim();
            } catch {
              // Keep the turn visible even if transcription fallback fails.
            }
          }
          if (!userText) {
            userText = "User speech captured (transcription unavailable).";
          }
          if (userText) {
            appendTranscriptEntry({
              id: makeTranscriptEntryId("user"),
              role: "user",
              text: userText,
              timestamp: Date.now(),
            });
          }

          const assistantText = response.text.trim();
          if (assistantText) {
            appendTranscriptEntry({
              id: makeTranscriptEntryId("assistant"),
              role: "assistant",
              text: assistantText,
              timestamp: Date.now(),
            });
          }

          await playAssistantBlob(response.audioBlob, turnId);
          return;
        }

        const wavBlob = await transcodeToWav(audioBlob, 16000);
        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        const userText = await streamUserTranscription(
          wavBlob,
          selectedAsrModel!,
        );

        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        if (!userText) {
          processingRef.current = false;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("listening");
          }
          return;
        }

        const assistantText = await streamAssistantResponse(
          userText,
          selectedTextModel!,
        );
        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        if (!assistantText) {
          processingRef.current = false;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("listening");
          }
          return;
        }

        await streamAssistantSpeech(
          assistantText,
          selectedTtsModel!,
          selectedSpeaker,
          turnId,
        );
      } catch (err) {
        if (turnId !== turnIdRef.current) {
          return;
        }

        const message =
          err instanceof Error ? err.message : "Voice turn failed";
        setError(message);
        onError?.(message);
        if (isSessionActiveRef.current) {
          setRuntimeStatus("listening");
        } else {
          setRuntimeStatus("idle");
        }
      } finally {
        if (turnId === turnIdRef.current) {
          processingRef.current = false;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current === "processing"
          ) {
            setRuntimeStatus("listening");
          }
        }
      }
    },
    [
      appendTranscriptEntry,
      hasRunnableConfig,
      lfm2DirectMode,
      onError,
      playAssistantBlob,
      selectedAsrModel,
      selectedS2sModel,
      selectedSpeaker,
      selectedTextModel,
      selectedTtsModel,
      streamAssistantResponse,
      streamAssistantSpeech,
      streamUserTranscription,
    ],
  );

  const startSession = useCallback(async () => {
    if (
      (lfm2DirectMode && !selectedS2sModel) ||
      (!lfm2DirectMode && !selectedAsrModel) ||
      (!lfm2DirectMode && (!selectedTextModel || !selectedTtsModel))
    ) {
      const message = lfm2DirectMode
        ? "Select a speech-to-speech model before starting voice mode."
        : "Select ASR, text, and TTS models before starting voice mode.";
      setError(message);
      onError?.(message);
      setIsConfigOpen(true);
      return;
    }

    if (!hasRunnableConfig) {
      const message =
        "Selected models must be loaded. Open Config to manage models.";
      setError(message);
      onError?.(message);
      setIsConfigOpen(true);
      return;
    }

    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);
      analyserRef.current = analyser;

      let recorder: MediaRecorder | null = null;
      if (lfm2DirectMode) {
        const mimeCandidates = [
          "audio/webm;codecs=opus",
          "audio/webm",
          "audio/mp4",
        ];
        for (const mimeType of mimeCandidates) {
          if (MediaRecorder.isTypeSupported(mimeType)) {
            recorder = new MediaRecorder(stream, { mimeType });
            break;
          }
        }
        if (!recorder) {
          recorder = new MediaRecorder(stream);
        }

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunksRef.current.push(event.data);
          }
        };

        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, {
            type: recorder?.mimeType || "audio/webm",
          });
          chunksRef.current = [];

          if (blob.size < 1200) {
            processingRef.current = false;
            if (
              isSessionActiveRef.current &&
              runtimeStatusRef.current !== "assistant_speaking"
            ) {
              setRuntimeStatus("listening");
            }
            return;
          }

          void processUtterance(blob);
        };
      } else {
        const processor = audioContext.createScriptProcessor(2048, 1, 1);
        const sink = audioContext.createGain();
        sink.gain.value = 0;

        processor.onaudioprocess = (event) => {
          if (!isSessionActiveRef.current) return;
          if (!voiceWsInputStreamStartedRef.current) return;

          const inputBuffer = event.inputBuffer;
          const channelCount = inputBuffer.numberOfChannels;
          const frameCount = inputBuffer.length;
          if (frameCount <= 0 || channelCount <= 0) return;

          const mono = new Float32Array(frameCount);
          for (let ch = 0; ch < channelCount; ch += 1) {
            const channel = inputBuffer.getChannelData(ch);
            for (let i = 0; i < frameCount; i += 1) {
              mono[i] += (channel[i] ?? 0) / channelCount;
            }
          }

          const pcm16 = encodeFloat32ToPcm16Bytes(mono);
          const nextSeq = (voiceWsInputFrameSeqRef.current + 1) >>> 0;
          voiceWsInputFrameSeqRef.current = nextSeq;

          try {
            sendVoiceRealtimeBinary(
              encodeVoiceRealtimeClientPcm16Frame(
                pcm16,
                Math.round(inputBuffer.sampleRate),
                nextSeq,
              ),
            );
          } catch {
            // Best-effort; websocket lifecycle handles reconnect/error surfaces.
          }
        };

        source.connect(processor);
        processor.connect(sink);
        sink.connect(audioContext.destination);
        streamingProcessorRef.current = processor;
        streamingProcessorSinkRef.current = sink;
      }

      mediaRecorderRef.current = recorder;
      isSessionActiveRef.current = true;
      processingRef.current = false;
      silenceMsRef.current = 0;
      speechStartRef.current = null;
      setRuntimeStatus("listening");

      if (!lfm2DirectMode) {
        // Warm up realtime websocket + server-side VAD stream without blocking mic startup.
        void ensureVoiceRealtimeInputStreamStarted(
          audioContext.sampleRate,
        ).catch((err) => {
          const message =
            err instanceof Error
              ? err.message
              : "Voice realtime connection failed";
          if (!isSessionActiveRef.current) {
            return;
          }
          setError(message);
          onError?.(message);
        });
      }

      const VAD_INTERVAL = 80;
      vadTimerRef.current = window.setInterval(() => {
        const analyserNode = analyserRef.current;
        const recorderNode = mediaRecorderRef.current;
        if (!analyserNode || !isSessionActiveRef.current) return;

        const data = new Uint8Array(analyserNode.fftSize);
        analyserNode.getByteTimeDomainData(data);

        let sumSquares = 0;
        for (let i = 0; i < data.length; i += 1) {
          const centered = (data[i] - 128) / 128;
          sumSquares += centered * centered;
        }
        const rms = Math.sqrt(sumSquares / data.length);
        setAudioLevel(rms);

        const isSpeech = rms >= vadThreshold;
        const now = Date.now();

        if (!lfm2DirectMode) {
          if (isSpeech && runtimeStatusRef.current === "assistant_speaking") {
            const nextAccepted =
              (voiceWsPlaybackRef.current?.utteranceSeq ?? 0) + 1;
            voiceMinAcceptedAssistantSeqRef.current = Math.max(
              voiceMinAcceptedAssistantSeqRef.current,
              nextAccepted,
            );
            try {
              sendVoiceRealtimeJson({ type: "interrupt", reason: "barge_in" });
            } catch {
              // Best-effort; local playback is stopped immediately.
            }
            clearAudioPlayback();
            setRuntimeStatus("listening");
          }
          return;
        }

        if (!recorderNode) return;
        const isRecording = recorderNode.state === "recording";

        if (isSpeech) {
          silenceMsRef.current = 0;

          if (runtimeStatusRef.current === "assistant_speaking") {
            clearAudioPlayback();
            setRuntimeStatus("listening");
          }

          if (!isRecording && !processingRef.current) {
            chunksRef.current = [];
            recorderNode.start();
            speechStartRef.current = now;
            setRuntimeStatus("user_speaking");
          }
          return;
        }

        if (isRecording) {
          silenceMsRef.current += VAD_INTERVAL;
          const speechDuration = speechStartRef.current
            ? now - speechStartRef.current
            : 0;
          if (
            speechDuration >= minSpeechMs &&
            silenceMsRef.current >= silenceDurationMs
          ) {
            processingRef.current = true;
            setRuntimeStatus("processing");
            recorderNode.stop();
            silenceMsRef.current = 0;
            speechStartRef.current = null;
          }
        }
      }, VAD_INTERVAL);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to start microphone session";
      setError(message);
      onError?.(message);
      stopSession();
    }
  }, [
    clearAudioPlayback,
    ensureVoiceRealtimeInputStreamStarted,
    hasRunnableConfig,
    lfm2DirectMode,
    minSpeechMs,
    onError,
    processUtterance,
    selectedAsrModel,
    selectedS2sModel,
    selectedTextModel,
    selectedTtsModel,
    sendVoiceRealtimeBinary,
    sendVoiceRealtimeJson,
    silenceDurationMs,
    stopSession,
    vadThreshold,
  ]);

  const toggleSession = () => {
    if (runtimeStatus === "idle") {
      void startSession();
    } else {
      stopSession();
    }
  };

  const statusLabel = {
    idle: "Idle",
    listening: "Listening",
    user_speaking: "User speaking",
    processing: "Thinking",
    assistant_speaking: "Assistant speaking",
  }[runtimeStatus];

  const vadPercent = Math.min(
    100,
    Math.round((audioLevel / Math.max(vadThreshold, 0.001)) * 40),
  );

  const getStatusClass = (status: ModelInfo["status"]) => {
    switch (status) {
      case "ready":
        return "bg-white/10 border-white/20 text-gray-300";
      case "loading":
      case "downloading":
        return "bg-amber-500/15 border-amber-500/40 text-amber-300";
      case "downloaded":
        return "bg-white/10 border-white/20 text-gray-300";
      case "error":
        return "bg-red-500/15 border-red-500/40 text-red-300";
      default:
        return "bg-[#1c1c1c] border-[#2a2a2a] text-gray-500";
    }
  };

  const getStatusLabel = (status: ModelInfo["status"]) => {
    switch (status) {
      case "not_downloaded":
        return "Not downloaded";
      case "downloading":
        return "Downloading";
      case "downloaded":
        return "Downloaded";
      case "loading":
        return "Loading";
      case "ready":
        return "Loaded";
      case "error":
        return "Error";
      default:
        return status;
    }
  };

  const getModelRoles = (variant: string): string[] => {
    const roles: string[] = [];
    if (isAsrVariant(variant)) roles.push("ASR");
    if (isTextVariant(variant)) roles.push("TEXT");
    if (isTtsVariant(variant)) roles.push("TTS");
    return roles;
  };

  const startDisabled =
    (lfm2DirectMode && !selectedS2sModel) ||
    (!lfm2DirectMode &&
      (!selectedAsrModel || !selectedTextModel || !selectedTtsModel)) ||
    !hasRunnableConfig;

  const modelSummary = lfm2DirectMode
    ? [{ label: "S2S Model", model: selectedS2sInfo }]
    : [
        { label: "ASR", model: selectedAsrInfo },
        { label: "Text", model: selectedTextInfo },
        { label: "TTS", model: selectedTtsInfo },
      ];

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col items-center justify-center py-24 gap-3">
          <motion.div
            className="w-8 h-8 border-2 border-white border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <p className="text-sm text-gray-400">Loading models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-start justify-between gap-3 mb-6">
        <div>
          <h1 className="text-xl font-semibold text-white">Realtime Voice</h1>
          <p className="text-sm text-gray-500 mt-1">
            {"Low latency realtime voice loop."}
          </p>
        </div>
        <button
          onClick={() => setIsConfigOpen(true)}
          className="btn btn-secondary text-sm"
        >
          <Settings2 className="w-4 h-4" />
          Config
        </button>
      </div>

      <div className="grid xl:grid-cols-[360px,1fr] gap-4 lg:gap-6">
        <div className="card p-5">
          <div className="flex flex-col items-center text-center">
            <div className="relative mb-5">
              <div className="w-28 h-28 rounded-full bg-[#141414] border border-[#2a2a2a] flex items-center justify-center">
                {runtimeStatus === "assistant_speaking" ? (
                  <Volume2 className="w-8 h-8 text-white" />
                ) : runtimeStatus === "user_speaking" ? (
                  <AudioLines className="w-8 h-8 text-white" />
                ) : runtimeStatus === "processing" ? (
                  <Loader2 className="w-8 h-8 text-white animate-spin" />
                ) : runtimeStatus === "listening" ? (
                  <Mic className="w-8 h-8 text-white" />
                ) : (
                  <MicOff className="w-8 h-8 text-gray-500" />
                )}
              </div>
              <div className="absolute -inset-2 rounded-full border border-white/10" />
            </div>

            <div className="text-sm text-white font-medium">{statusLabel}</div>
            <p className="text-xs text-gray-500 mt-1">
              Barge-in is enabled while the assistant is speaking.
            </p>
            <span className="mt-2 inline-flex items-center rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-3)] px-2 py-0.5 text-[11px] text-[var(--text-secondary)]">
              {currentPipelineLabel}
            </span>

            <Button
              onClick={toggleSession}
              variant={runtimeStatus === "idle" ? "default" : "destructive"}
              className="w-full mt-5 text-sm min-h-[46px] gap-2"
              disabled={startDisabled}
            >
              {runtimeStatus === "idle" ? (
                <>
                  <Mic className="w-4 h-4" />
                  Start Session
                </>
              ) : (
                <>
                  <PhoneOff className="w-4 h-4" />
                  Stop Session
                </>
              )}
            </Button>
          </div>

          <div className="mt-5 pt-4 border-t border-[#252525] space-y-3">
            <div className="h-2 rounded bg-[#1b1b1b] border border-[#2a2a2a] overflow-hidden">
              <div
                className="h-full bg-white transition-all duration-75"
                style={{ width: `${vadPercent}%` }}
              />
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="text-gray-500">Mode</span>
                <span className="inline-flex items-center rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-3)] px-2 py-0.5 text-[11px] text-[var(--text-secondary)]">
                  {currentPipelineLabel}
                </span>
              </div>
              {modelSummary.map((item) => (
                <div
                  key={item.label}
                  className="flex items-center justify-between gap-2"
                >
                  <span className="text-gray-500">{item.label}</span>
                  {item.model ? (
                    <span
                      className={clsx(
                        "inline-flex items-center rounded-md border px-2 py-0.5 max-w-[220px] truncate text-[11px]",
                        getStatusClass(item.model.status),
                      )}
                      title={item.model.variant}
                    >
                      {formatModelVariantLabel(item.model.variant)}
                    </span>
                  ) : (
                    <span className="text-[var(--status-warning-text)]">
                      Not selected
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="card p-4 flex flex-col h-[420px] sm:h-[520px] lg:h-[640px] overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm text-white font-medium">Conversation</span>
            <div className="flex items-center gap-2">
              <span className="text-xs px-2 py-1 rounded bg-[#1a1a1a] border border-[#2a2a2a] text-gray-300">
                {statusLabel}
              </span>
              <span className="text-xs px-2 py-1 rounded border border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-secondary)]">
                {currentPipelineLabel}
              </span>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto pr-1 space-y-3">
            {transcript.length === 0 ? (
              <div className="h-full flex items-center justify-center text-center">
                <div>
                  <p className="text-sm text-gray-400">No conversation yet.</p>
                  <p className="text-xs text-gray-600 mt-1">
                    Configure your voice stack and start a realtime session.
                  </p>
                </div>
              </div>
            ) : (
              transcript.map((entry) => {
                const isUser = entry.role === "user";
                return (
                  <div
                    key={entry.id}
                    className={clsx(
                      "flex",
                      isUser ? "justify-end" : "justify-start",
                    )}
                  >
                    <div
                      className={clsx(
                        "max-w-[85%] rounded-lg px-3 py-2.5 border text-sm whitespace-pre-wrap",
                        isUser
                          ? "bg-white text-black border-white"
                          : "bg-[#171717] text-gray-200 border-[#2a2a2a]",
                      )}
                    >
                      <div
                        className={clsx(
                          "text-[10px] mb-1 uppercase tracking-wide",
                          isUser ? "text-black/60" : "text-gray-500",
                        )}
                      >
                        {isUser ? "User" : "Assistant"}
                      </div>
                      {entry.text}
                    </div>
                  </div>
                );
              })
            )}
            <div ref={transcriptEndRef} />
          </div>

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 p-2 rounded bg-red-950/50 border border-red-900/50 text-red-300 text-xs"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <AnimatePresence>
        {isConfigOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm p-4 sm:p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsConfigOpen(false)}
          >
            <motion.div
              initial={{ y: 16, opacity: 0, scale: 0.98 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 16, opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2 }}
              className="mx-auto max-w-5xl max-h-[90vh] overflow-hidden card"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="px-4 sm:px-5 py-4 border-b border-[#262626] flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold text-white">
                    Voice Configuration
                  </h2>
                  <p className="text-xs text-gray-500 mt-1">
                    Configure realtime model stack and manage model lifecycle.
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-2"
                  onClick={() => setIsConfigOpen(false)}
                >
                  <X className="w-3.5 h-3.5" />
                  Close
                </Button>
              </div>

              <div className="p-4 sm:p-5 overflow-y-auto max-h-[calc(90vh-88px)] space-y-5">
                <section className="space-y-4">
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="text-sm font-medium text-white">
                      Pipeline Mode
                    </h3>
                    <span className="text-[11px] text-gray-500">
                      Choose your realtime voice route.
                    </span>
                  </div>
                  <div className="grid md:grid-cols-2 gap-3">
                    <button
                      className={cn(
                        "rounded-lg border p-3 text-left transition-colors",
                        lfm2DirectMode
                          ? "border-primary bg-primary/5 ring-1 ring-primary/20"
                          : "border-border bg-card hover:bg-accent hover:text-accent-foreground",
                      )}
                      onClick={() => setPipelineMode("s2s")}
                    >
                      <div className="text-sm font-medium">
                        Speech-to-Speech (S2S)
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        One LFM2 model handles user speech understanding and
                        assistant speech output.
                      </p>
                    </button>
                    <button
                      className={cn(
                        "rounded-lg border p-3 text-left transition-colors",
                        !lfm2DirectMode
                          ? "border-primary bg-primary/5 ring-1 ring-primary/20"
                          : "border-border bg-card hover:bg-accent hover:text-accent-foreground",
                      )}
                      onClick={() => setPipelineMode("stt_chat_tts")}
                    >
                      <div className="text-sm font-medium">
                        {"STT -> Chat -> TTS"}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Use separate ASR, language model, and TTS models for
                        maximum control.
                      </p>
                    </button>
                  </div>
                  <div className="text-[11px] text-[var(--text-muted)]">
                    Current mode: {currentPipelineLabel}
                  </div>
                </section>

                <section className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="text-sm font-medium text-white">
                      Model Assignment
                    </h3>
                    <span className="text-[11px] text-gray-500">
                      Assign one model to each active stage.
                    </span>
                  </div>
                  {lfm2DirectMode ? (
                    <div className="grid md:grid-cols-2 gap-3">
                      <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3 space-y-2">
                        <div className="flex items-center justify-between gap-2">
                          <label className="text-xs text-gray-400">
                            S2S Model
                          </label>
                          {selectedS2sInfo && (
                            <span
                              className={clsx(
                                "text-[10px] px-1.5 py-0.5 rounded border",
                                getStatusClass(selectedS2sInfo.status),
                              )}
                            >
                              {getStatusLabel(selectedS2sInfo.status)}
                            </span>
                          )}
                        </div>
                        <Select
                          value={selectedS2sModel ?? undefined}
                          onValueChange={setSelectedS2sModel}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select speech-to-speech model" />
                          </SelectTrigger>
                          <SelectContent>
                            {s2sModels.map((m) => (
                              <SelectItem key={m.variant} value={m.variant}>
                                {formatModelVariantLabel(m.variant)} •{" "}
                                {getStatusLabel(m.status)}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3">
                        <p className="text-xs text-gray-400">
                          STT, text, and TTS selectors are disabled in S2S mode
                          because the selected LFM2 model handles the entire
                          realtime speech loop.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="grid md:grid-cols-2 gap-3">
                      <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3 space-y-2">
                        <div className="flex items-center justify-between gap-2">
                          <label className="text-xs text-gray-400">
                            ASR Model
                          </label>
                          {selectedAsrInfo && (
                            <span
                              className={clsx(
                                "text-[10px] px-1.5 py-0.5 rounded border",
                                getStatusClass(selectedAsrInfo.status),
                              )}
                            >
                              {getStatusLabel(selectedAsrInfo.status)}
                            </span>
                          )}
                        </div>
                        <Select
                          value={selectedAsrModel ?? undefined}
                          onValueChange={setSelectedAsrModel}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select ASR model" />
                          </SelectTrigger>
                          <SelectContent>
                            {sttAsrModels.map((m) => (
                              <SelectItem key={m.variant} value={m.variant}>
                                {formatModelVariantLabel(m.variant)} •{" "}
                                {getStatusLabel(m.status)}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3 space-y-2">
                        <div className="flex items-center justify-between gap-2">
                          <label className="text-xs text-gray-400">
                            Text Model
                          </label>
                          {selectedTextInfo && (
                            <span
                              className={clsx(
                                "text-[10px] px-1.5 py-0.5 rounded border",
                                getStatusClass(selectedTextInfo.status),
                              )}
                            >
                              {getStatusLabel(selectedTextInfo.status)}
                            </span>
                          )}
                        </div>
                        <Select
                          value={selectedTextModel ?? undefined}
                          onValueChange={setSelectedTextModel}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select text model" />
                          </SelectTrigger>
                          <SelectContent>
                            {textModels.map((m) => (
                              <SelectItem key={m.variant} value={m.variant}>
                                {formatModelVariantLabel(m.variant)} •{" "}
                                {getStatusLabel(m.status)}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3 space-y-2">
                        <div className="flex items-center justify-between gap-2">
                          <label className="text-xs text-gray-400">
                            TTS Model
                          </label>
                          {selectedTtsInfo && (
                            <span
                              className={clsx(
                                "text-[10px] px-1.5 py-0.5 rounded border",
                                getStatusClass(selectedTtsInfo.status),
                              )}
                            >
                              {getStatusLabel(selectedTtsInfo.status)}
                            </span>
                          )}
                        </div>
                        <Select
                          value={selectedTtsModel ?? undefined}
                          onValueChange={setSelectedTtsModel}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select TTS model" />
                          </SelectTrigger>
                          <SelectContent>
                            {ttsConfigModels.map((m) => (
                              <SelectItem key={m.variant} value={m.variant}>
                                {formatModelVariantLabel(m.variant)} •{" "}
                                {getStatusLabel(m.status)}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3 space-y-2">
                        <label className="text-xs text-gray-400">
                          Assistant Voice
                        </label>
                        <Select
                          value={selectedSpeaker}
                          onValueChange={setSelectedSpeaker}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select assistant voice" />
                          </SelectTrigger>
                          <SelectContent>
                            {assistantSpeakers.map((speaker) => (
                              <SelectItem key={speaker.id} value={speaker.id}>
                                {speaker.name} ({speaker.language})
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  )}
                </section>

                <section className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3">
                  <details>
                    <summary className="cursor-pointer text-sm text-white">
                      Advanced Speech Detection
                    </summary>
                    <div className="mt-3 grid md:grid-cols-3 gap-4">
                      <div>
                        <label className="text-xs text-gray-500">
                          VAD Sensitivity ({vadThreshold.toFixed(3)})
                        </label>
                        <Slider
                          aria-label="VAD sensitivity"
                          min={0.005}
                          max={0.08}
                          step={0.001}
                          value={[vadThreshold]}
                          onValueChange={(value) => {
                            const next = value[0];
                            if (typeof next === "number") {
                              setVadThreshold(next);
                            }
                          }}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-gray-500">
                          End Silence (ms): {silenceDurationMs}
                        </label>
                        <Slider
                          aria-label="End silence duration"
                          min={400}
                          max={1800}
                          step={50}
                          value={[silenceDurationMs]}
                          onValueChange={(value) => {
                            const next = value[0];
                            if (typeof next === "number") {
                              setSilenceDurationMs(next);
                            }
                          }}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-gray-500">
                          Minimum Speech (ms): {minSpeechMs}
                        </label>
                        <Slider
                          aria-label="Minimum speech duration"
                          min={150}
                          max={1200}
                          step={50}
                          value={[minSpeechMs]}
                          onValueChange={(value) => {
                            const next = value[0];
                            if (typeof next === "number") {
                              setMinSpeechMs(next);
                            }
                          }}
                          className="mt-2"
                        />
                      </div>
                    </div>
                  </details>
                </section>

                <section className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium text-white">
                      Model Library
                    </h3>
                    <span className="text-[11px] text-gray-500">
                      Download, load, unload, and delete
                    </span>
                  </div>

                  {voiceRouteModels.map((model) => {
                    const roles = getModelRoles(model.variant);
                    const progressValue = downloadProgress[model.variant];
                    const progress =
                      progressValue?.percent ?? model.download_progress ?? 0;
                    const isSelectedS2s = selectedS2sModel === model.variant;
                    const isSelectedAsr = selectedAsrModel === model.variant;
                    const isSelectedText = selectedTextModel === model.variant;
                    const isSelectedTts = selectedTtsModel === model.variant;

                    return (
                      <div
                        key={model.variant}
                        className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-sm text-white font-medium truncate">
                              {model.variant}
                            </div>
                            <div className="mt-1 flex flex-nowrap items-center gap-1.5 overflow-x-auto">
                              {roles.map((role) => (
                                <span
                                  key={role}
                                  className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2f2f2f] text-gray-300 whitespace-nowrap"
                                >
                                  {role}
                                </span>
                              ))}
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2f2f2f] text-gray-300 whitespace-nowrap">
                                {getStatusLabel(model.status)}
                              </span>
                              {isSelectedS2s && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2f2f2f] text-gray-300 whitespace-nowrap">
                                  S2S selected
                                </span>
                              )}
                              {isSelectedAsr && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2f2f2f] text-gray-300 whitespace-nowrap">
                                  ASR selected
                                </span>
                              )}
                              {isSelectedText && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2f2f2f] text-gray-300 whitespace-nowrap">
                                  Text selected
                                </span>
                              )}
                              {isSelectedTts && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2f2f2f] text-gray-300 whitespace-nowrap">
                                  TTS selected
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center justify-end gap-2">
                            {model.status === "downloading" &&
                              onCancelDownload && (
                                <Button
                                  onClick={() =>
                                    onCancelDownload(model.variant)
                                  }
                                  variant="destructive"
                                  size="sm"
                                  className="text-xs h-8 gap-2"
                                >
                                  <X className="w-3.5 h-3.5" />
                                  Cancel
                                </Button>
                              )}
                            {(model.status === "not_downloaded" ||
                              model.status === "error") &&
                              (requiresManualDownload(model.variant) ? (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="text-xs h-8 gap-2"
                                  disabled
                                  title="Manual download required. See docs/user/manual-gemma-3-1b-download.md."
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  Manual download
                                </Button>
                              ) : (
                                <Button
                                  onClick={() => onDownload(model.variant)}
                                  size="sm"
                                  className="text-xs h-8 gap-2"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  Download
                                </Button>
                              ))}
                            {model.status === "downloaded" && (
                              <Button
                                onClick={() => onLoad(model.variant)}
                                size="sm"
                                className="text-xs h-8 gap-2"
                              >
                                <Play className="w-3.5 h-3.5" />
                                Load
                              </Button>
                            )}
                            {model.status === "ready" && (
                              <Button
                                onClick={() => onUnload(model.variant)}
                                variant="outline"
                                size="sm"
                                className="text-xs h-8 gap-2"
                              >
                                <Square className="w-3.5 h-3.5" />
                                Unload
                              </Button>
                            )}
                            {(model.status === "downloaded" ||
                              model.status === "ready") &&
                              (pendingDeleteVariant === model.variant ? (
                                <div className="flex items-center gap-1">
                                  <Button
                                    onClick={() =>
                                      setPendingDeleteVariant(null)
                                    }
                                    variant="outline"
                                    size="sm"
                                    className="text-xs h-8 gap-2"
                                  >
                                    <X className="w-3.5 h-3.5" />
                                    Cancel
                                  </Button>
                                  <Button
                                    onClick={() =>
                                      handleConfigDelete(model.variant)
                                    }
                                    variant="destructive"
                                    size="sm"
                                    className="text-xs h-8 gap-2"
                                  >
                                    <Trash2 className="w-3.5 h-3.5" />
                                    Confirm Delete
                                  </Button>
                                </div>
                              ) : (
                                <Button
                                  onClick={() =>
                                    setPendingDeleteVariant(model.variant)
                                  }
                                  variant="destructive"
                                  size="sm"
                                  className="text-xs h-8 gap-2 opacity-80 hover:opacity-100"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                  Delete
                                </Button>
                              ))}
                          </div>
                        </div>

                        {model.status === "downloading" && (
                          <div className="mt-2">
                            <div className="h-1.5 rounded bg-[#1f1f1f] overflow-hidden">
                              <div
                                className="h-full rounded bg-white transition-all duration-300"
                                style={{ width: `${progress}%` }}
                              />
                            </div>
                            <div className="mt-1 text-[11px] text-gray-500">
                              Downloading {Math.round(progress)}%
                              {progressValue &&
                                progressValue.totalBytes > 0 && (
                                  <>
                                    {" "}
                                    (
                                    {formatBytes(
                                      progressValue.downloadedBytes,
                                    )}{" "}
                                    / {formatBytes(progressValue.totalBytes)})
                                  </>
                                )}
                            </div>
                            {progressValue?.currentFile && (
                              <div className="mt-0.5 text-[11px] text-gray-600 truncate">
                                {progressValue.currentFile}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </section>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <audio
        ref={audioRef}
        className="hidden"
        onEnded={() => {
          clearAudioPlayback();
          if (isSessionActiveRef.current && !processingRef.current) {
            setRuntimeStatus("listening");
          } else if (!isSessionActiveRef.current) {
            setRuntimeStatus("idle");
          }
        }}
      />
    </div>
  );
}
