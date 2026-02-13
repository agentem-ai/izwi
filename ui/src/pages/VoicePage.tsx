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

import { api, ChatMessage, ModelInfo } from "../api";
import { SPEAKERS, VIEW_CONFIGS } from "../types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { Slider } from "../components/ui/slider";

type RuntimeStatus =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "assistant_speaking";

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

const SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Keep responses brief unless asked for details.",
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
  return variant.includes("Qwen3-ASR") || variant.includes("Voxtral");
}

function isTextVariant(variant: string): boolean {
  return VIEW_CONFIGS.chat.modelFilter(variant);
}

function isTtsVariant(variant: string): boolean {
  return variant.includes("Qwen3-TTS") && !variant.includes("Tokenizer");
}

function isCustomVoiceTtsVariant(variant: string): boolean {
  return isTtsVariant(variant) && variant.includes("CustomVoice");
}

function formatModelVariantLabel(variant: string): string {
  const normalized = variant
    .replace(/-4bit\b/g, "-4-bit")
    .replace(/-8bit\b/g, "-8-bit");

  if (normalized.startsWith("Qwen3-ASR-")) {
    return normalized.replace("Qwen3-ASR-", "ASR ");
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

  return normalized.replace(/-/g, " ");
}

function isRunnableModelStatus(status: ModelInfo["status"]): boolean {
  return status === "ready";
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
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
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);

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
  const [pendingDeleteVariant, setPendingDeleteVariant] = useState<string | null>(
    null,
  );

  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadTimerRef = useRef<number | null>(null);
  const speechStartRef = useRef<number | null>(null);
  const silenceMsRef = useRef(0);
  const processingRef = useRef(false);
  const runtimeStatusRef = useRef<RuntimeStatus>("idle");
  const conversationRef = useRef<ChatMessage[]>([]);
  const isSessionActiveRef = useRef(false);
  const turnIdRef = useRef(0);

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

  useEffect(() => {
    if (!isConfigOpen) {
      setPendingDeleteVariant(null);
    }
  }, [isConfigOpen]);

  const handleConfigDelete = useCallback(
    (variant: string) => {
      setPendingDeleteVariant(null);
      onDelete(variant);
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
    [onDelete, selectedAsrModel, selectedTextModel, selectedTtsModel],
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
    runtimeStatusRef.current = runtimeStatus;
  }, [runtimeStatus]);

  useEffect(() => {
    conversationRef.current = conversation;
  }, [conversation]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, runtimeStatus]);

  useEffect(() => {
    if (
      !selectedAsrModel ||
      !asrModels.some((m) => m.variant === selectedAsrModel)
    ) {
      const preferredAsr =
        asrModels.find(
          (m) => m.variant === "Qwen3-ASR-0.6B-4bit" && m.status === "ready",
        ) ||
        asrModels.find(
          (m) =>
            m.variant.includes("Qwen3-ASR-0.6B") &&
            m.variant.includes("4bit") &&
            m.status === "ready",
        ) ||
        asrModels.find((m) => m.status === "ready") ||
        asrModels.find((m) => m.variant === "Qwen3-ASR-0.6B-4bit") ||
        asrModels.find(
          (m) =>
            m.variant.includes("Qwen3-ASR-0.6B") &&
            m.variant.includes("4bit"),
        ) ||
        asrModels[0];
      setSelectedAsrModel(preferredAsr?.variant ?? null);
    }
  }, [asrModels, selectedAsrModel]);

  useEffect(() => {
    if (
      !selectedTextModel ||
      !textModels.some((m) => m.variant === selectedTextModel)
    ) {
      const preferredText =
        textModels.find(
          (m) => m.variant === "Qwen3-0.6B-4bit" && m.status === "ready",
        ) ||
        textModels.find((m) => m.status === "ready") ||
        textModels.find((m) => m.variant === "Qwen3-0.6B-4bit") ||
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
            m.variant.includes("0.6B") &&
            m.variant.includes("4bit") &&
            m.status === "ready",
        ) ||
        ttsConfigModels.find((m) => m.status === "ready") ||
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

  const selectedAsrInfo = useMemo(
    () => asrModels.find((m) => m.variant === selectedAsrModel) ?? null,
    [asrModels, selectedAsrModel],
  );
  const selectedTextInfo = useMemo(
    () => textModels.find((m) => m.variant === selectedTextModel) ?? null,
    [textModels, selectedTextModel],
  );
  const selectedTtsInfo = useMemo(
    () => ttsConfigModels.find((m) => m.variant === selectedTtsModel) ?? null,
    [ttsConfigModels, selectedTtsModel],
  );

  const hasRunnableConfig = useMemo(
    () =>
      !!selectedAsrInfo &&
      !!selectedTextInfo &&
      !!selectedTtsInfo &&
      isRunnableModelStatus(selectedAsrInfo.status) &&
      isRunnableModelStatus(selectedTextInfo.status) &&
      isRunnableModelStatus(selectedTtsInfo.status),
    [selectedAsrInfo, selectedTextInfo, selectedTtsInfo],
  );

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

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;

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

    analyserRef.current = null;
    clearAudioPlayback();
  }, [clearAudioPlayback]);

  useEffect(() => {
    return () => stopSession();
  }, [stopSession]);

  const appendTranscriptEntry = useCallback((entry: TranscriptEntry) => {
    setTranscript((prev) => [...prev, entry]);
  }, []);

  const setTranscriptEntryText = useCallback((entryId: string, text: string) => {
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
  }, []);

  const removeTranscriptEntry = useCallback((entryId: string) => {
    setTranscript((prev) => prev.filter((entry) => entry.id !== entryId));
  }, []);

  const streamUserTranscription = useCallback(
    (audioBase64: string, modelId: string): Promise<string> =>
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
            audio_base64: audioBase64,
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

  const streamAssistantResponse = useCallback(
    (messages: ChatMessage[], modelId: string): Promise<string> =>
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

        chatStreamAbortRef.current = api.chatCompletionsStream(
          {
            model_id: modelId,
            messages,
            max_tokens: 1536,
          },
          {
            onDelta: (delta) => {
              rawText += delta;
              updateVisibleText();
            },
            onDone: (message) => {
              settle(() => {
                chatStreamAbortRef.current = null;
                if (message) {
                  rawText = message;
                }

                const finalText = parseFinalAnswer(rawText) || rawText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                resolve(finalText);
              });
            },
            onError: (errorMessage) => {
              settle(() => {
                chatStreamAbortRef.current = null;
                const finalText = parseFinalAnswer(rawText) || rawText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                reject(new Error(errorMessage));
              });
            },
          },
        );
      }),
    [appendTranscriptEntry, removeTranscriptEntry, setTranscriptEntryText],
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

  const processUtterance = useCallback(
    async (audioBlob: Blob) => {
      if (!isSessionActiveRef.current) {
        return;
      }

      if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
        setError(
          "Select ASR, text, and TTS models before starting voice mode.",
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
        const wavBlob = await transcodeToWav(audioBlob, 16000);
        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        const audioBase64 = await blobToBase64(wavBlob);
        const userText = await streamUserTranscription(
          audioBase64,
          selectedAsrModel,
        );

        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        if (!userText) {
          processingRef.current = false;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("listening");
          }
          return;
        }

        const requestMessages: ChatMessage[] = [
          SYSTEM_PROMPT,
          ...conversationRef.current,
          { role: "user", content: userText },
        ];

        const assistantText = await streamAssistantResponse(
          requestMessages,
          selectedTextModel,
        );
        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        if (!assistantText) {
          processingRef.current = false;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("listening");
          }
          return;
        }

        setConversation((prev) => [
          ...prev,
          { role: "user", content: userText },
          { role: "assistant", content: assistantText },
        ]);

        await streamAssistantSpeech(
          assistantText,
          selectedTtsModel,
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
      hasRunnableConfig,
      onError,
      selectedAsrModel,
      selectedSpeaker,
      selectedTextModel,
      selectedTtsModel,
      streamAssistantResponse,
      streamAssistantSpeech,
      streamUserTranscription,
    ],
  );

  const startSession = useCallback(async () => {
    if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
      const message =
        "Select ASR, text, and TTS models before starting voice mode.";
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

      mediaRecorderRef.current = recorder;
      isSessionActiveRef.current = true;
      processingRef.current = false;
      silenceMsRef.current = 0;
      speechStartRef.current = null;
      setRuntimeStatus("listening");

      const VAD_INTERVAL = 80;
      vadTimerRef.current = window.setInterval(() => {
        const analyserNode = analyserRef.current;
        const recorderNode = mediaRecorderRef.current;
        if (!analyserNode || !recorderNode || !isSessionActiveRef.current)
          return;

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
        const isRecording = recorderNode.state === "recording";
        const now = Date.now();

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
    hasRunnableConfig,
    minSpeechMs,
    onError,
    processUtterance,
    selectedAsrModel,
    selectedTextModel,
    selectedTtsModel,
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
        return "bg-emerald-500/15 border-emerald-500/40 text-emerald-300";
      case "loading":
      case "downloading":
        return "bg-sky-500/15 border-sky-500/40 text-sky-300";
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
            Low-latency speech loop with configurable ASR, LLM, and TTS.
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

            <button
              onClick={toggleSession}
              className={clsx(
                "btn w-full mt-5 text-sm min-h-[46px]",
                runtimeStatus === "idle" ? "btn-primary" : "btn-danger",
              )}
              disabled={
                !selectedAsrModel ||
                !selectedTextModel ||
                !selectedTtsModel ||
                !hasRunnableConfig
              }
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
            </button>
          </div>

          <div className="mt-5 pt-4 border-t border-[#252525] space-y-3">
            <div className="h-2 rounded bg-[#1b1b1b] border border-[#2a2a2a] overflow-hidden">
              <div
                className="h-full bg-white transition-all duration-75"
                style={{ width: `${vadPercent}%` }}
              />
            </div>
            <div className="space-y-2 text-xs">
              {[
                { label: "ASR", model: selectedAsrInfo },
                { label: "Text", model: selectedTextInfo },
                { label: "TTS", model: selectedTtsInfo },
              ].map((item) => (
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
                    <span className="text-amber-400">Not selected</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="card p-4 flex flex-col min-h-[420px] sm:min-h-[520px] lg:min-h-[640px]">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm text-white font-medium">Conversation</span>
            <span className="text-xs px-2 py-1 rounded bg-[#1a1a1a] border border-[#2a2a2a] text-gray-300">
              {statusLabel}
            </span>
          </div>

          <div className="flex-1 overflow-y-auto pr-1 space-y-3">
            {transcript.length === 0 ? (
              <div className="h-full flex items-center justify-center text-center">
                <div>
                  <p className="text-sm text-gray-400">
                    No conversation yet.
                  </p>
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
                <button
                  className="btn btn-ghost text-xs"
                  onClick={() => setIsConfigOpen(false)}
                >
                  <X className="w-3.5 h-3.5" />
                  Close
                </button>
              </div>

              <div className="p-4 sm:p-5 overflow-y-auto max-h-[calc(90vh-88px)] space-y-5">
                <section className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="text-sm font-medium text-white">
                      Pipeline Selection
                    </h3>
                    <span className="text-[11px] text-gray-500">
                      Assign one model to each stage.
                    </span>
                  </div>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3 space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <label className="text-xs text-gray-400">ASR Model</label>
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
                          {asrModels.map((m) => (
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
                        <label className="text-xs text-gray-400">Text Model</label>
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
                        <label className="text-xs text-gray-400">TTS Model</label>
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
                      <label className="text-xs text-gray-400">Assistant Voice</label>
                      <Select value={selectedSpeaker} onValueChange={setSelectedSpeaker}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select assistant voice" />
                        </SelectTrigger>
                        <SelectContent>
                          {SPEAKERS.map((speaker) => (
                            <SelectItem key={speaker.id} value={speaker.id}>
                              {speaker.name} ({speaker.language})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
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
                    <h3 className="text-sm font-medium text-white">Model Library</h3>
                    <span className="text-[11px] text-gray-500">
                      Download, load, unload, and delete
                    </span>
                  </div>

                  {voiceRouteModels.map((model) => {
                    const roles = getModelRoles(model.variant);
                    const progressValue = downloadProgress[model.variant];
                    const progress =
                      progressValue?.percent ?? model.download_progress ?? 0;
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
                              <span
                                className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2f2f2f] text-gray-300 whitespace-nowrap"
                              >
                                {getStatusLabel(model.status)}
                              </span>
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
                            {model.status === "downloading" && onCancelDownload && (
                              <button
                                onClick={() => onCancelDownload(model.variant)}
                                className="btn btn-danger text-xs"
                              >
                                <X className="w-3.5 h-3.5" />
                                Cancel
                              </button>
                            )}
                            {(model.status === "not_downloaded" ||
                              model.status === "error") && (
                              requiresManualDownload(model.variant) ? (
                                <button
                                  className="btn btn-secondary text-xs"
                                  disabled
                                  title="Manual download required. See docs/user/manual-gemma-3-1b-download.md."
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  Manual download
                                </button>
                              ) : (
                                <button
                                  onClick={() => onDownload(model.variant)}
                                  className="btn btn-primary text-xs"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  Download
                                </button>
                              )
                            )}
                            {model.status === "downloaded" && (
                              <button
                                onClick={() => onLoad(model.variant)}
                                className="btn btn-primary text-xs"
                              >
                                <Play className="w-3.5 h-3.5" />
                                Load
                              </button>
                            )}
                            {model.status === "ready" && (
                              <button
                                onClick={() => onUnload(model.variant)}
                                className="btn btn-secondary text-xs"
                              >
                                <Square className="w-3.5 h-3.5" />
                                Unload
                              </button>
                            )}
                            {(model.status === "downloaded" ||
                              model.status === "ready") && (
                              pendingDeleteVariant === model.variant ? (
                                <div className="flex items-center gap-1">
                                  <button
                                    onClick={() => setPendingDeleteVariant(null)}
                                    className="btn btn-secondary text-xs"
                                  >
                                    <X className="w-3.5 h-3.5" />
                                    Cancel
                                  </button>
                                  <button
                                    onClick={() => handleConfigDelete(model.variant)}
                                    className="btn text-xs border border-red-500/45 bg-red-500/15 text-red-300 hover:bg-red-500/25 hover:text-red-200"
                                  >
                                    <Trash2 className="w-3.5 h-3.5" />
                                    Confirm Delete
                                  </button>
                                </div>
                              ) : (
                                <button
                                  onClick={() =>
                                    setPendingDeleteVariant(model.variant)
                                  }
                                  className="btn text-xs border border-red-500/45 bg-red-500/15 text-red-300 hover:bg-red-500/25 hover:text-red-200"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                  Delete
                                </button>
                              )
                            )}
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
                              {progressValue && progressValue.totalBytes > 0 && (
                                <>
                                  {" "}
                                  ({formatBytes(progressValue.downloadedBytes)} /{" "}
                                  {formatBytes(progressValue.totalBytes)})
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
