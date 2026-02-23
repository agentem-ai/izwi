import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Volume2,
  Square,
  Download,
  RotateCcw,
  ChevronDown,
  Loader2,
  CheckCircle2,
  AlertCircle,
  MessageSquare,
  Radio,
  Settings2,
} from "lucide-react";
import {
  api,
  type SpeechHistoryRecord,
  type TTSGenerationStats,
} from "../api";
import { isLfmAudioVariant, LFM2_SPEAKERS, QWEN_SPEAKERS } from "../types";
import { GenerationStats } from "./GenerationStats";
import clsx from "clsx";
import { SpeechHistoryPanel } from "./SpeechHistoryPanel";
import { useDownloadIndicator } from "../utils/useDownloadIndicator";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface CustomVoicePlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelLabel?: string | null;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
}

const MAX_BUFFERED_PCM_BYTES = 256 * 1024 * 1024;
const ABORT_ERROR_NAME = "AbortError";

function createAbortError(message: string): Error {
  const error = new Error(message);
  error.name = ABORT_ERROR_NAME;
  return error;
}

function decodePcmI16Base64(base64Data: string): Int16Array {
  const binary = atob(base64Data);
  const sampleCount = Math.floor(binary.length / 2);
  const out = new Int16Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const lo = binary.charCodeAt(i * 2);
    const hi = binary.charCodeAt(i * 2 + 1);
    let value = (hi << 8) | lo;
    if (value & 0x8000) {
      value -= 0x10000;
    }
    out[i] = value;
  }

  return out;
}

function pcmI16ToFloat32(samples: Int16Array): Float32Array<ArrayBuffer> {
  const floatSamples = new Float32Array(
    samples.length,
  ) as Float32Array<ArrayBuffer>;
  for (let i = 0; i < samples.length; i += 1) {
    floatSamples[i] = samples[i] / 0x8000;
  }
  return floatSamples;
}

function wavHeader(sampleRate: number, dataSize: number): Uint8Array {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44);
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
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  return new Uint8Array(buffer, 0, 44);
}

function copyToArrayBuffer(view: Uint8Array): ArrayBuffer {
  const copied = new Uint8Array(view.byteLength);
  copied.set(view);
  return copied.buffer;
}

function encodeWavPcm16Chunks(
  sampleRate: number,
  pcmChunks: Uint8Array[],
  totalPcmBytes: number,
): Blob {
  const parts: BlobPart[] = [copyToArrayBuffer(wavHeader(sampleRate, totalPcmBytes))];
  for (const chunk of pcmChunks) {
    parts.push(copyToArrayBuffer(chunk));
  }

  return new Blob(parts, {
    type: "audio/wav",
  });
}

function revokeObjectUrlIfNeeded(url: string | null): void {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

function mapRecordToStats(record: SpeechHistoryRecord): TTSGenerationStats {
  return {
    generation_time_ms: record.generation_time_ms,
    audio_duration_secs: record.audio_duration_secs ?? 0,
    rtf: record.rtf ?? 0,
    tokens_generated: record.tokens_generated ?? 0,
  };
}

export function CustomVoicePlayground({
  selectedModel,
  selectedModelReady = false,
  modelLabel,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
}: CustomVoicePlaygroundProps) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("Vivian");
  const [instruct, setInstruct] = useState("");
  const [showSpeakerSelect, setShowSpeakerSelect] = useState(false);
  const [showInstruct, setShowInstruct] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [generationStats, setGenerationStats] =
    useState<TTSGenerationStats | null>(null);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [latestRecord, setLatestRecord] = useState<SpeechHistoryRecord | null>(null);
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const audioUrlRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const playbackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextPlaybackTimeRef = useRef(0);
  const streamSampleRateRef = useRef(24000);
  const streamPcmChunksRef = useRef<Uint8Array[]>([]);
  const bufferedPcmBytesRef = useRef(0);
  const mergeSuppressedRef = useRef(false);
  const generationSessionRef = useRef(0);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const isLfm2Model = selectedModel ? isLfmAudioVariant(selectedModel) : false;
  const availableSpeakers = useMemo(
    () => (isLfm2Model ? LFM2_SPEAKERS : QWEN_SPEAKERS),
    [isLfm2Model],
  );
  const defaultSpeaker = isLfm2Model ? "US Female" : "Vivian";

  const selectedSpeaker = availableSpeakers.find((s) => s.id === speaker);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return modelOptions.find((option) => option.value === selectedModel) || null;
  }, [selectedModel, modelOptions]);

  useEffect(() => {
    if (!availableSpeakers.some((candidate) => candidate.id === speaker)) {
      setSpeaker(defaultSpeaker);
    }
  }, [availableSpeakers, defaultSpeaker, speaker]);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (
        modelMenuRef.current &&
        event.target instanceof Node &&
        !modelMenuRef.current.contains(event.target)
      ) {
        setIsModelMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, []);

  const replaceAudioUrl = useCallback((nextUrl: string | null) => {
    revokeObjectUrlIfNeeded(audioUrlRef.current);
    audioUrlRef.current = nextUrl;
    setAudioUrl(nextUrl);
  }, []);

  const stopStreamingSession = useCallback(() => {
    generationSessionRef.current += 1;
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }

    for (const source of playbackSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // Ignore already-stopped sources.
      }
    }
    playbackSourcesRef.current.clear();

    if (playbackContextRef.current) {
      playbackContextRef.current.close().catch(() => {});
      playbackContextRef.current = null;
    }

    nextPlaybackTimeRef.current = 0;
    streamPcmChunksRef.current = [];
    bufferedPcmBytesRef.current = 0;
    mergeSuppressedRef.current = false;
  }, []);

  useEffect(() => {
    return () => {
      stopStreamingSession();
      revokeObjectUrlIfNeeded(audioUrlRef.current);
      audioUrlRef.current = null;
    };
  }, [stopStreamingSession]);

  const handleGenerate = async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      setError("Please enter some text");
      return;
    }

    const trimmedText = text.trim();

    try {
      setGenerating(true);
      setIsStreaming(false);
      setError(null);
      setGenerationStats(null);
      stopStreamingSession();
      replaceAudioUrl(null);
      const generationSession = generationSessionRef.current;

      const requestBase = {
        model_id: selectedModel,
        max_tokens: 0,
        speaker,
        voice_description: instruct.trim() || undefined,
      };

      if (!streamingEnabled) {
        const record = await api.createTextToSpeechRecord({
          ...requestBase,
          text: trimmedText,
        });
        replaceAudioUrl(api.textToSpeechRecordAudioUrl(record.id));
        setGenerationStats(mapRecordToStats(record));
        setLatestRecord(record);

        setTimeout(() => {
          audioRef.current?.play().catch(() => {});
        }, 100);

        setGenerating(false);
        return;
      }

      const audioContext = new AudioContext();
      playbackContextRef.current = audioContext;
      nextPlaybackTimeRef.current = audioContext.currentTime + 0.05;
      streamSampleRateRef.current = 24000;
      streamPcmChunksRef.current = [];
      bufferedPcmBytesRef.current = 0;
      mergeSuppressedRef.current = false;
      setIsStreaming(streamingEnabled);

      const appendPcmChunk = (pcmSamples: Int16Array) => {
        if (mergeSuppressedRef.current) {
          return;
        }
        const pcmBytes = new Uint8Array(pcmSamples.buffer);
        const nextTotal = bufferedPcmBytesRef.current + pcmBytes.byteLength;
        if (nextTotal > MAX_BUFFERED_PCM_BYTES) {
          mergeSuppressedRef.current = true;
          streamPcmChunksRef.current = [];
          bufferedPcmBytesRef.current = 0;
          return;
        }
        streamPcmChunksRef.current.push(pcmBytes);
        bufferedPcmBytesRef.current = nextTotal;
      };

      const schedulePlayback = (pcmSamples: Int16Array) => {
        if (!streamingEnabled) {
          return;
        }
        const context = playbackContextRef.current;
        if (!context) {
          return;
        }

        const floatSamples = pcmI16ToFloat32(pcmSamples);
        const buffer = context.createBuffer(
          1,
          floatSamples.length,
          streamSampleRateRef.current,
        );
        buffer.copyToChannel(floatSamples, 0);

        const source = context.createBufferSource();
        source.buffer = buffer;
        source.connect(context.destination);

        const scheduledAt = Math.max(
          context.currentTime + 0.02,
          nextPlaybackTimeRef.current,
        );
        source.start(scheduledAt);
        nextPlaybackTimeRef.current = scheduledAt + buffer.duration;

        playbackSourcesRef.current.add(source);
        source.onended = () => {
          playbackSourcesRef.current.delete(source);
        };

        if (context.state === "suspended") {
          context.resume().catch(() => {});
        }
      };

      const finalRecordRef = { current: null as SpeechHistoryRecord | null };
      const finalStatsRef = { current: null as TTSGenerationStats | null };

      const streamRequest = (): Promise<void> =>
        new Promise((resolve, reject) => {
          let settled = false;
          const resolveOnce = () => {
            if (settled) return;
            settled = true;
            resolve();
          };
          const rejectOnce = (error: Error) => {
            if (settled) return;
            settled = true;
            reject(error);
          };

          streamAbortRef.current = api.createTextToSpeechRecordStream(
            {
              ...requestBase,
              text: trimmedText,
            },
            {
              onStart: ({ sampleRate, audioFormat }) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                streamSampleRateRef.current = sampleRate;
                if (audioFormat !== "pcm_i16") {
                  const message = `Unsupported streamed audio format '${audioFormat}'. Expected pcm_i16.`;
                  setError(message);
                  streamAbortRef.current?.abort();
                  rejectOnce(new Error(message));
                }
              },
              onChunk: ({ audioBase64 }) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                const pcmSamples = decodePcmI16Base64(audioBase64);
                if (pcmSamples.length === 0) {
                  return;
                }
                appendPcmChunk(pcmSamples);
                schedulePlayback(pcmSamples);
              },
              onFinal: ({ record, stats }) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                finalRecordRef.current = record;
                finalStatsRef.current = stats;
              },
              onError: (errorMessage) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                setError(errorMessage);
                rejectOnce(new Error(errorMessage));
              },
              onDone: () => {
                streamAbortRef.current = null;
                resolveOnce();
              },
            },
          );
        });

      if (generationSession !== generationSessionRef.current) {
        throw createAbortError("Generation cancelled");
      }
      await streamRequest();

      if (generationSession !== generationSessionRef.current) {
        throw createAbortError("Generation cancelled");
      }

      if (finalStatsRef.current) {
        setGenerationStats(finalStatsRef.current);
      }
      const finalRecord = finalRecordRef.current;
      if (finalRecord) {
        setLatestRecord(finalRecord);
      }
      const finalRecordId = finalRecord?.id;

      if (
        !mergeSuppressedRef.current &&
        bufferedPcmBytesRef.current > 0 &&
        streamPcmChunksRef.current.length > 0
      ) {
        const wavBlob = encodeWavPcm16Chunks(
          streamSampleRateRef.current,
          streamPcmChunksRef.current,
          bufferedPcmBytesRef.current,
        );
        replaceAudioUrl(URL.createObjectURL(wavBlob));
      } else if (finalRecordId) {
        replaceAudioUrl(api.textToSpeechRecordAudioUrl(finalRecordId));
      }

      setIsStreaming(false);
      setGenerating(false);
    } catch (err) {
      if ((err as Error).name === ABORT_ERROR_NAME) {
        setGenerating(false);
        setIsStreaming(false);
        return;
      }
      setError(err instanceof Error ? err.message : "Generation failed");
      setGenerating(false);
      setIsStreaming(false);
    }
  };

  const handleStop = () => {
    stopStreamingSession();
    setGenerating(false);
    setIsStreaming(false);

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const handleDownload = async () => {
    const record = latestRecord;
    const localAudioUrl = !record ? audioUrl : null;
    if ((!record && !localAudioUrl) || isDownloading) {
      return;
    }

    beginDownload();
    try {
      if (record) {
        const downloadUrl = api.textToSpeechRecordAudioUrl(record.id, {
          download: true,
        });
        const filename =
          record.audio_filename || `izwi-${speaker.toLowerCase()}-${Date.now()}.wav`;
        await api.downloadAudioFile(downloadUrl, filename);
        completeDownload();
        return;
      }

      if (!localAudioUrl) {
        return;
      }
      await api.downloadAudioFile(
        localAudioUrl,
        `izwi-${speaker.toLowerCase()}-${Date.now()}.wav`,
      );
      completeDownload();
    } catch (error) {
      failDownload(error);
    }
  };

  const handleReset = () => {
    stopStreamingSession();
    setText("");
    setInstruct("");
    setError(null);
    setGenerationStats(null);
    setGenerating(false);
    setIsStreaming(false);
    replaceAudioUrl(null);
    textareaRef.current?.focus();
  };

  const getStatusTone = (option: ModelOption): string => {
    if (option.isReady) {
      return "text-[var(--text-primary)] bg-[var(--bg-surface-3)] border border-[var(--border-strong)]";
    }
    if (
      option.statusLabel.toLowerCase().includes("downloading") ||
      option.statusLabel.toLowerCase().includes("loading")
    ) {
      return "text-[var(--text-secondary)] bg-[var(--bg-surface-3)] border border-[var(--border-strong)]";
    }
    if (option.statusLabel.toLowerCase().includes("error")) {
      return "text-[var(--danger-text)] bg-[var(--danger-bg)] border border-[var(--danger-border)]";
    }
    return "text-[var(--text-muted)] bg-[var(--bg-surface-2)] border border-[var(--border-strong)]";
  };

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager?.();
  };

  const renderModelSelector = () => (
    <div className="relative inline-block w-[280px] max-w-[85vw]" ref={modelMenuRef}>
      <button
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={clsx(
          "h-9 w-full px-3 rounded-lg border inline-flex items-center justify-between gap-2 text-xs transition-colors",
          selectedOption?.isReady
            ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
            : "border-[var(--border-strong)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown className={clsx("w-3.5 h-3.5 shrink-0 transition-transform", isModelMenuOpen && "rotate-180")} />
      </button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 top-full mt-2 rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-1.5 shadow-2xl z-50"
          >
            <div className="max-h-64 overflow-y-auto pr-1 space-y-0.5">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel?.(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={clsx(
                    "w-full text-left rounded-lg px-3 py-2 transition-colors",
                    selectedOption?.value === option.value
                      ? "bg-[var(--bg-surface-3)]"
                      : "hover:bg-[var(--bg-surface-2)]",
                  )}
                >
                  <div className="text-xs text-[var(--text-primary)] truncate">
                    {option.label}
                  </div>
                  <span
                    className={clsx(
                      "mt-1 inline-flex items-center rounded px-1.5 py-0.5 text-[10px]",
                      getStatusTone(option),
                    )}
                  >
                    {option.statusLabel}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr),320px] items-stretch">
      <div className="card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded bg-[var(--bg-surface-2)] border border-[var(--border-strong)]">
            <Volume2 className="w-5 h-5 text-[var(--text-muted)]" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-[var(--text-primary)]">Text to Speech</h2>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative">
            <button
              onClick={() => setShowSpeakerSelect(!showSpeakerSelect)}
              className="flex w-56 sm:w-64 items-center justify-between gap-2 px-3 py-1.5 rounded bg-[var(--bg-surface-2)] border border-[var(--border-strong)] hover:bg-[var(--bg-surface-3)] text-sm"
            >
              <div className="speaker-avatar w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-medium">
                {speaker.charAt(0)}
              </div>
              <span className="text-[var(--text-primary)] flex-1 min-w-0 truncate text-left">
                {selectedSpeaker?.name || speaker}
              </span>
              <ChevronDown
                className={clsx(
                  "w-3.5 h-3.5 text-[var(--text-subtle)] transition-transform",
                  showSpeakerSelect && "rotate-180",
                )}
              />
            </button>

            <AnimatePresence>
              {showSpeakerSelect && (
                <motion.div
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className="absolute left-0 right-0 top-full mt-1 max-h-80 overflow-y-auto p-1 rounded bg-[var(--bg-surface-1)] border border-[var(--border-strong)] shadow-xl z-50"
                >
                  {availableSpeakers.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => {
                        setSpeaker(s.id);
                        setShowSpeakerSelect(false);
                      }}
                      className={clsx(
                        "w-full px-3 py-2 rounded text-left transition-colors flex items-center gap-3",
                        speaker === s.id
                          ? "bg-[var(--bg-surface-3)]"
                          : "hover:bg-[var(--bg-surface-2)]",
                      )}
                    >
                      <div className="speaker-avatar w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0">
                        {s.name.charAt(0)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div
                          className={clsx(
                            "text-sm font-medium",
                            speaker === s.id
                              ? "text-[var(--text-primary)]"
                              : "text-[var(--text-secondary)]",
                          )}
                        >
                          {s.name}
                        </div>
                        <div className="text-[10px] text-[var(--text-subtle)] truncate">
                          {s.description}
                        </div>
                      </div>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-surface-3)] text-[var(--text-subtle)]">
                        {s.language}
                      </span>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <div className="mb-4 rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="text-[11px] text-[var(--text-subtle)] uppercase tracking-wide">
              Active Model
            </div>
            <div className="mt-1 text-sm text-[var(--text-primary)] truncate">
              {modelLabel ?? "No model selected"}
            </div>
            <div
              className={clsx(
                "mt-1 text-xs",
                selectedModelReady
                  ? "text-[var(--text-primary)]"
                  : "text-[var(--text-secondary)]",
              )}
            >
              {selectedModelReady
                ? "Loaded and ready"
                : "Open Models and load the selected TTS model"}
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {modelOptions.length > 0 && renderModelSelector()}
            {onOpenModelManager && (
              <button
                onClick={handleOpenModels}
                className="btn btn-secondary text-xs"
              >
                <Settings2 className="w-4 h-4" />
                Models
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Text input */}
      <div className="space-y-3">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to synthesize..."
            rows={6}
            disabled={generating}
            className="textarea text-sm"
          />
          <div className="absolute bottom-2 right-2">
            <span className="text-xs text-[var(--text-subtle)]">{text.length}</span>
          </div>
        </div>

        <div className="flex items-center justify-between rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-2">
          {/* Instruct toggle */}
          <button
            onClick={() => setShowInstruct(!showInstruct)}
            className="flex items-center gap-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            <MessageSquare className="w-3.5 h-3.5" />
            {showInstruct ? "Hide" : "Add"} speaking instructions
            <ChevronDown
              className={clsx(
                "w-3 h-3 transition-transform",
                showInstruct && "rotate-180",
              )}
            />
          </button>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={streamingEnabled}
              onChange={(e) => setStreamingEnabled(e.target.checked)}
              className="app-checkbox w-4 h-4"
              disabled={generating}
            />
            <span className="text-xs text-[var(--text-secondary)] flex items-center gap-1">
              <Radio className="w-3 h-3" />
              Stream
            </span>
          </label>
        </div>

        <AnimatePresence>
          {showInstruct && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="p-3 rounded-lg bg-[var(--bg-surface-2)] border border-[var(--border-strong)]">
                <label className="block text-xs text-[var(--text-muted)] mb-1.5">
                  Speaking Style Instructions
                </label>
                <input
                  type="text"
                  value={instruct}
                  onChange={(e) => setInstruct(e.target.value)}
                  placeholder="e.g., 'Speak with excitement' or 'Very calm and soothing'"
                  className="input text-sm"
                />
                <p className="text-[10px] text-[var(--text-secondary)] mt-1.5">
                  Optional: Guide the emotional tone and speaking style
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="p-2 rounded bg-red-950/50 border border-red-900/50 text-red-400 text-xs"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {isStreaming && (
          <div className="p-2 rounded bg-[var(--bg-surface-2)] border border-[var(--border-strong)] text-[var(--text-secondary)] text-xs flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-secondary)] animate-pulse" />
            Streaming audio chunks...
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
          <button
            onClick={handleGenerate}
            disabled={generating || !selectedModelReady}
            className="btn btn-primary flex-1 min-h-[44px]"
          >
            {generating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {isStreaming ? "Streaming..." : "Generating..."}
              </>
            ) : (
              "Generate"
            )}
          </button>

          {(audioUrl || isStreaming) && (
            <>
              <button
                onClick={handleStop}
                className="btn btn-secondary min-h-[44px] min-w-[44px]"
              >
                <Square className="w-4 h-4" />
              </button>
              {audioUrl && (
                <button
                  onClick={handleDownload}
                  disabled={isDownloading}
                  className={clsx(
                    "btn btn-secondary min-h-[44px] min-w-[44px]",
                    isDownloading && "opacity-75",
                  )}
                >
                  {isDownloading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Download className="w-4 h-4" />
                  )}
                </button>
              )}
              <button
                onClick={handleReset}
                className="btn btn-ghost min-h-[44px] min-w-[44px]"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </>
          )}
        </div>

        <AnimatePresence>
          {downloadState !== "idle" && downloadMessage && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className={clsx(
                "p-2 rounded border text-xs flex items-center gap-2",
                downloadState === "downloading" &&
                  "bg-[var(--bg-surface-2)] border-[var(--border-strong)] text-[var(--text-secondary)]",
                downloadState === "success" &&
                  "bg-emerald-950/40 border-emerald-900/50 text-emerald-300",
                downloadState === "error" &&
                  "bg-red-950/50 border-red-900/50 text-red-400",
              )}
            >
              {downloadState === "downloading" ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : downloadState === "success" ? (
                <CheckCircle2 className="w-3.5 h-3.5" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5" />
              )}
              {downloadMessage}
            </motion.div>
          )}
        </AnimatePresence>

        {!selectedModelReady && (
          <p className="text-xs text-[var(--text-secondary)]">
            Load a compatible Qwen3 CustomVoice or LFM2 model to generate speech
          </p>
        )}
      </div>

      {/* Audio player */}
      <AnimatePresence>
        {audioUrl && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="mt-4 space-y-3"
          >
            <div className="p-3 rounded bg-[var(--bg-surface-2)] border border-[var(--border-strong)]">
              <audio
                ref={audioRef}
                src={audioUrl}
                className="w-full"
                controls
              />
            </div>
            {generationStats && (
              <GenerationStats stats={generationStats} type="tts" />
            )}
          </motion.div>
        )}
      </AnimatePresence>
      </div>

      <SpeechHistoryPanel
        route="text-to-speech"
        title="Speech History"
        emptyMessage="No saved text-to-speech generations yet."
        latestRecord={latestRecord}
      />
    </div>
  );
}
