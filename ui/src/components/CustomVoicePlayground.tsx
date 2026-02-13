import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Volume2,
  Square,
  Download,
  RotateCcw,
  ChevronDown,
  Loader2,
  MessageSquare,
  Radio,
  Settings2,
} from "lucide-react";
import { api, TTSGenerationStats } from "../api";
import { LFM2_SPEAKERS, QWEN_SPEAKERS } from "../types";
import { GenerationStats } from "./GenerationStats";
import clsx from "clsx";

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

function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
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
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
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

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const audioUrlRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const playbackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextPlaybackTimeRef = useRef(0);
  const streamSampleRateRef = useRef(24000);
  const streamSamplesRef = useRef<Float32Array[]>([]);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const isLfm2Model = selectedModel?.toLowerCase().includes("lfm2-audio") ?? false;
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
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }
    audioUrlRef.current = nextUrl;
    setAudioUrl(nextUrl);
  }, []);

  const stopStreamingSession = useCallback(() => {
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
    streamSamplesRef.current = [];
  }, []);

  useEffect(() => {
    return () => {
      stopStreamingSession();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
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

    try {
      setGenerating(true);
      setIsStreaming(false);
      setError(null);
      setGenerationStats(null);
      stopStreamingSession();
      replaceAudioUrl(null);

      const request = {
        text: text.trim(),
        model_id: selectedModel,
        max_tokens: 0,
        speaker,
        voice_description: instruct.trim() || undefined,
      };

      if (!streamingEnabled) {
        const result = await api.generateTTSWithStats(request);
        const url = URL.createObjectURL(result.audioBlob);
        replaceAudioUrl(url);
        setGenerationStats(result.stats);

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
      streamSamplesRef.current = [];
      setIsStreaming(true);

      streamAbortRef.current = api.generateTTSStream(
        {
          ...request,
          format: "pcm",
        },
        {
          onStart: ({ sampleRate, audioFormat }) => {
            streamSampleRateRef.current = sampleRate;
            if (audioFormat !== "pcm_i16") {
              setError(
                `Unsupported streamed audio format '${audioFormat}'. Expected pcm_i16.`,
              );
            }
          },
          onChunk: ({ audioBase64 }) => {
            const context = playbackContextRef.current;
            if (!context) return;

            const samples = decodePcmI16Base64(audioBase64);
            if (samples.length === 0) return;
            streamSamplesRef.current.push(samples);

            const buffer = context.createBuffer(
              1,
              samples.length,
              streamSampleRateRef.current,
            );
            const chunkForPlayback = new Float32Array(samples.length);
            chunkForPlayback.set(samples);
            buffer.copyToChannel(chunkForPlayback, 0);

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
          },
          onFinal: (stats) => {
            setGenerationStats(stats);
          },
          onError: (errorMessage) => {
            setError(errorMessage);
          },
          onDone: () => {
            streamAbortRef.current = null;
            setIsStreaming(false);
            setGenerating(false);

            const merged = mergeSampleChunks(streamSamplesRef.current);
            if (merged.length > 0) {
              const wavBlob = encodeWavPcm16(merged, streamSampleRateRef.current);
              const url = URL.createObjectURL(wavBlob);
              replaceAudioUrl(url);
            }
          },
        },
      );
    } catch (err) {
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

  const handleDownload = () => {
    if (audioUrl) {
      const a = document.createElement("a");
      a.href = audioUrl;
      a.download = `izwi-${speaker.toLowerCase()}-${Date.now()}.wav`;
      a.click();
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
                  className="btn btn-secondary min-h-[44px] min-w-[44px]"
                >
                  <Download className="w-4 h-4" />
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
  );
}
