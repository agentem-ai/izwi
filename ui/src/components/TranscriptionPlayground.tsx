import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Clock3,
  Copy,
  Download,
  FileAudio,
  FileText,
  History,
  Loader2,
  Mic,
  MicOff,
  Radio,
  RefreshCw,
  RotateCcw,
  Settings2,
  Upload,
  ChevronDown,
} from "lucide-react";
import clsx from "clsx";
import {
  api,
  type TranscriptionRecord,
  type TranscriptionRecordSummary,
} from "../api";
import { ASRStats, GenerationStats } from "./GenerationStats";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface TranscriptionPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelLabel?: string | null;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
}

interface ProcessAudioOptions {
  filename?: string;
  transcode?: boolean;
}

const LANGUAGE_OPTIONS = [
  "English",
  "Chinese",
  "Cantonese",
  "Arabic",
  "German",
  "French",
  "Spanish",
  "Portuguese",
  "Indonesian",
  "Italian",
  "Korean",
  "Russian",
  "Thai",
  "Vietnamese",
  "Japanese",
  "Turkish",
  "Hindi",
  "Malay",
  "Dutch",
  "Swedish",
  "Danish",
  "Finnish",
  "Polish",
  "Czech",
  "Filipino",
  "Persian",
  "Greek",
  "Romanian",
  "Hungarian",
  "Macedonian",
];

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

function normalizeTranscript(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function buildTranscriptPreview(text: string, maxChars = 160): string {
  const normalized = normalizeTranscript(text);
  if (!normalized) {
    return "No transcript";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)}...`;
}

function summarizeRecord(record: TranscriptionRecord): TranscriptionRecordSummary {
  return {
    id: record.id,
    created_at: record.created_at,
    model_id: record.model_id,
    language: record.language,
    duration_secs: record.duration_secs,
    processing_time_ms: record.processing_time_ms,
    rtf: record.rtf,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
    transcription_preview: buildTranscriptPreview(record.transcription),
    transcription_chars: Array.from(record.transcription).length,
  };
}

function formatCreatedAt(timestampMs: number): string {
  if (!Number.isFinite(timestampMs)) {
    return "Unknown time";
  }
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown time";
  }
  return value.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatAudioDuration(durationSecs: number | null): string {
  if (durationSecs === null || !Number.isFinite(durationSecs) || durationSecs < 0) {
    return "Unknown length";
  }
  if (durationSecs < 60) {
    return `${durationSecs.toFixed(1)}s`;
  }
  const minutes = Math.floor(durationSecs / 60);
  const seconds = Math.floor(durationSecs % 60);
  return `${minutes}m ${seconds}s`;
}

export function TranscriptionPlayground({
  selectedModel,
  selectedModelReady = false,
  modelLabel,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
}: TranscriptionPlaygroundProps) {
  const [transcription, setTranscription] = useState("");
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [processingStats, setProcessingStats] = useState<ASRStats | null>(null);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("English");
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [historyRecords, setHistoryRecords] = useState<TranscriptionRecordSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedHistoryRecordId, setSelectedHistoryRecordId] = useState<string | null>(null);
  const [selectedHistoryRecord, setSelectedHistoryRecord] = useState<TranscriptionRecord | null>(
    null,
  );
  const [selectedHistoryLoading, setSelectedHistoryLoading] = useState(false);
  const [selectedHistoryError, setSelectedHistoryError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return modelOptions.find((option) => option.value === selectedModel) || null;
  }, [selectedModel, modelOptions]);

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

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return false;
    }
    return true;
  }, [selectedModel, selectedModelReady, onModelRequired]);

  const mergeHistorySummary = useCallback((summary: TranscriptionRecordSummary) => {
    setHistoryRecords((previous) => {
      const next = [summary, ...previous.filter((item) => item.id !== summary.id)];
      next.sort((a, b) => b.created_at - a.created_at);
      return next;
    });
  }, []);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const records = await api.listTranscriptionRecords();
      setHistoryRecords(records);
      setSelectedHistoryRecordId((current) => {
        if (current && records.some((item) => item.id === current)) {
          return current;
        }
        return records[0]?.id ?? null;
      });
    } catch (err) {
      setHistoryError(
        err instanceof Error ? err.message : "Failed to load transcription history.",
      );
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!selectedHistoryRecordId) {
      setSelectedHistoryRecord(null);
      setSelectedHistoryError(null);
      return;
    }

    if (selectedHistoryRecord?.id === selectedHistoryRecordId) {
      return;
    }

    let cancelled = false;
    setSelectedHistoryLoading(true);
    setSelectedHistoryError(null);

    api
      .getTranscriptionRecord(selectedHistoryRecordId)
      .then((record) => {
        if (cancelled) {
          return;
        }
        setSelectedHistoryRecord(record);
        mergeHistorySummary(summarizeRecord(record));
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSelectedHistoryError(
          err instanceof Error
            ? err.message
            : "Failed to load transcription record details.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setSelectedHistoryLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [mergeHistorySummary, selectedHistoryRecord, selectedHistoryRecordId]);

  const processAudio = useCallback(
    async (audioBlob: Blob, options: ProcessAudioOptions = {}) => {
      if (!requireReadyModel()) {
        return;
      }

      setIsProcessing(true);
      setError(null);
      setProcessingStats(null);
      setTranscription("");

      try {
        const shouldTranscode =
          options.transcode ?? !(audioBlob instanceof File);
        const uploadBlob = shouldTranscode
          ? await transcodeToWav(audioBlob, 16000)
          : audioBlob;
        const uploadFilename =
          options.filename?.trim() ||
          (audioBlob instanceof File && audioBlob.name
            ? audioBlob.name
            : "audio.wav");

        const url = URL.createObjectURL(uploadBlob);
        setAudioUrl((previousUrl) => {
          if (previousUrl) {
            URL.revokeObjectURL(previousUrl);
          }
          return url;
        });

        if (streamingEnabled) {
          setIsStreaming(true);
          let finalRecordId: string | null = null;

          streamAbortRef.current = api.createTranscriptionRecordStream(
            {
              audio_file: uploadBlob,
              audio_filename: uploadFilename,
              model_id: selectedModel || undefined,
              language: selectedLanguage,
            },
            {
              onStart: () => {},
              onDelta: (delta) => {
                setTranscription((previous) => `${previous}${delta}`);
              },
              onFinal: (record) => {
                finalRecordId = record.id;
                setTranscription(record.transcription);
                setDetectedLanguage(record.language || null);
                setProcessingStats({
                  processing_time_ms: record.processing_time_ms,
                  audio_duration_secs: record.duration_secs,
                  rtf: record.rtf,
                });
                mergeHistorySummary(summarizeRecord(record));
                setSelectedHistoryRecord(record);
                setSelectedHistoryRecordId(record.id);
                setSelectedHistoryError(null);
              },
              onError: (errorMsg) => {
                setError(errorMsg);
              },
              onDone: () => {
                setIsStreaming(false);
                setIsProcessing(false);
                streamAbortRef.current = null;
                if (!finalRecordId) {
                  void loadHistory();
                }
              },
            },
          );
        } else {
          const record = await api.createTranscriptionRecord({
            audio_file: uploadBlob,
            audio_filename: uploadFilename,
            model_id: selectedModel || undefined,
            language: selectedLanguage,
          });

          setTranscription(record.transcription);
          setDetectedLanguage(record.language || null);
          setProcessingStats({
            processing_time_ms: record.processing_time_ms,
            audio_duration_secs: record.duration_secs,
            rtf: record.rtf,
          });
          mergeHistorySummary(summarizeRecord(record));
          setSelectedHistoryRecord(record);
          setSelectedHistoryRecordId(record.id);
          setSelectedHistoryError(null);
          setIsProcessing(false);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Transcription failed");
        setIsProcessing(false);
        setIsStreaming(false);
      }
    },
    [
      loadHistory,
      mergeHistorySummary,
      requireReadyModel,
      selectedModel,
      selectedLanguage,
      streamingEnabled,
    ],
  );

  const startRecording = useCallback(async () => {
    if (!requireReadyModel()) {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      let mediaRecorder: MediaRecorder | null = null;
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ];
      for (const mimeType of mimeCandidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          mediaRecorder = new MediaRecorder(stream, { mimeType });
          break;
        }
      }
      if (!mediaRecorder) {
        mediaRecorder = new MediaRecorder(stream);
      }
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder?.mimeType || "audio/webm",
        });
        stream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch {
      setError("Could not access microphone. Please grant permission.");
    }
  }, [processAudio, requireReadyModel]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await processAudio(file, {
      filename: file.name,
      transcode: false,
    });
    event.target.value = "";
  };

  const handleReset = () => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setTranscription("");
    setDetectedLanguage(null);
    setAudioUrl(null);
    setError(null);
    setProcessingStats(null);
    setIsStreaming(false);
    setIsProcessing(false);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(transcription);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([transcription], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `transcription-${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
    };
  }, []);

  const canRunInput = !isProcessing && !isRecording && selectedModelReady;
  const showResult = Boolean(transcription || isStreaming || isProcessing);
  const hasDraft = Boolean(transcription || audioUrl || error);
  const selectedHistorySummary = useMemo(
    () =>
      selectedHistoryRecordId
        ? historyRecords.find((record) => record.id === selectedHistoryRecordId) ?? null
        : null,
    [historyRecords, selectedHistoryRecordId],
  );
  const selectedHistoryAudioUrl = useMemo(
    () =>
      selectedHistoryRecordId
        ? api.transcriptionRecordAudioUrl(selectedHistoryRecordId)
        : null,
    [selectedHistoryRecordId],
  );
  const activeHistoryRecord =
    selectedHistoryRecord && selectedHistoryRecord.id === selectedHistoryRecordId
      ? selectedHistoryRecord
      : null;

  const getStatusTone = (option: ModelOption): string => {
    if (option.isReady) {
      return "text-gray-300 bg-white/10";
    }
    if (
      option.statusLabel.toLowerCase().includes("downloading") ||
      option.statusLabel.toLowerCase().includes("loading")
    ) {
      return "text-amber-400 bg-amber-500/10";
    }
    if (option.statusLabel.toLowerCase().includes("error")) {
      return "text-red-400 bg-red-500/10";
    }
    return "text-gray-400 bg-white/5";
  };

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager?.();
  };

  const renderModelSelector = () => (
    <div className="relative w-full" ref={modelMenuRef}>
      <button
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={clsx(
          "h-9 px-3 rounded-lg border w-full flex items-center justify-between gap-2 text-xs transition-colors",
          selectedOption?.isReady
            ? "border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)]"
            : "border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-strong)]",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown className={clsx("w-3.5 h-3.5 transition-transform shrink-0", isModelMenuOpen && "rotate-180")} />
      </button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 top-full mt-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-1.5 shadow-2xl z-50"
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
                      : "hover:bg-[var(--bg-surface-3)]",
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
    <div className="grid gap-4 lg:gap-6 xl:grid-cols-[340px,minmax(0,1fr),320px]">
      <div className="card p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-[var(--text-muted)]">
              <FileAudio className="w-3.5 h-3.5" />
              Capture
            </div>
            <h2 className="text-sm font-medium text-[var(--text-primary)] mt-1">
              Audio Input
            </h2>
          </div>
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

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-4 space-y-3">
          <div>
            <div className="text-[11px] text-[var(--text-subtle)] uppercase tracking-wide mb-2">
              Active Model
            </div>
            {modelOptions.length > 0 && renderModelSelector()}
          </div>

          <div className="pt-2 border-t border-[var(--border-muted)]">
            <div className="text-sm text-[var(--text-primary)] truncate">
              {modelLabel ?? "No model selected"}
            </div>
            <div
              className={clsx(
                "mt-1 text-xs",
                selectedModelReady
                  ? "text-[var(--text-secondary)]"
                  : "text-amber-500",
              )}
            >
              {selectedModelReady
                ? "Loaded and ready"
                : "Select and load a transcription model"}
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
          <div className="flex items-center justify-center">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={clsx(
                "h-24 w-24 rounded-full border transition-all duration-150 flex items-center justify-center",
                isRecording
                  ? "bg-[var(--accent-solid)] border-[var(--accent-solid)] text-[var(--text-on-accent)] shadow-[0_0_0_8px_rgba(255,255,255,0.08)]"
                  : "bg-[var(--bg-surface-2)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-strong)]",
              )}
              disabled={!selectedModelReady || isProcessing}
            >
              {isRecording ? (
                <MicOff className="w-8 h-8" />
              ) : (
                <Mic className="w-8 h-8" />
              )}
            </button>
          </div>
          <p className="text-center text-xs text-[var(--text-subtle)] mt-3">
            {isRecording
              ? "Recording... click again to stop"
              : "Tap to record from microphone"}
          </p>

          <div className="mt-4">
            <button
              onClick={() => {
                if (!requireReadyModel()) {
                  return;
                }
                fileInputRef.current?.click();
              }}
              className="btn btn-secondary w-full text-sm"
              disabled={!canRunInput}
            >
              <Upload className="w-4 h-4" />
              Upload Audio File
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>
        </div>

        {audioUrl && (
          <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-3">
            <div className="text-xs text-[var(--text-subtle)] mb-2">Latest input</div>
            <audio src={audioUrl} controls className="w-full h-9" />
          </div>
        )}

        {hasDraft && (
          <button onClick={handleReset} className="btn btn-ghost w-full text-xs">
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Session
          </button>
        )}
      </div>

      <div className="card p-4 sm:p-5 min-h-[460px] lg:min-h-[560px] flex flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 min-w-0">
              <FileText className="w-4 h-4 text-[var(--text-muted)]" />
              <h3 className="text-sm font-medium text-[var(--text-primary)]">
                Transcript
              </h3>
              {isStreaming && (
                <span className="text-[10px] px-1.5 py-0.5 rounded flex items-center gap-1 bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--status-positive-solid)] animate-pulse" />
                  Live
                </span>
              )}
              {detectedLanguage && !isStreaming && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]">
                  {detectedLanguage}
                </span>
              )}
            </div>
            <p className="text-[11px] text-[var(--text-subtle)] mt-1">
              Saved automatically to transcription history.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Select
              value={selectedLanguage}
              onValueChange={setSelectedLanguage}
              disabled={isProcessing}
            >
              <SelectTrigger className="h-[34px] w-[190px] sm:w-[220px] border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-xs text-[var(--text-secondary)]">
                <SelectValue placeholder="Language" />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGE_OPTIONS.map((language) => (
                  <SelectItem key={language} value={language}>
                    {language}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <label className="flex items-center gap-1.5 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1.5 text-xs text-[var(--text-secondary)]">
              <Radio className="w-3 h-3 text-[var(--text-muted)]" />
              Stream
              <input
                type="checkbox"
                checked={streamingEnabled}
                onChange={(event) =>
                  setStreamingEnabled(event.target.checked)
                }
                className="app-checkbox w-3.5 h-3.5 disabled:opacity-50"
                disabled={isProcessing}
              />
            </label>
            <button
              onClick={handleCopy}
              className="p-1.5 rounded transition-colors text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface-3)] disabled:opacity-40"
              disabled={!transcription || isStreaming}
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-[var(--text-primary)]" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
            <button
              onClick={handleDownload}
              className="p-1.5 rounded transition-colors text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface-3)] disabled:opacity-40"
              disabled={!transcription || isStreaming}
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div className="flex-1 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 overflow-y-auto">
          {showResult ? (
            <>
              {isProcessing && !transcription ? (
                <div className="h-full flex items-center justify-center text-sm text-[var(--text-muted)] gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {isStreaming ? "Streaming transcription..." : "Transcribing..."}
                </div>
              ) : (
                <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap min-h-[2em]">
                  {transcription || (isStreaming ? "Listening for speech..." : "")}
                </p>
              )}
            </>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-6">
              <div>
                <p className="text-sm text-[var(--text-muted)]">
                  Record audio or upload a file to start.
                </p>
                <p className="text-xs text-[var(--text-subtle)] mt-1">
                  The transcript appears live and is stored automatically.
                </p>
              </div>
            </div>
          )}
        </div>

        {processingStats && !isStreaming && (
          <GenerationStats stats={processingStats} type="asr" className="mt-3" />
        )}

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="p-2 rounded border text-xs mt-3 bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <aside className="card p-4 sm:p-5 min-h-[440px] lg:min-h-[560px] flex flex-col">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-[var(--text-muted)]">
              <History className="w-3.5 h-3.5" />
              History
            </div>
            <h3 className="text-sm font-medium text-[var(--text-primary)] mt-1">
              Transcription History
            </h3>
            <p className="text-xs text-[var(--text-subtle)] mt-1">
              {historyRecords.length}{" "}
              {historyRecords.length === 1 ? "record" : "records"}
            </p>
          </div>
          <button
            onClick={() => void loadHistory()}
            className="btn btn-ghost px-2.5 py-1.5 text-xs"
            disabled={historyLoading}
            title="Refresh history"
          >
            <RefreshCw
              className={clsx("w-3.5 h-3.5", historyLoading && "animate-spin")}
            />
            Refresh
          </button>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-2 max-h-[260px] overflow-y-auto">
          {historyLoading ? (
            <div className="h-full min-h-[120px] flex items-center justify-center gap-2 text-xs text-[var(--text-muted)]">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Loading history...
            </div>
          ) : historyRecords.length === 0 ? (
            <div className="h-full min-h-[120px] flex items-center justify-center text-center px-3 text-xs text-[var(--text-subtle)]">
              No saved transcriptions yet.
            </div>
          ) : (
            <div className="space-y-2">
              {historyRecords.map((record) => {
                const isActive = record.id === selectedHistoryRecordId;
                return (
                  <button
                    key={record.id}
                    onClick={() => setSelectedHistoryRecordId(record.id)}
                    className={clsx(
                      "w-full text-left rounded-lg border px-3 py-2 transition-colors",
                      isActive
                        ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)]"
                        : "border-[var(--border-muted)] bg-[var(--bg-surface-2)] hover:border-[var(--border-strong)]",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] text-[var(--text-secondary)] truncate">
                        {record.audio_filename || record.model_id || "Audio input"}
                      </span>
                      <span className="text-[10px] text-[var(--text-subtle)] shrink-0">
                        {formatCreatedAt(record.created_at)}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--text-primary)] mt-1 max-h-10 overflow-hidden">
                      {record.transcription_preview}
                    </p>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <AnimatePresence>
          {historyError && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="p-2 rounded border text-xs mt-3 bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]"
            >
              {historyError}
            </motion.div>
          )}
        </AnimatePresence>

        <div className="mt-3 flex-1 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3 flex flex-col min-h-[220px]">
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)] mb-2">
            <Clock3 className="w-3.5 h-3.5" />
            {selectedHistorySummary
              ? formatCreatedAt(selectedHistorySummary.created_at)
              : "Select a record"}
          </div>

          {selectedHistoryLoading ? (
            <div className="flex-1 flex items-center justify-center gap-2 text-xs text-[var(--text-muted)]">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Loading record...
            </div>
          ) : selectedHistoryError ? (
            <div className="rounded border px-2.5 py-2 text-xs bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]">
              {selectedHistoryError}
            </div>
          ) : activeHistoryRecord ? (
            <>
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-[var(--text-subtle)] mb-3">
                <span>{activeHistoryRecord.model_id || "Unknown model"}</span>
                <span>{activeHistoryRecord.language || "Unknown language"}</span>
                <span>{formatAudioDuration(activeHistoryRecord.duration_secs)}</span>
              </div>
              {selectedHistoryAudioUrl && (
                <audio
                  src={selectedHistoryAudioUrl}
                  controls
                  preload="metadata"
                  className="w-full h-9 mb-3"
                />
              )}
              <div className="flex-1 overflow-y-auto rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2">
                <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap">
                  {activeHistoryRecord.transcription || "No transcript text available."}
                </p>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-center px-3 text-xs text-[var(--text-subtle)]">
              Select a transcription from history to play audio and review text.
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
