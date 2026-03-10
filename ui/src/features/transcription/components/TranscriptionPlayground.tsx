import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Copy,
  Download,
  FileAudio,
  FileText,
  Loader2,
  Mic,
  Radio,
  RotateCcw,
  Settings2,
  Square,
  Upload,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { MiniWaveform } from "@/components/ui/Waveform";
import { ASRStats, GenerationStats } from "@/components/GenerationStats";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api, type TranscriptionRecord } from "@/api";
import { cn } from "@/lib/utils";
import {
  LANGUAGE_OPTIONS,
  LIVE_MIC_PCM_FRAME_SIZE,
  type ModelOption,
  type ProcessAudioOptions,
  type TranscriptionPlaygroundProps,
  buildTranscriptionRealtimeWebSocketUrl,
  encodeLiveMicChunk,
  encodeTranscriptionRealtimePcm16Frame,
  isTranscriptionRealtimeServerEvent,
  transcodeToWav,
} from "@/features/transcription/playground/support";
import { TranscriptionCorrectionsPanel } from "@/features/transcription/components/TranscriptionCorrectionsPanel";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import { TranscriptionHistoryPanel } from "@/features/transcription/components/TranscriptionHistoryPanel";
import { TranscriptionQualityPanel } from "@/features/transcription/components/TranscriptionQualityPanel";
import { TranscriptionReviewWorkspace } from "@/features/transcription/components/TranscriptionReviewWorkspace";
import { formattedTranscriptFromRecord } from "@/features/transcription/utils/transcriptionTranscript";

function revokeObjectUrlIfNeeded(url: string | null): void {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

export function TranscriptionPlayground({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  historyActionContainer,
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
  const [latestRecord, setLatestRecord] = useState<TranscriptionRecord | null>(
    null,
  );
  const [workspaceTab, setWorkspaceTab] = useState("transcript");
  const [updatePending, setUpdatePending] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const liveMicWsRef = useRef<WebSocket | null>(null);
  const liveMicWsReadyRef = useRef(false);
  const liveMicSessionRef = useRef(0);
  const liveMicInputFrameSeqRef = useRef(0);
  const liveMicAudioContextRef = useRef<AudioContext | null>(null);
  const liveMicAudioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const liveMicProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const liveMicProcessorSinkRef = useRef<GainNode | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return (
      modelOptions.find((option) => option.value === selectedModel) ?? null
    );
  }, [modelOptions, selectedModel]);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return false;
    }
    return true;
  }, [onModelRequired, selectedModel, selectedModelReady]);

  const draftRecord = useMemo<TranscriptionRecord | null>(() => {
    const draftText = transcription.trim();
    if (!draftText) {
      return null;
    }

    return {
      id: "draft",
      created_at: 0,
      model_id: selectedModel,
      aligner_model_id: null,
      language: detectedLanguage ?? selectedLanguage,
      duration_secs: processingStats?.audio_duration_secs ?? null,
      processing_time_ms: processingStats?.processing_time_ms ?? 0,
      rtf: processingStats?.rtf ?? null,
      audio_mime_type: "audio/wav",
      audio_filename: audioUrl ? "Current session" : null,
      raw_transcription: draftText,
      transcription: draftText,
      words: [],
      segments: [],
    };
  }, [
    audioUrl,
    detectedLanguage,
    processingStats,
    selectedLanguage,
    selectedModel,
    transcription,
  ]);

  const workspaceRecord = latestRecord ?? draftRecord;
  const workspaceTranscript = useMemo(
    () => formattedTranscriptFromRecord(workspaceRecord),
    [workspaceRecord],
  );
  const canRunInput = !isProcessing && !isRecording && selectedModelReady;
  const hasOutput = Boolean(workspaceRecord || isStreaming || isProcessing);
  const hasDraft = Boolean(
    transcription.trim() || audioUrl || latestRecord || error,
  );

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

  const stopLiveMicAudioPipeline = useCallback(() => {
    const processor = liveMicProcessorRef.current;
    liveMicProcessorRef.current = null;
    if (processor) {
      processor.onaudioprocess = null;
      try {
        processor.disconnect();
      } catch {
        // Best effort cleanup.
      }
    }

    const source = liveMicAudioSourceRef.current;
    liveMicAudioSourceRef.current = null;
    if (source) {
      try {
        source.disconnect();
      } catch {
        // Best effort cleanup.
      }
    }

    const sink = liveMicProcessorSinkRef.current;
    liveMicProcessorSinkRef.current = null;
    if (sink) {
      try {
        sink.disconnect();
      } catch {
        // Best effort cleanup.
      }
    }

    const context = liveMicAudioContextRef.current;
    liveMicAudioContextRef.current = null;
    if (context) {
      void context.close().catch(() => {});
    }

    liveMicInputFrameSeqRef.current = 0;
  }, []);

  const abortLiveMicStream = useCallback(() => {
    stopLiveMicAudioPipeline();
    liveMicWsReadyRef.current = false;
    liveMicInputFrameSeqRef.current = 0;

    const ws = liveMicWsRef.current;
    liveMicWsRef.current = null;
    if (
      ws &&
      (ws.readyState === WebSocket.OPEN ||
        ws.readyState === WebSocket.CONNECTING)
    ) {
      try {
        ws.close(1000, "transcription_reset");
      } catch {
        // Best effort cleanup.
      }
    }
  }, [stopLiveMicAudioPipeline]);

  const processAudio = useCallback(
    async (audioBlob: Blob, options: ProcessAudioOptions = {}) => {
      if (!requireReadyModel()) {
        return;
      }

      setIsProcessing(true);
      setError(null);
      setUpdateError(null);
      setProcessingStats(null);
      setLatestRecord(null);
      setWorkspaceTab("transcript");
      if (!options.preserveTranscript) {
        setTranscription("");
      }

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
          revokeObjectUrlIfNeeded(previousUrl);
          return url;
        });

        if (streamingEnabled) {
          setIsStreaming(true);
          let finalRecord: TranscriptionRecord | null = null;

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
                finalRecord = record;
                setLatestRecord(record);
                setWorkspaceTab("transcript");
                setTranscription(record.transcription);
                setDetectedLanguage(record.language || null);
                setProcessingStats({
                  processing_time_ms: record.processing_time_ms,
                  audio_duration_secs: record.duration_secs,
                  rtf: record.rtf,
                });
              },
              onError: (errorMessage) => {
                setError(errorMessage);
              },
              onDone: () => {
                setIsStreaming(false);
                setIsProcessing(false);
                streamAbortRef.current = null;
                if (!finalRecord) {
                  setLatestRecord(null);
                }
              },
            },
          );
          return;
        }

        const record = await api.createTranscriptionRecord({
          audio_file: uploadBlob,
          audio_filename: uploadFilename,
          model_id: selectedModel || undefined,
          language: selectedLanguage,
        });

        setLatestRecord(record);
        setWorkspaceTab("transcript");
        setTranscription(record.transcription);
        setDetectedLanguage(record.language || null);
        setProcessingStats({
          processing_time_ms: record.processing_time_ms,
          audio_duration_secs: record.duration_secs,
          rtf: record.rtf,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Transcription failed");
      } finally {
        setIsProcessing(false);
        setIsStreaming(false);
      }
    },
    [
      requireReadyModel,
      selectedLanguage,
      selectedModel,
      streamingEnabled,
    ],
  );

  const startRecording = useCallback(async () => {
    if (!requireReadyModel()) {
      return;
    }

    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const liveStream = stream;
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
      const recordingSession = liveMicSessionRef.current + 1;
      liveMicSessionRef.current = recordingSession;
      abortLiveMicStream();
      liveMicSessionRef.current = recordingSession;
      liveMicInputFrameSeqRef.current = 0;

      const ws = new WebSocket(
        buildTranscriptionRealtimeWebSocketUrl(api.baseUrl),
      );
      ws.binaryType = "arraybuffer";
      liveMicWsRef.current = ws;
      liveMicWsReadyRef.current = false;

      ws.onopen = () => {
        if (liveMicSessionRef.current !== recordingSession) {
          try {
            ws.close(1000, "stale_session");
          } catch {
            // Best effort cleanup.
          }
          return;
        }

        ws.send(
          JSON.stringify({
            type: "session_start",
            model_id: selectedModel || undefined,
            language: selectedLanguage,
          }),
        );
      };

      ws.onmessage = (messageEvent) => {
        if (liveMicSessionRef.current !== recordingSession) {
          return;
        }
        if (typeof messageEvent.data !== "string") {
          return;
        }

        let parsed: unknown;
        try {
          parsed = JSON.parse(messageEvent.data);
        } catch {
          return;
        }
        if (!isTranscriptionRealtimeServerEvent(parsed)) {
          return;
        }

        switch (parsed.type) {
          case "session_ready":
            liveMicWsReadyRef.current = true;
            break;
          case "session_started":
            break;
          case "transcript_partial":
            setLatestRecord(null);
            setWorkspaceTab("transcript");
            setTranscription(parsed.text || "");
            setDetectedLanguage(parsed.language || null);
            break;
          case "error":
            setError(parsed.message || "Realtime transcription error");
            break;
          case "session_done":
          case "pong":
            break;
        }
      };

      ws.onclose = () => {
        if (liveMicWsRef.current === ws) {
          liveMicWsRef.current = null;
        }
        liveMicWsReadyRef.current = false;
      };

      ws.onerror = () => {
        if (liveMicSessionRef.current !== recordingSession) {
          return;
        }
        setError("Live transcription connection error");
      };

      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
        streamAbortRef.current = null;
      }

      setLatestRecord(null);
      setWorkspaceTab("transcript");
      setTranscription("");
      setDetectedLanguage(null);
      setProcessingStats(null);
      setError(null);
      setIsStreaming(true);

      const audioContext = new AudioContext();
      await audioContext.resume();
      liveMicAudioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(liveStream);
      liveMicAudioSourceRef.current = source;

      const processor = audioContext.createScriptProcessor(
        LIVE_MIC_PCM_FRAME_SIZE,
        1,
        1,
      );
      liveMicProcessorRef.current = processor;

      const sink = audioContext.createGain();
      sink.gain.value = 0;
      liveMicProcessorSinkRef.current = sink;

      processor.onaudioprocess = (event) => {
        if (liveMicSessionRef.current !== recordingSession) {
          return;
        }
        const socket = liveMicWsRef.current;
        if (
          !socket ||
          socket.readyState !== WebSocket.OPEN ||
          !liveMicWsReadyRef.current
        ) {
          return;
        }

        const inputBuffer = event.inputBuffer;
        const channelCount = inputBuffer.numberOfChannels;
        const frameCount = inputBuffer.length;
        if (frameCount <= 0 || channelCount <= 0) {
          return;
        }

        const mono = new Float32Array(frameCount);
        for (
          let channelIndex = 0;
          channelIndex < channelCount;
          channelIndex += 1
        ) {
          const channel = inputBuffer.getChannelData(channelIndex);
          for (
            let sampleIndex = 0;
            sampleIndex < frameCount;
            sampleIndex += 1
          ) {
            mono[sampleIndex] += (channel[sampleIndex] ?? 0) / channelCount;
          }
        }

        const pcm16 = encodeLiveMicChunk(mono);
        const frameSeq = (liveMicInputFrameSeqRef.current + 1) >>> 0;
        liveMicInputFrameSeqRef.current = frameSeq;

        try {
          socket.send(
            encodeTranscriptionRealtimePcm16Frame(
              pcm16,
              Math.round(inputBuffer.sampleRate),
              frameSeq,
            ),
          );
        } catch {
          // Best effort send while websocket is open.
        }
      };

      source.connect(processor);
      processor.connect(sink);
      sink.connect(audioContext.destination);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        liveMicSessionRef.current = 0;
        abortLiveMicStream();
        setIsStreaming(false);
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder?.mimeType || "audio/webm",
        });
        liveStream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob, { preserveTranscript: true });
      };

      mediaRecorder.start(1000);
      setIsRecording(true);
    } catch {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      abortLiveMicStream();
      setError("Could not access microphone. Please grant permission.");
    }
  }, [
    abortLiveMicStream,
    processAudio,
    requireReadyModel,
    selectedLanguage,
    selectedModel,
  ]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      liveMicSessionRef.current = 0;
      const ws = liveMicWsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: "session_stop" }));
        } catch {
          // Best effort cleanup.
        }
      }
      abortLiveMicStream();
      setIsStreaming(false);
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [abortLiveMicStream, isRecording]);

  const handleFileUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const input = event.currentTarget;
      const file = input.files?.[0];
      if (!file) {
        return;
      }
      input.value = "";
      await processAudio(file, {
        filename: file.name,
        transcode: false,
      });
    },
    [processAudio],
  );

  const openFilePicker = useCallback(() => {
    if (!requireReadyModel()) {
      return;
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  }, [requireReadyModel]);

  const handleReset = useCallback(() => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
    liveMicSessionRef.current = 0;
    abortLiveMicStream();
    revokeObjectUrlIfNeeded(audioUrl);
    setAudioUrl(null);
    setLatestRecord(null);
    setWorkspaceTab("transcript");
    setTranscription("");
    setDetectedLanguage(null);
    setProcessingStats(null);
    setUpdateError(null);
    setError(null);
    setIsStreaming(false);
    setIsProcessing(false);
  }, [abortLiveMicStream, audioUrl]);

  const handleCopy = useCallback(async () => {
    if (!workspaceTranscript) {
      return;
    }
    await navigator.clipboard.writeText(workspaceTranscript);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  }, [workspaceTranscript]);

  const handleSaveCorrections = useCallback(
    async (segments: TranscriptionRecord["segments"]) => {
      if (!latestRecord || updatePending) {
        return;
      }

      setUpdatePending(true);
      setUpdateError(null);
      try {
        const transcriptionText = segments
          .map((segment) => segment.text.trim())
          .filter(Boolean)
          .join("\n\n");
        const updatedRecord = await api.updateTranscriptionRecord(
          latestRecord.id,
          {
            transcription: transcriptionText,
            segments,
          },
        );
        setLatestRecord(updatedRecord);
        setTranscription(updatedRecord.transcription);
        setDetectedLanguage(updatedRecord.language || null);
        setWorkspaceTab("transcript");
      } catch (err) {
        setUpdateError(
          err instanceof Error
            ? err.message
            : "Failed to save transcription corrections.",
        );
      } finally {
        setUpdatePending(false);
      }
    },
    [latestRecord, updatePending],
  );

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
      abortLiveMicStream();
      revokeObjectUrlIfNeeded(audioUrl);
    };
  }, [abortLiveMicStream, audioUrl]);

  const getStatusTone = (option: ModelOption): string => {
    if (option.isReady) {
      return "text-green-500 bg-green-500/10";
    }
    if (
      option.statusLabel.toLowerCase().includes("downloading") ||
      option.statusLabel.toLowerCase().includes("loading")
    ) {
      return "text-[var(--text-muted)] bg-amber-500/10";
    }
    if (option.statusLabel.toLowerCase().includes("error")) {
      return "text-destructive bg-destructive/10";
    }
    return "text-muted-foreground bg-muted";
  };

  const handleOpenModels = useCallback(() => {
    setIsModelMenuOpen(false);
    onOpenModelManager?.();
  }, [onOpenModelManager]);

  const renderModelSelector = () => (
    <div className="relative w-full" ref={modelMenuRef}>
      <Button
        variant="outline"
        onClick={() => setIsModelMenuOpen((previous) => !previous)}
        className={cn(
          "h-9 w-full justify-between font-normal",
          selectedOption?.isReady ? "border-primary/20 bg-primary/5" : "",
        )}
      >
        <span className="min-w-0 flex-1 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 shrink-0 opacity-50 transition-transform",
            isModelMenuOpen && "rotate-180",
          )}
        />
      </Button>

      <AnimatePresence>
        {isModelMenuOpen ? (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className="absolute left-0 right-0 top-full z-[90] mt-2 rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
          >
            <div className="max-h-64 overflow-y-auto">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel?.(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={cn(
                    "relative flex w-full cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
                    selectedOption?.value === option.value &&
                      "bg-accent text-accent-foreground",
                  )}
                >
                  <div className="flex w-full min-w-0 flex-col items-start">
                    <span className="w-full truncate text-left font-medium">
                      {option.label}
                    </span>
                    <span
                      className={cn(
                        "mt-1 rounded-sm px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                        getStatusTone(option),
                      )}
                    >
                      {option.statusLabel}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );

  return (
    <div className="grid gap-4 lg:gap-6 xl:h-[calc(100dvh-11.75rem)] xl:grid-cols-[340px,minmax(0,1fr)]">
      <div className="space-y-4 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 shadow-sm sm:p-5 xl:h-full xl:min-h-0 xl:overflow-y-auto">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              <FileAudio className="h-3.5 w-3.5" />
              Capture
            </div>
            <h2 className="mt-1.5 text-base font-semibold text-[var(--text-primary)]">
              Audio Input
            </h2>
          </div>
          {onOpenModelManager ? (
            <Button
              onClick={handleOpenModels}
              variant="outline"
              size="sm"
              className="h-8 gap-1.5 bg-[var(--bg-surface-1)] text-xs text-[var(--text-secondary)] shadow-sm hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
            >
              <Settings2 className="h-4 w-4" />
              Models
            </Button>
          ) : null}
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 shadow-inner">
          <div className="text-[11px] uppercase tracking-wide text-[var(--text-subtle)]">
            Active Model
          </div>
          <div className="mt-2">{modelOptions.length > 0 ? renderModelSelector() : null}</div>
          <div className="mt-3 border-t border-[var(--border-muted)] pt-3 text-xs">
            <span
              className={cn(
                selectedModelReady
                  ? "text-[var(--text-secondary)]"
                  : "text-[var(--text-muted)]",
              )}
            >
              {selectedModelReady
                ? "Loaded and ready"
                : "Select and load a transcription model"}
            </span>
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
                Transcription Settings
              </div>
              <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">
                Set the language hint and choose whether uploads stream partial text
                into the workspace while processing.
              </p>
            </div>
          </div>
          <div className="mt-3 grid gap-3">
            <label className="space-y-2">
              <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-subtle)]">
                Source language
              </span>
              <Select
                value={selectedLanguage}
                onValueChange={setSelectedLanguage}
                disabled={isProcessing}
              >
                <SelectTrigger className="h-9 border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm">
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
            </label>

            <label className="flex items-center justify-between gap-3 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2.5">
              <div className="flex items-center gap-2">
                <Radio className="h-3.5 w-3.5 text-[var(--text-subtle)]" />
                <div>
                  <div className="text-sm font-medium text-[var(--text-primary)]">
                    Stream partial transcript
                  </div>
                  <div className="text-[11px] text-[var(--text-muted)]">
                    Show draft text while uploads are processing.
                  </div>
                </div>
              </div>
              <input
                type="checkbox"
                checked={streamingEnabled}
                onChange={(event) => setStreamingEnabled(event.target.checked)}
                className="app-checkbox h-4 w-4"
                disabled={isProcessing}
                aria-label="Stream partial transcript"
              />
            </label>
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
          <div className="flex flex-col items-center">
            <button
              onClick={() => {
                if (isRecording) {
                  stopRecording();
                } else {
                  void startRecording();
                }
              }}
              className={cn(
                "flex h-24 w-24 items-center justify-center rounded-full border-2 shadow-md transition-all duration-300",
                isRecording
                  ? "scale-110 border-red-500 bg-red-500 text-white shadow-xl shadow-red-500/20"
                  : "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)] hover:border-[var(--text-muted)] hover:bg-[var(--border-muted)]",
                (!selectedModelReady || isProcessing) &&
                  "cursor-not-allowed opacity-50",
              )}
              disabled={!selectedModelReady || isProcessing}
            >
              {isRecording ? (
                <Square className="h-10 w-10 fill-current" />
              ) : (
                <Mic className="h-10 w-10" />
              )}
            </button>
            <p className="mt-4 text-sm font-medium text-[var(--text-secondary)]">
              {isRecording
                ? "Recording... click to stop"
                : "Tap to record audio"}
            </p>

            <div className="mt-6 w-full">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-[var(--border-muted)]" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-[var(--bg-surface-0)] px-2 text-[var(--text-muted)]">
                    Or
                  </span>
                </div>
              </div>

              <div
                onClick={openFilePicker}
                className={cn(
                  "mt-4 cursor-pointer rounded-xl border-2 border-dashed p-6 transition-colors",
                  canRunInput
                    ? "border-[var(--border-strong)] bg-[var(--bg-surface-1)] hover:border-primary/50 hover:bg-[var(--bg-surface-2)]"
                    : "cursor-not-allowed border-[var(--border-muted)] bg-[var(--bg-surface-1)] opacity-50",
                )}
              >
                <div className="flex flex-col items-center justify-center">
                  <Upload className="mb-2 h-6 w-6 text-[var(--text-muted)]" />
                  <p className="text-sm font-medium text-[var(--text-primary)]">
                    Upload audio file
                  </p>
                  <p className="mt-1 text-xs text-[var(--text-muted)]">
                    WAV, MP3, M4A, AAC
                  </p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={!canRunInput}
                />
              </div>
            </div>
          </div>
        </div>

        {audioUrl ? (
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
              Latest input
            </div>
            <audio src={audioUrl} controls className="mt-3 h-9 w-full" />
          </div>
        ) : null}

        {latestRecord || processingStats ? (
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
              Review Snapshot
            </div>
            <div className="mt-3 grid gap-2.5 sm:grid-cols-2 xl:grid-cols-1">
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2.5">
                <div className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                  Language
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                  {latestRecord?.language || detectedLanguage || selectedLanguage}
                </div>
              </div>
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2.5">
                <div className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                  Structure
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                  {latestRecord ? latestRecord.segments.length : 1} segments
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                  {latestRecord ? latestRecord.words.length : 0} timed words
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {processingStats && !isStreaming ? (
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <GenerationStats stats={processingStats} type="asr" />
          </div>
        ) : null}

        {hasDraft ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            disabled={isRecording}
            className="mt-2 h-9 w-full gap-2 border border-transparent bg-transparent text-xs hover:border-[var(--border-muted)] hover:bg-[var(--bg-surface-1)]"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset Session
          </Button>
        ) : null}
      </div>

      <div className="flex min-h-[460px] flex-col overflow-hidden rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] xl:h-full xl:min-h-0">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-4 sm:px-5">
          <div className="min-w-0">
            <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              <FileText className="h-3.5 w-3.5" />
              Review
            </div>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <h3 className="text-base font-semibold text-[var(--text-primary)]">
                Transcript Workspace
              </h3>
              {isStreaming ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-[10px] font-medium text-green-600">
                  <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                  Live
                </span>
              ) : null}
              {workspaceRecord?.language ? (
                <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-secondary)]">
                  {workspaceRecord.language}
                </span>
              ) : null}
            </div>
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              {latestRecord
                ? "Saved to transcription history and ready for review, corrections, and export."
                : "Record audio or upload a file to generate a review-ready transcript."}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              onClick={() => void handleCopy()}
              variant="outline"
              size="icon"
              className="h-9 w-9 border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
              disabled={!workspaceTranscript}
              title="Copy transcript"
            >
              {copied ? (
                <Check className="h-4 w-4 text-green-500" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
            <TranscriptionExportDialog record={latestRecord}>
              <Button
                variant="outline"
                size="icon"
                className="h-9 w-9 border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                disabled={!latestRecord || isProcessing}
                title="Export transcript"
              >
                <Download className="h-4 w-4" />
              </Button>
            </TranscriptionExportDialog>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto bg-[var(--bg-surface-0)] p-4 scrollbar-thin sm:p-6">
          {isProcessing && !transcription && !latestRecord ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-sm font-medium text-[var(--text-muted)]">
              <Loader2 className="h-5 w-5 animate-spin text-[var(--text-primary)]" />
              {isStreaming
                ? "Streaming transcription into the review workspace..."
                : "Transcribing audio..."}
            </div>
          ) : hasOutput ? (
            <Tabs
              value={workspaceTab}
              onValueChange={setWorkspaceTab}
              className="space-y-4"
            >
              <TabsList className="w-full justify-start bg-[var(--bg-surface-1)]">
                <TabsTrigger value="transcript">Transcript</TabsTrigger>
                <TabsTrigger value="corrections">Corrections</TabsTrigger>
                <TabsTrigger value="quality">Quality</TabsTrigger>
              </TabsList>

              <TabsContent value="transcript" className="mt-0 space-y-4">
                <TranscriptionReviewWorkspace
                  record={workspaceRecord}
                  audioUrl={audioUrl}
                  emptyMessage="Run transcription to review timed text."
                />
                {isStreaming ? (
                  <div className="flex items-center gap-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 text-sm text-[var(--text-secondary)]">
                    <MiniWaveform isActive={true} />
                    Listening for speech...
                  </div>
                ) : null}
              </TabsContent>

              <TabsContent value="corrections" className="mt-0">
                {latestRecord ? (
                  <TranscriptionCorrectionsPanel
                    record={latestRecord}
                    isSaving={updatePending}
                    error={updateError}
                    onSave={handleSaveCorrections}
                  />
                ) : (
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-5 text-sm leading-6 text-[var(--text-muted)] shadow-sm">
                    Corrections are available once the transcript has been saved to
                    history. Finish the current run to edit segments while keeping
                    subtitle timing intact.
                  </div>
                )}
              </TabsContent>

              <TabsContent value="quality" className="mt-0">
                <TranscriptionQualityPanel record={workspaceRecord} />
              </TabsContent>
            </Tabs>
          ) : (
            <div className="flex h-full items-center justify-center px-6 text-center">
              <div className="max-w-sm">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)]">
                  <FileText className="h-8 w-8 text-[var(--text-subtle)]" />
                </div>
                <p className="mb-2 text-base font-semibold text-[var(--text-secondary)]">
                  Ready to transcribe
                </p>
                <p className="text-sm leading-relaxed text-[var(--text-muted)]">
                  Record audio from your microphone or upload a file to create a
                  business-ready transcript with review, correction, and export
                  tools.
                </p>
              </div>
            </div>
          )}
        </div>

        <AnimatePresence>
          {error ? (
            <motion.div
              initial={{ opacity: 0, height: 0, y: 10 }}
              animate={{ opacity: 1, height: "auto", y: 0 }}
              exit={{ opacity: 0, height: 0, y: 10 }}
              className="m-4 flex items-start gap-3 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] p-3.5 text-sm font-medium text-[var(--danger-text)]"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      <TranscriptionHistoryPanel
        latestRecord={latestRecord}
        historyActionContainer={historyActionContainer}
      />
    </div>
  );
}
