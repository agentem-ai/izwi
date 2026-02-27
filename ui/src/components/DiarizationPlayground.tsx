import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Copy,
  Download,
  FileAudio,
  Loader2,
  Mic,
  RotateCcw,
  Settings2,
  Upload,
  Square,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { api, type DiarizationRecord } from "../api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { DiarizationHistoryPanel } from "./DiarizationHistoryPanel";

function revokeObjectUrlIfNeeded(url: string | null): void {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface DiarizationPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onTogglePipelineLoadAll?: () => void;
  pipelineAllLoaded?: boolean;
  pipelineLoadAllBusy?: boolean;
  onModelRequired: () => void;
  pipelineAsrModelId?: string | null;
  pipelineAlignerModelId?: string | null;
  pipelineLlmModelId?: string | null;
  pipelineModelsReady?: boolean;
  onPipelineModelsRequired?: () => void;
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

function isWavMimeType(mimeType: string | null | undefined): boolean {
  if (!mimeType) {
    return false;
  }
  const normalized = mimeType.toLowerCase();
  return (
    normalized === "audio/wav" ||
    normalized === "audio/x-wav" ||
    normalized === "audio/wave" ||
    normalized === "audio/vnd.wave"
  );
}

async function transcodeToWav(
  inputBlob: Blob,
  targetSampleRate = 16000,
  sourceFileName?: string,
): Promise<Blob> {
  const filenameLooksWav = sourceFileName
    ? sourceFileName.toLowerCase().endsWith(".wav")
    : false;
  if (isWavMimeType(inputBlob.type) || filenameLooksWav) {
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

interface TranscriptEntry {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

function normalizeDiarizedTranscript(
  transcript: string,
  rawTranscript: string,
): string {
  const source = (transcript || rawTranscript || "").trim();
  if (!source) {
    return "";
  }

  const withoutThink = source.replace(/<think>[\s\S]*?<\/think>/gi, " ");
  const withoutFences = withoutThink
    .replace(/```text/gi, "")
    .replace(/```/g, "")
    .trim();

  const lines = withoutFences
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^[-*]\s+/, "").replace(/^\d+\.\s+/, ""))
    .filter((line) =>
      /^[A-Za-z0-9_]+\s+\[\d+(?:\.\d+)?s\s*-\s*\d+(?:\.\d+)?s\]:/.test(line),
    );

  if (lines.length > 0) {
    return lines.join("\n");
  }

  return withoutFences;
}

function parseTranscriptEntries(transcript: string): TranscriptEntry[] {
  return transcript
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line): TranscriptEntry | null => {
      const match = line.match(
        /^([A-Za-z0-9_]+)\s+\[([0-9]+(?:\.[0-9]+)?)s\s*-\s*([0-9]+(?:\.[0-9]+)?)s\]:\s*(.*)$/,
      );
      if (!match) {
        return null;
      }
      const start = Number(match[2]);
      const end = Number(match[3]);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        return null;
      }
      return {
        speaker: match[1],
        start,
        end,
        text: match[4].trim(),
      };
    })
    .filter((entry): entry is TranscriptEntry => entry !== null);
}

export function DiarizationPlayground({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onTogglePipelineLoadAll,
  pipelineAllLoaded = false,
  pipelineLoadAllBusy = false,
  onModelRequired,
  pipelineAsrModelId = null,
  pipelineAlignerModelId = null,
  pipelineLlmModelId = null,
  pipelineModelsReady = true,
  onPipelineModelsRequired,
}: DiarizationPlaygroundProps) {
  const [speakerTranscript, setSpeakerTranscript] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [latestRecord, setLatestRecord] = useState<DiarizationRecord | null>(
    null,
  );
  const [minSpeakers, setMinSpeakers] = useState(1);
  const [maxSpeakers, setMaxSpeakers] = useState(4);
  const [minSpeechMs, setMinSpeechMs] = useState(240);
  const [minSilenceMs, setMinSilenceMs] = useState(200);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return false;
    }
    return true;
  }, [selectedModel, selectedModelReady, onModelRequired]);

  const requireReadyPipelineModels = useCallback(() => {
    if (
      !pipelineAsrModelId ||
      !pipelineAlignerModelId ||
      !pipelineLlmModelId ||
      !pipelineModelsReady
    ) {
      onPipelineModelsRequired?.();
      if (!onPipelineModelsRequired) {
        setError(
          "Load ASR, forced aligner, and transcript refiner models before diarization.",
        );
      }
      return false;
    }
    return true;
  }, [
    onPipelineModelsRequired,
    pipelineAlignerModelId,
    pipelineAsrModelId,
    pipelineLlmModelId,
    pipelineModelsReady,
  ]);

  const processAudio = useCallback(
    async (audioBlob: Blob) => {
      if (!requireReadyModel()) {
        return;
      }
      if (!requireReadyPipelineModels()) {
        return;
      }

      setIsProcessing(true);
      setError(null);
      setSpeakerTranscript("");

      try {
        const sourceFileName =
          audioBlob instanceof File && audioBlob.name
            ? audioBlob.name
            : "audio.wav";
        const wavBlob = await transcodeToWav(
          audioBlob,
          16000,
          sourceFileName,
        ).catch(() => audioBlob);
        revokeObjectUrlIfNeeded(audioUrl);
        setAudioUrl(null);

        const uploadFilename =
          wavBlob === audioBlob ? sourceFileName : "audio.wav";

        const record = await api.createDiarizationRecord({
          audio_file: wavBlob,
          audio_filename: uploadFilename,
          model_id: selectedModel || undefined,
          asr_model_id: pipelineAsrModelId || undefined,
          aligner_model_id: pipelineAlignerModelId || undefined,
          llm_model_id: pipelineLlmModelId || undefined,
          enable_llm_refinement: true,
          min_speakers: minSpeakers,
          max_speakers: maxSpeakers,
          min_speech_duration_ms: minSpeechMs,
          min_silence_duration_ms: minSilenceMs,
        });

        setLatestRecord(record);
        setAudioUrl(api.diarizationRecordAudioUrl(record.id));

        const cleanedTranscript = normalizeDiarizedTranscript(
          record.transcript || "",
          record.raw_transcript || "",
        );

        setSpeakerTranscript(cleanedTranscript);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Diarization failed");
      } finally {
        setIsProcessing(false);
      }
    },
    [
      audioUrl,
      maxSpeakers,
      minSilenceMs,
      minSpeakers,
      minSpeechMs,
      pipelineAlignerModelId,
      pipelineAsrModelId,
      pipelineLlmModelId,
      requireReadyModel,
      requireReadyPipelineModels,
      selectedModel,
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
    await processAudio(file);
    event.target.value = "";
  };

  const handleReset = () => {
    revokeObjectUrlIfNeeded(audioUrl);
    setSpeakerTranscript("");
    setAudioUrl(null);
    setLatestRecord(null);
    setError(null);
    setIsProcessing(false);
  };

  const transcriptEntries = useMemo(
    () => parseTranscriptEntries(speakerTranscript),
    [speakerTranscript],
  );

  const asText = useMemo(() => speakerTranscript.trim(), [speakerTranscript]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(asText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([asText], {
      type: "text/plain; charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `diarization-${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    if (minSpeakers > maxSpeakers) {
      setMinSpeakers(maxSpeakers);
    }
  }, [minSpeakers, maxSpeakers]);

  useEffect(() => {
    return () => {
      revokeObjectUrlIfNeeded(audioUrl);
    };
  }, [audioUrl]);

  const canRunInput = !isProcessing && !isRecording && selectedModelReady;
  const hasOutput = speakerTranscript.trim().length > 0;

  return (
    <div className="grid xl:grid-cols-[360px,minmax(0,1fr),320px] gap-4 lg:gap-6 xl:h-[calc(100dvh-11.75rem)]">
      <div className="card p-4 sm:p-5 space-y-4 xl:h-full xl:min-h-0 xl:overflow-y-auto">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-muted-foreground font-medium uppercase tracking-wider">
              <FileAudio className="w-3.5 h-3.5" />
              Capture
            </div>
            <h2 className="text-sm font-semibold mt-1">Audio Input</h2>
          </div>
          <div className="flex items-center gap-2">
            {onTogglePipelineLoadAll && (
              <Button
                onClick={onTogglePipelineLoadAll}
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 text-xs shadow-sm"
                disabled={pipelineLoadAllBusy}
              >
                {pipelineLoadAllBusy ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading...
                  </>
                ) : pipelineAllLoaded ? (
                  "Unload All"
                ) : (
                  "Load All"
                )}
              </Button>
            )}
            {onOpenModelManager && (
              <Button
                onClick={onOpenModelManager}
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 text-xs shadow-sm"
              >
                <Settings2 className="w-4 h-4" />
                Models
              </Button>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-muted/30 p-4 space-y-3 shadow-inner">
          <div className="text-[11px] text-muted-foreground uppercase tracking-wide">
            Active Model
          </div>
          <Select
            value={selectedModel ?? undefined}
            onValueChange={onSelectModel}
            disabled={modelOptions.length === 0}
          >
            <SelectTrigger className="h-[34px] text-xs">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {modelOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label} ({option.statusLabel})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div
            className={cn(
              "text-xs",
              selectedModelReady ? "text-muted-foreground" : "text-amber-500",
            )}
          >
            {selectedModelReady ? "Loaded and ready" : "Model not loaded yet"}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-card p-4 space-y-3 shadow-sm">
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs font-medium space-y-1.5">
              <span className="text-muted-foreground">Min Speakers</span>
              <input
                type="number"
                min={1}
                max={4}
                value={minSpeakers}
                onChange={(event) =>
                  setMinSpeakers(
                    Math.max(1, Math.min(4, Number(event.target.value) || 1)),
                  )
                }
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
            <label className="text-xs font-medium space-y-1.5">
              <span className="text-muted-foreground">Max Speakers</span>
              <input
                type="number"
                min={1}
                max={4}
                value={maxSpeakers}
                onChange={(event) =>
                  setMaxSpeakers(
                    Math.max(1, Math.min(4, Number(event.target.value) || 4)),
                  )
                }
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs font-medium space-y-1.5">
              <span className="text-muted-foreground">Min Speech (ms)</span>
              <input
                type="number"
                min={40}
                max={5000}
                value={minSpeechMs}
                onChange={(event) =>
                  setMinSpeechMs(
                    Math.max(
                      40,
                      Math.min(5000, Number(event.target.value) || 240),
                    ),
                  )
                }
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
            <label className="text-xs font-medium space-y-1.5">
              <span className="text-muted-foreground">Min Silence (ms)</span>
              <input
                type="number"
                min={40}
                max={5000}
                value={minSilenceMs}
                onChange={(event) =>
                  setMinSilenceMs(
                    Math.max(
                      40,
                      Math.min(5000, Number(event.target.value) || 200),
                    ),
                  )
                }
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
          </div>
        </div>

        <div className="py-2 space-y-4">
          <div className="flex items-center justify-center">
            <Button
              onClick={isRecording ? stopRecording : startRecording}
              variant={isRecording ? "destructive" : "outline"}
              className={cn(
                "h-24 w-24 rounded-full transition-all duration-300 flex items-center justify-center border-4",
                isRecording
                  ? "border-destructive/30 shadow-[0_0_0_8px_rgba(239,68,68,0.15)] animate-pulse"
                  : "border-primary/10 hover:border-primary/30",
              )}
              disabled={!selectedModelReady || isProcessing}
            >
              {isRecording ? (
                <Square className="w-8 h-8 fill-current" />
              ) : (
                <Mic className="w-8 h-8" />
              )}
            </Button>
          </div>
          <p className="text-center text-xs text-muted-foreground mt-3">
            {isRecording
              ? "Recording... click again to stop"
              : "Tap to record from microphone"}
          </p>

          <div className="relative border-t pt-4 mt-4">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-card px-2 text-xs text-muted-foreground font-medium uppercase tracking-widest">
              Or upload
            </div>
            <Button
              variant="secondary"
              className="w-full relative overflow-hidden h-12"
              onClick={() => {
                if (!requireReadyModel()) {
                  return;
                }
                fileInputRef.current?.click();
              }}
              disabled={!canRunInput}
            >
              <div className="absolute inset-0 flex items-center justify-center gap-2 pointer-events-none">
                <Upload className="w-4 h-4" />
                <span className="font-medium text-sm">Upload Audio File</span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                disabled={!canRunInput}
              />
            </Button>
          </div>
        </div>

        {audioUrl && (
          <div className="rounded-lg border border-[var(--border-muted)] bg-muted/30 p-3 shadow-inner">
            <div className="text-xs text-muted-foreground mb-2">
              Latest input
            </div>
            <audio src={audioUrl} controls className="w-full h-9" />
          </div>
        )}

        {(hasOutput || audioUrl || error) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="w-full gap-2 text-xs text-muted-foreground hover:text-foreground mt-2"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Session
          </Button>
        )}
      </div>

      <div className="rounded-xl border border-[var(--border-muted)] bg-card text-card-foreground shadow-sm flex flex-col h-[560px] xl:h-full overflow-hidden">
        <div className="px-4 sm:px-6 py-4 border-b border-[var(--border-muted)] flex items-center justify-between gap-3 bg-muted/20">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold tracking-tight">
              Diarized Transcript
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={handleCopy}
              variant="outline"
              size="sm"
              className="h-8 w-8 p-0 shadow-sm"
              disabled={!hasOutput || isProcessing}
              title="Copy transcript"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-500" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </Button>
            <Button
              onClick={handleDownload}
              variant="outline"
              size="sm"
              className="h-8 w-8 p-0 shadow-sm"
              disabled={!hasOutput || isProcessing}
              title="Download transcript"
            >
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 sm:p-5 bg-background/50">
          {isProcessing ? (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-primary" />
              Running diarization and transcript pipeline...
            </div>
          ) : hasOutput ? (
            <div className="space-y-4">
              {transcriptEntries.length > 0 ? (
                <div className="space-y-3">
                  {transcriptEntries.map((entry, index) => (
                    <div
                      key={`${entry.speaker}-${entry.start}-${entry.end}-${index}`}
                      className="rounded-lg border border-[var(--border-muted)] bg-card p-4 shadow-sm"
                    >
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className="text-xs font-semibold text-foreground">
                          {entry.speaker}
                        </span>
                        <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded border border-[var(--border-muted)]">
                          {entry.start.toFixed(2)}s - {entry.end.toFixed(2)}s
                        </span>
                      </div>
                      <p className="text-sm text-foreground/90 whitespace-pre-wrap break-words leading-relaxed">
                        {entry.text}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-lg border border-[var(--border-muted)] bg-card p-4 shadow-sm">
                  <pre className="text-sm text-foreground/90 whitespace-pre-wrap break-words leading-relaxed font-sans">
                    {speakerTranscript}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-6">
              <div className="max-w-xs">
                <p className="text-sm font-medium text-muted-foreground">
                  Record audio or upload a file to start diarization.
                </p>
                <p className="text-xs text-muted-foreground/70 mt-1">
                  Your diarized transcript will appear here.
                </p>
              </div>
            </div>
          )}
        </div>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0, y: 10 }}
              animate={{ opacity: 1, height: "auto", y: 0 }}
              exit={{ opacity: 0, height: 0, y: 10 }}
              className="m-3 p-3 rounded-lg border border-destructive/20 bg-destructive/10 text-destructive text-xs font-medium"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <DiarizationHistoryPanel
        latestRecord={latestRecord}
        desktopHeightClassName="xl:h-full"
      />
    </div>
  );
}
