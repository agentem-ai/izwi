import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Copy,
  Download,
  FileAudio,
  Loader2,
  Mic,
  MicOff,
  RotateCcw,
  Settings2,
  Upload,
} from "lucide-react";
import clsx from "clsx";
import { api } from "../api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

const PIPELINE_ASR_MODEL_ID = "Qwen3-ASR-0.6B";
const PIPELINE_ALIGNER_MODEL_ID = "Qwen3-ForcedAligner-0.6B";
const PIPELINE_LLM_MODEL_ID = "Gemma-3-1b-it";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface DiarizationPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelLabel?: string | null;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
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
  modelLabel,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
}: DiarizationPlaygroundProps) {
  const [speakerTranscript, setSpeakerTranscript] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [minSpeakers, setMinSpeakers] = useState(1);
  const [maxSpeakers, setMaxSpeakers] = useState(4);
  const [minSpeechMs, setMinSpeechMs] = useState(240);
  const [minSilenceMs, setMinSilenceMs] = useState(200);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return modelOptions.find((option) => option.value === selectedModel) || null;
  }, [selectedModel, modelOptions]);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return false;
    }
    return true;
  }, [selectedModel, selectedModelReady, onModelRequired]);

  const processAudio = useCallback(
    async (audioBlob: Blob) => {
      if (!requireReadyModel()) {
        return;
      }

      setIsProcessing(true);
      setError(null);
      setSpeakerTranscript("");

      try {
        const wavBlob = await transcodeToWav(audioBlob, 16000);
        const url = URL.createObjectURL(wavBlob);
        setAudioUrl((previousUrl) => {
          if (previousUrl) {
            URL.revokeObjectURL(previousUrl);
          }
          return url;
        });

        const response = await api.diarize({
          audio_file: wavBlob,
          audio_filename: "audio.wav",
          model_id: selectedModel || undefined,
          asr_model_id: PIPELINE_ASR_MODEL_ID,
          aligner_model_id: PIPELINE_ALIGNER_MODEL_ID,
          llm_model_id: PIPELINE_LLM_MODEL_ID,
          enable_llm_refinement: true,
          min_speakers: minSpeakers,
          max_speakers: maxSpeakers,
          min_speech_duration_ms: minSpeechMs,
          min_silence_duration_ms: minSilenceMs,
        });

        const cleanedTranscript = normalizeDiarizedTranscript(
          response.transcript || "",
          response.raw_transcript || "",
        );

        setSpeakerTranscript(cleanedTranscript);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Diarization failed");
      } finally {
        setIsProcessing(false);
      }
    },
    [
      maxSpeakers,
      minSilenceMs,
      minSpeakers,
      minSpeechMs,
      requireReadyModel,
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
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setSpeakerTranscript("");
    setAudioUrl(null);
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

  const canRunInput = !isProcessing && !isRecording && selectedModelReady;
  const hasOutput = speakerTranscript.trim().length > 0;

  return (
    <div className="grid xl:grid-cols-[360px,1fr] gap-4 lg:gap-6">
      <div className="card p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-gray-400">
              <FileAudio className="w-3.5 h-3.5" />
              Capture
            </div>
            <h2 className="text-sm font-medium text-white mt-1">
              Audio Input
            </h2>
          </div>
          {onOpenModelManager && (
            <button
              onClick={onOpenModelManager}
              className="btn btn-secondary text-xs"
            >
              <Settings2 className="w-4 h-4" />
              Models
            </button>
          )}
        </div>

        <div className="rounded-xl border border-[#2b2b2b] bg-[#171717] p-4 space-y-3">
          <div className="text-[11px] text-gray-500 uppercase tracking-wide">
            Active Model
          </div>
          <Select
            value={selectedModel ?? undefined}
            onValueChange={onSelectModel}
            disabled={modelOptions.length === 0}
          >
            <SelectTrigger className="h-[34px] border-[#2a2a2a] bg-[#171717] text-xs text-gray-300">
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
            className={clsx(
              "text-xs",
              selectedModelReady ? "text-gray-300" : "text-amber-400",
            )}
          >
            {selectedOption?.label || modelLabel || "No model selected"}
            {selectedModelReady
              ? " is loaded and ready"
              : " is not loaded yet"}
          </div>
        </div>

        <div className="rounded-xl border border-[#2b2b2b] bg-[#171717] p-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-gray-400 space-y-1">
              <span>Min Speakers</span>
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
                className="input h-8 px-2 py-0 text-sm"
              />
            </label>
            <label className="text-xs text-gray-400 space-y-1">
              <span>Max Speakers</span>
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
                className="input h-8 px-2 py-0 text-sm"
              />
            </label>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-gray-400 space-y-1">
              <span>Min Speech (ms)</span>
              <input
                type="number"
                min={40}
                max={5000}
                value={minSpeechMs}
                onChange={(event) =>
                  setMinSpeechMs(
                    Math.max(40, Math.min(5000, Number(event.target.value) || 240)),
                  )
                }
                className="input h-8 px-2 py-0 text-sm"
              />
            </label>
            <label className="text-xs text-gray-400 space-y-1">
              <span>Min Silence (ms)</span>
              <input
                type="number"
                min={40}
                max={5000}
                value={minSilenceMs}
                onChange={(event) =>
                  setMinSilenceMs(
                    Math.max(40, Math.min(5000, Number(event.target.value) || 200)),
                  )
                }
                className="input h-8 px-2 py-0 text-sm"
              />
            </label>
          </div>
        </div>

        <div className="rounded-2xl border border-[#2b2b2b] bg-[#111214] p-5">
          <div className="flex items-center justify-center">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={clsx(
                "h-24 w-24 rounded-full border transition-all duration-150 flex items-center justify-center",
                isRecording
                  ? "bg-white border-white text-black shadow-[0_0_0_8px_rgba(255,255,255,0.08)]"
                  : "bg-[#181a1e] border-[#2f3239] text-gray-300 hover:text-white hover:border-[#4c5565]",
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
          <p className="text-center text-xs text-gray-500 mt-3">
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
          <div className="rounded-lg border border-[#2a2a2a] bg-[#171717] p-3">
            <div className="text-xs text-gray-500 mb-2">Latest input</div>
            <audio src={audioUrl} controls className="w-full h-9" />
          </div>
        )}

        {(hasOutput || audioUrl || error) && (
          <button onClick={handleReset} className="btn btn-ghost w-full text-xs">
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Session
          </button>
        )}
      </div>

      <div className="card p-4 sm:p-5 min-h-[560px] flex flex-col">
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium text-white">Diarized Transcript</h3>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300 disabled:opacity-40"
              disabled={!hasOutput || isProcessing}
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-gray-300" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
            <button
              onClick={handleDownload}
              className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300 disabled:opacity-40"
              disabled={!hasOutput || isProcessing}
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div className="flex-1 rounded-xl border border-[#262626] bg-[#101114] p-4 overflow-y-auto">
          {isProcessing ? (
            <div className="h-full flex items-center justify-center text-sm text-gray-400 gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Running diarization and transcript pipeline...
            </div>
          ) : hasOutput ? (
            <div className="space-y-3">
              {transcriptEntries.length > 0 ? (
                <div className="space-y-2">
                  {transcriptEntries.map((entry, index) => (
                    <div
                      key={`${entry.speaker}-${entry.start}-${entry.end}-${index}`}
                      className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3"
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="text-xs font-medium text-gray-200">
                          {entry.speaker}
                        </span>
                        <span className="text-[11px] text-gray-400">
                          {entry.start.toFixed(2)}s - {entry.end.toFixed(2)}s
                        </span>
                      </div>
                      <p className="text-sm text-gray-100 whitespace-pre-wrap break-words">
                        {entry.text}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-lg border border-[#2a2a2a] bg-[#151515] p-3">
                  <pre className="text-sm text-gray-200 whitespace-pre-wrap break-words">
                    {speakerTranscript}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-6">
              <div>
                <p className="text-sm text-gray-400">
                  Record audio or upload a file to start diarization.
                </p>
                <p className="text-xs text-gray-600 mt-1">
                  Your diarized transcript will appear here.
                </p>
              </div>
            </div>
          )}
        </div>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="p-2 rounded bg-red-950/50 border border-red-900/50 text-red-300 text-xs mt-3"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
