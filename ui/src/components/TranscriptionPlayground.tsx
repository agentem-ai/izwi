import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Copy,
  Download,
  FileAudio,
  FileText,
  Loader2,
  Mic,
  MicOff,
  Radio,
  RotateCcw,
  Settings2,
  Upload,
} from "lucide-react";
import clsx from "clsx";
import { api } from "../api";
import { ASRStats, GenerationStats } from "./GenerationStats";

interface TranscriptionPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelLabel?: string | null;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
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

export function TranscriptionPlayground({
  selectedModel,
  selectedModelReady = false,
  modelLabel,
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

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

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
      setProcessingStats(null);
      setTranscription("");

      const url = URL.createObjectURL(audioBlob);
      setAudioUrl((previousUrl) => {
        if (previousUrl) {
          URL.revokeObjectURL(previousUrl);
        }
        return url;
      });

      try {
        const reader = new FileReader();
        const audioBase64 = await new Promise<string>((resolve, reject) => {
          reader.onloadend = () => {
            const base64 = (reader.result as string).split(",")[1];
            resolve(base64);
          };
          reader.onerror = reject;
          reader.readAsDataURL(audioBlob);
        });

        if (streamingEnabled) {
          setIsStreaming(true);
          const startTime = Date.now();
          let audioDuration: number | null = null;

          streamAbortRef.current = api.asrTranscribeStream(
            {
              audio_base64: audioBase64,
              model_id: selectedModel || undefined,
              language: selectedLanguage,
            },
            {
              onStart: (duration) => {
                audioDuration = duration;
              },
              onDelta: (delta) => {
                setTranscription((prev) => `${prev}${delta}`);
              },
              onFinal: (text, language, duration) => {
                setTranscription(text);
                setDetectedLanguage(language);
                audioDuration = duration;

                const processingTimeMs = Date.now() - startTime;
                const rtf =
                  audioDuration && audioDuration > 0
                    ? processingTimeMs / 1000 / audioDuration
                    : null;

                setProcessingStats({
                  processing_time_ms: processingTimeMs,
                  audio_duration_secs: audioDuration,
                  rtf,
                });
              },
              onError: (errorMsg) => {
                setError(errorMsg);
              },
              onDone: () => {
                setIsStreaming(false);
                setIsProcessing(false);
                streamAbortRef.current = null;
              },
            },
          );
        } else {
          const response = await api.asrTranscribe({
            audio_base64: audioBase64,
            model_id: selectedModel || undefined,
            language: selectedLanguage,
          });

          setTranscription(response.transcription);
          setDetectedLanguage(response.language || null);

          if (response.stats) {
            setProcessingStats({
              processing_time_ms: response.stats.processing_time_ms,
              audio_duration_secs: response.stats.audio_duration_secs,
              rtf: response.stats.rtf,
            });
          }
          setIsProcessing(false);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Transcription failed");
        setIsProcessing(false);
        setIsStreaming(false);
      }
    },
    [requireReadyModel, selectedModel, selectedLanguage, streamingEnabled],
  );

  const startRecording = useCallback(async () => {
    if (!requireReadyModel()) {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
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
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const canRunInput = !isProcessing && !isRecording && selectedModelReady;
  const showResult = Boolean(transcription || isStreaming || isProcessing);
  const hasDraft = Boolean(transcription || audioUrl || error);

  return (
    <div className="grid xl:grid-cols-[360px,1fr] gap-4 lg:gap-6">
      <div className="card p-4 sm:p-5 space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-gray-400">
              <FileAudio className="w-3.5 h-3.5" />
              Capture
            </div>
            <h2 className="text-sm font-medium text-white mt-1">
              Audio Input
            </h2>
          </div>
          <div className="flex items-center gap-2">
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
        </div>

        <div className="rounded-xl border border-[#2b2b2b] bg-[#171717] p-3">
          <div className="text-[11px] text-gray-500 uppercase tracking-wide">
            Active Model
          </div>
          <div className="mt-1 text-sm text-white truncate">
            {modelLabel ?? "No model selected"}
          </div>
          <div
            className={clsx(
              "mt-2 text-xs",
              selectedModelReady ? "text-emerald-300" : "text-amber-300",
            )}
          >
            {selectedModelReady
              ? "Loaded and ready"
              : "Select and load a transcription model"}
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

        {hasDraft && (
          <button onClick={handleReset} className="btn btn-ghost w-full text-xs">
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Session
          </button>
        )}
      </div>

      <div className="card p-4 sm:p-5 min-h-[560px] flex flex-col">
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-gray-400" />
            <h3 className="text-sm font-medium text-white">Transcript</h3>
            {isStreaming && (
              <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                Live
              </span>
            )}
            {detectedLanguage && !isStreaming && (
              <span className="text-[10px] px-1.5 py-0.5 bg-white/10 text-gray-300 rounded">
                {detectedLanguage}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <select
              value={selectedLanguage}
              onChange={(event) => setSelectedLanguage(event.target.value)}
              className="bg-[#101216] border border-[#333842] text-xs text-gray-300 rounded-md px-2 py-1.5 focus:outline-none focus:ring-0 focus:border-[#333842] disabled:text-gray-500 disabled:border-[#2a2e36] [&>option]:bg-[#101216] [&>option]:text-gray-300"
              style={{ colorScheme: "dark" }}
              disabled={isProcessing}
            >
              {LANGUAGE_OPTIONS.map((language) => (
                <option key={language} value={language}>
                  {language}
                </option>
              ))}
            </select>
            <label className="flex items-center gap-1.5 rounded-md border border-[#333842] bg-[#101216] px-2 py-1.5 text-xs text-gray-300">
              <Radio className="w-3 h-3 text-gray-400" />
              Stream
              <input
                type="checkbox"
                checked={streamingEnabled}
                onChange={(event) =>
                  setStreamingEnabled(event.target.checked)
                }
                className="w-3.5 h-3.5 rounded border-[#3a404b] bg-[#0f1115] text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0 disabled:opacity-50"
                style={{ colorScheme: "dark" }}
                disabled={isProcessing}
              />
            </label>
            <button
              onClick={handleCopy}
              className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300 disabled:opacity-40"
              disabled={!transcription || isStreaming}
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
            <button
              onClick={handleDownload}
              className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300 disabled:opacity-40"
              disabled={!transcription || isStreaming}
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div className="flex-1 rounded-xl border border-[#262626] bg-[#101114] p-4 overflow-y-auto">
          {showResult ? (
            <>
              {isProcessing && !transcription ? (
                <div className="h-full flex items-center justify-center text-sm text-gray-400 gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {isStreaming ? "Streaming transcription..." : "Transcribing..."}
                </div>
              ) : (
                <p className="text-sm text-gray-200 whitespace-pre-wrap min-h-[2em]">
                  {transcription || (isStreaming ? "Listening for speech..." : "")}
                </p>
              )}
            </>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-6">
              <div>
                <p className="text-sm text-gray-400">
                  Record audio or upload a file to start.
                </p>
                <p className="text-xs text-gray-600 mt-1">
                  Your transcript will appear here with timing stats.
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
