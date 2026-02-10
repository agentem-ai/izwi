import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Mic,
  MicOff,
  Upload,
  Download,
  RotateCcw,
  Loader2,
  Copy,
  Check,
  Radio,
} from "lucide-react";
import { api } from "../api";
import { GenerationStats, ASRStats } from "./GenerationStats";
import clsx from "clsx";

interface TranscriptionPlaygroundProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

export function TranscriptionPlayground({
  selectedModel,
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
  const [streamingEnabled, setStreamingEnabled] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("English");

  const languageOptions = [
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

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  const startRecording = useCallback(async () => {
    if (!selectedModel) {
      onModelRequired();
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
    } catch (err) {
      setError("Could not access microphone. Please grant permission.");
    }
  }, [selectedModel, onModelRequired]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setError(null);
    setProcessingStats(null);
    setTranscription("");

    const url = URL.createObjectURL(audioBlob);
    setAudioUrl(url);

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
        // Use streaming transcription
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

              // Calculate processing stats
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
        // Use non-streaming transcription
        const response = await api.asrTranscribe({
          audio_base64: audioBase64,
          model_id: selectedModel || undefined,
          language: selectedLanguage,
        });

        setTranscription(response.transcription);
        setDetectedLanguage(response.language || null);

        // Set processing stats if available
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
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!selectedModel) {
      onModelRequired();
      return;
    }

    await processAudio(file);
  };

  const handleReset = () => {
    // Cancel any ongoing streaming
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
    const a = document.createElement("a");
    a.href = url;
    a.download = `transcription-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded bg-[#1a1a1a] border border-[#2a2a2a]">
            <FileText className="w-5 h-5 text-gray-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">Transcription</h2>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400" htmlFor="asr-language">
              Language
            </label>
            <select
              id="asr-language"
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              disabled={isProcessing}
            >
              {languageOptions.map((lang) => (
                <option key={lang} value={lang}>
                  {lang}
                </option>
              ))}
            </select>
          </div>
          {/* Streaming Toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={streamingEnabled}
              onChange={(e) => setStreamingEnabled(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-[#1a1a1a] text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
              disabled={isProcessing}
            />
            <span className="text-xs text-gray-400 flex items-center gap-1">
              <Radio className="w-3 h-3" />
              Stream
            </span>
          </label>

          {(transcription || audioUrl) && (
            <button onClick={handleReset} className="btn btn-ghost text-xs">
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Audio Preview */}
      {audioUrl && (
        <div className="mb-4 p-3 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a]">
          <div className="text-xs text-gray-500 mb-2">Audio Input</div>
          <audio src={audioUrl} controls className="w-full h-8" />
        </div>
      )}

      {/* Transcription Result */}
      {(transcription || isStreaming) && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Transcription</span>
              {isStreaming && (
                <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  Live
                </span>
              )}
              {detectedLanguage && !isStreaming && (
                <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded">
                  {detectedLanguage}
                </span>
              )}
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={handleCopy}
                className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300"
                disabled={!transcription || isStreaming}
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-green-500" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
              <button
                onClick={handleDownload}
                className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300"
                disabled={!transcription || isStreaming}
              >
                <Download className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
          <div
            className={clsx(
              "p-3 rounded-lg bg-[#1a1a1a] border",
              isStreaming ? "border-emerald-500/30" : "border-[#2a2a2a]",
            )}
          >
            <p className="text-sm text-gray-300 whitespace-pre-wrap min-h-[2em]">
              {transcription || (isStreaming ? "..." : "")}
            </p>
          </div>
          {processingStats && !isStreaming && (
            <GenerationStats
              stats={processingStats}
              type="asr"
              className="mt-3"
            />
          )}
        </div>
      )}

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-2 rounded bg-red-950/50 border border-red-900/50 text-red-400 text-xs mb-4"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Processing indicator */}
      {isProcessing && !transcription && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center gap-2 text-gray-400 text-sm py-8"
        >
          <Loader2 className="w-5 h-5 animate-spin" />
          {isStreaming ? "Streaming transcription..." : "Transcribing..."}
        </motion.div>
      )}

      {/* Input Controls */}
      {!isProcessing && (
        <div className="space-y-4">
          {/* Record / Upload buttons */}
          <div className="flex items-center justify-center gap-6">
            {/* Record button */}
            <div className="flex flex-col items-center gap-2">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={clsx(
                  "p-4 rounded-full transition-all min-h-[56px] min-w-[56px]",
                  isRecording
                    ? "bg-white hover:bg-gray-200 animate-pulse"
                    : "bg-[#1a1a1a] hover:bg-[#2a2a2a] border border-[#2a2a2a]",
                )}
              >
                {isRecording ? (
                  <MicOff className="w-6 h-6 text-black" />
                ) : (
                  <Mic className="w-6 h-6 text-gray-300" />
                )}
              </button>
              <span className="text-xs text-gray-500">
                {isRecording ? "Stop" : "Record"}
              </span>
            </div>

            <div className="text-gray-600 text-xs">or</div>

            {/* Upload button */}
            <div className="flex flex-col items-center gap-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-4 rounded-full bg-[#1a1a1a] hover:bg-[#2a2a2a] border border-[#2a2a2a] transition-all min-h-[56px] min-w-[56px]"
              >
                <Upload className="w-6 h-6 text-gray-300" />
              </button>
              <span className="text-xs text-gray-500">Upload</span>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
          </div>

          {isRecording && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center justify-center gap-2 text-gray-400 text-sm"
            >
              <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
              Recording...
            </motion.div>
          )}

          <p className="text-center text-xs text-gray-500">
            {isRecording
              ? "Click the microphone to stop recording"
              : "Record audio or upload a file for transcription"}
          </p>
        </div>
      )}
    </div>
  );
}
