import { useState, useRef, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  Square,
  Download,
  RotateCcw,
  Loader2,
  Globe,
  ChevronDown,
  Settings2,
} from "lucide-react";
import {
  api,
  type SpeechHistoryRecord,
  type TTSGenerationStats,
} from "../api";
import { VoiceClone } from "./VoiceClone";
import { LANGUAGES } from "../types";
import clsx from "clsx";
import { GenerationStats } from "./GenerationStats";
import { SpeechHistoryPanel } from "./SpeechHistoryPanel";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface VoiceClonePlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelLabel?: string | null;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
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

export function VoiceClonePlayground({
  selectedModel,
  selectedModelReady = false,
  modelLabel,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
}: VoiceClonePlaygroundProps) {
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [showLanguageSelect, setShowLanguageSelect] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [generationStats, setGenerationStats] = useState<TTSGenerationStats | null>(null);
  const [latestRecord, setLatestRecord] = useState<SpeechHistoryRecord | null>(null);
  const [voiceCloneAudio, setVoiceCloneAudio] = useState<string | null>(null);
  const [voiceCloneTranscript, setVoiceCloneTranscript] = useState<
    string | null
  >(null);
  const [isVoiceReady, setIsVoiceReady] = useState(false);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

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

  const handleGenerate = async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      setError("Please enter some text to synthesize");
      return;
    }

    if (!voiceCloneAudio || !voiceCloneTranscript) {
      setError("Please provide a voice reference (audio + transcript)");
      return;
    }

    try {
      setGenerating(true);
      setError(null);
      setGenerationStats(null);

      revokeObjectUrlIfNeeded(audioUrl);
      setAudioUrl(null);

      const record = await api.createVoiceCloningRecord({
        text: text.trim(),
        model_id: selectedModel,
        language: language === "Auto" ? undefined : language,
        max_tokens: 0,
        reference_audio: voiceCloneAudio,
        reference_text: voiceCloneTranscript,
      });

      setAudioUrl(api.voiceCloningRecordAudioUrl(record.id));
      setGenerationStats(mapRecordToStats(record));
      setLatestRecord(record);

      setTimeout(() => {
        audioRef.current?.play().catch(() => {});
      }, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const handleDownload = async () => {
    const record = latestRecord;
    if (record) {
      const downloadUrl = api.voiceCloningRecordAudioUrl(record.id, {
        download: true,
      });
      const filename = record.audio_filename || `izwi-voice-clone-${Date.now()}.wav`;
      const savedByDesktop = await api
        .saveAudioFile(downloadUrl, filename)
        .catch(() => false);
      if (savedByDesktop) {
        return;
      }

      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = filename;
      a.click();
      return;
    }

    if (!audioUrl) {
      return;
    }
    const a = document.createElement("a");
    a.href = audioUrl;
    a.download = `izwi-voice-clone-${Date.now()}.wav`;
    a.click();
  };

  const handleReset = () => {
    setText("");
    setError(null);
    setGenerationStats(null);
    revokeObjectUrlIfNeeded(audioUrl);
    setAudioUrl(null);
    textareaRef.current?.focus();
  };

  const handleVoiceCloneReady = (audio: string, transcript: string) => {
    setVoiceCloneAudio(audio);
    setVoiceCloneTranscript(transcript);
    setIsVoiceReady(true);
  };

  const handleVoiceCloneClear = () => {
    setVoiceCloneAudio(null);
    setVoiceCloneTranscript(null);
    setIsVoiceReady(false);
  };

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
    <div className="relative inline-block w-[280px] max-w-[85vw]" ref={modelMenuRef}>
      <button
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={clsx(
          "h-9 w-full px-3 rounded-lg border inline-flex items-center justify-between gap-2 text-xs transition-colors",
          selectedOption?.isReady
            ? "border-white/20 bg-white/10 text-gray-300"
            : "border-white/20 bg-[#1a1a1a] text-gray-300 hover:border-white/30",
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
            className="absolute left-0 right-0 top-full mt-2 rounded-xl border border-[#2a2a2a] bg-[#1a1a1a] p-1.5 shadow-2xl z-50"
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
                      ? "bg-white/10"
                      : "hover:bg-white/5",
                  )}
                >
                  <div className="text-xs text-gray-200 truncate">
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
          <div className="p-2 rounded bg-[#1a1a1a] border border-[#2a2a2a]">
            <Users className="w-5 h-5 text-gray-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">Voice Cloning</h2>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative">
            <button
              onClick={() => setShowLanguageSelect(!showLanguageSelect)}
              className="flex w-52 sm:w-56 items-center justify-between gap-2 px-3 py-1.5 rounded bg-[#1a1a1a] border border-[#2a2a2a] hover:bg-[#1f1f1f] text-sm"
            >
              <Globe className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-white flex-1 min-w-0 truncate text-left">
                {LANGUAGES.find((l) => l.id === language)?.name || language}
              </span>
              <ChevronDown
                className={clsx(
                  "w-3.5 h-3.5 text-gray-500 transition-transform",
                  showLanguageSelect && "rotate-180",
                )}
              />
            </button>

            <AnimatePresence>
              {showLanguageSelect && (
                <motion.div
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className="absolute left-0 right-0 top-full mt-1 max-h-64 overflow-y-auto p-1 rounded bg-[#1a1a1a] border border-[#2a2a2a] shadow-xl z-50"
                >
                  {LANGUAGES.map((lang) => (
                    <button
                      key={lang.id}
                      onClick={() => {
                        setLanguage(lang.id);
                        setShowLanguageSelect(false);
                      }}
                      className={clsx(
                        "w-full px-2 py-1.5 rounded text-left text-sm transition-colors",
                        language === lang.id
                          ? "bg-white/10 text-white"
                          : "hover:bg-[#2a2a2a] text-gray-400",
                      )}
                    >
                      {lang.name}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <div className="mb-4 rounded-xl border border-[#2b2b2b] bg-[#171717] p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="text-[11px] text-gray-500 uppercase tracking-wide">
              Active Model
            </div>
            <div className="mt-1 text-sm text-white truncate">
              {modelLabel ?? "No model selected"}
            </div>
            <div
              className={clsx(
                "mt-1 text-xs",
                selectedModelReady ? "text-gray-300" : "text-amber-400",
              )}
            >
              {selectedModelReady
                ? "Loaded and ready"
                : "Open Models and load a Base model"}
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

      <div className="space-y-4">
        {/* Voice Reference Section */}
        <div className="p-3 rounded-lg bg-[#161616] border border-[#2a2a2a]">
          <div className="flex items-center gap-2 mb-3">
            <Users className="w-4 h-4 text-gray-400" />
            <span className="text-xs font-medium text-white">
              Voice Reference
            </span>
            {isVoiceReady && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-gray-300 border border-white/20">
                Ready
              </span>
            )}
          </div>
          <VoiceClone
            onVoiceCloneReady={handleVoiceCloneReady}
            onClear={handleVoiceCloneClear}
          />
        </div>

        {/* Text to speak */}
        <div>
          <label className="block text-xs text-gray-500 font-medium mb-2">
            Text to Speak
          </label>
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the text you want to synthesize with the cloned voice..."
              rows={5}
              disabled={generating}
              className="textarea text-sm"
            />
            <div className="absolute bottom-2 right-2">
              <span className="text-xs text-gray-600">{text.length}</span>
            </div>
          </div>
        </div>

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

        {/* Actions */}
        <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
          <button
            onClick={handleGenerate}
            disabled={generating || !selectedModelReady || !isVoiceReady}
            className="btn btn-primary flex-1 min-h-[44px]"
          >
            {generating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Cloning Voice...
              </>
            ) : (
              <>
                <Users className="w-4 h-4" />
                Generate
              </>
            )}
          </button>

          {audioUrl && (
            <>
              <button
                onClick={handleStop}
                className="btn btn-secondary min-h-[44px] min-w-[44px]"
              >
                <Square className="w-4 h-4" />
              </button>
              <button
                onClick={handleDownload}
                className="btn btn-secondary min-h-[44px] min-w-[44px]"
              >
                <Download className="w-4 h-4" />
              </button>
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
          <p className="text-xs text-gray-400">
            Load a Base model to clone voices
          </p>
        )}

        {selectedModelReady && !isVoiceReady && (
          <p className="text-xs text-gray-400">
            Record or upload a voice sample to get started
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
            <div className="p-3 rounded bg-[#1a1a1a] border border-[#2a2a2a]">
              <audio ref={audioRef} src={audioUrl} className="w-full" controls />
            </div>
            {generationStats && (
              <GenerationStats stats={generationStats} type="tts" />
            )}
          </motion.div>
        )}
      </AnimatePresence>
      </div>

      <SpeechHistoryPanel
        route="voice-cloning"
        title="Voice Cloning History"
        emptyMessage="No saved voice-cloning generations yet."
        latestRecord={latestRecord}
      />
    </div>
  );
}
