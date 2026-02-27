import { useState, useRef, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  Square,
  Download,
  RotateCcw,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Globe,
  ChevronDown,
  Settings2,
} from "lucide-react";
import { api, type SpeechHistoryRecord, type TTSGenerationStats } from "../api";
import { VoiceClone } from "./VoiceClone";
import { LANGUAGES } from "../types";
import clsx from "clsx";
import { GenerationStats } from "./GenerationStats";
import { SpeechHistoryPanel } from "./SpeechHistoryPanel";
import { useDownloadIndicator } from "../utils/useDownloadIndicator";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface VoiceClonePlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
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
  const [generationStats, setGenerationStats] =
    useState<TTSGenerationStats | null>(null);
  const [latestRecord, setLatestRecord] = useState<SpeechHistoryRecord | null>(
    null,
  );
  const [voiceCloneAudio, setVoiceCloneAudio] = useState<string | null>(null);
  const [voiceCloneTranscript, setVoiceCloneTranscript] = useState<
    string | null
  >(null);
  const [isVoiceReady, setIsVoiceReady] = useState(false);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
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
  const modelMenuRef = useRef<HTMLDivElement>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return (
      modelOptions.find((option) => option.value === selectedModel) || null
    );
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
    const localAudioUrl = !record ? audioUrl : null;
    if ((!record && !localAudioUrl) || isDownloading) {
      return;
    }

    beginDownload();
    try {
      if (record) {
        const downloadUrl = api.voiceCloningRecordAudioUrl(record.id, {
          download: true,
        });
        const filename =
          record.audio_filename || `izwi-voice-clone-${Date.now()}.wav`;
        await api.downloadAudioFile(downloadUrl, filename);
        completeDownload();
        return;
      }

      if (!localAudioUrl) {
        return;
      }
      await api.downloadAudioFile(
        localAudioUrl,
        `izwi-voice-clone-${Date.now()}.wav`,
      );
      completeDownload();
    } catch (error) {
      failDownload(error);
    }
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
      return "text-[var(--text-secondary)] bg-[var(--bg-surface-3)] border border-[var(--border-muted)]";
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
    return "text-[var(--text-muted)] bg-[var(--bg-surface-2)] border border-[var(--border-muted)]";
  };

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager?.();
  };

  const renderModelSelector = () => (
    <div
      className="relative inline-block w-[280px] max-w-[85vw]"
      ref={modelMenuRef}
    >
      <button
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={clsx(
          "h-9 w-full px-3 rounded-lg border inline-flex items-center justify-between gap-2 text-xs transition-colors",
          selectedOption?.isReady
            ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
            : "border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-strong)]",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown
          className={clsx(
            "w-3.5 h-3.5 shrink-0 transition-transform",
            isModelMenuOpen && "rotate-180",
          )}
        />
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
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr),320px] items-stretch xl:h-[calc(100dvh-11.75rem)]">
      <div className="card p-4 flex min-h-0 flex-col">
        <div className="flex-1 min-h-0 overflow-y-auto pr-1 scrollbar-thin">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)]">
                <Users className="w-5 h-5 text-[var(--text-muted)]" />
              </div>
              <div>
                <h2 className="text-sm font-medium text-[var(--text-primary)]">
                  Voice Cloning
                </h2>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className="relative">
                <button
                  onClick={() => setShowLanguageSelect(!showLanguageSelect)}
                  className="flex w-52 sm:w-56 items-center justify-between gap-2 px-3 py-1.5 rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)] hover:bg-[var(--bg-surface-3)] text-sm"
                >
                  <Globe className="w-3.5 h-3.5 text-[var(--text-subtle)]" />
                  <span className="text-[var(--text-primary)] flex-1 min-w-0 truncate text-left">
                    {LANGUAGES.find((l) => l.id === language)?.name || language}
                  </span>
                  <ChevronDown
                    className={clsx(
                      "w-3.5 h-3.5 text-[var(--text-subtle)] transition-transform",
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
                      className="absolute left-0 right-0 top-full mt-1 max-h-64 overflow-y-auto p-1 rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)] shadow-xl z-50"
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
                              ? "bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
                              : "hover:bg-[var(--bg-surface-3)] text-[var(--text-secondary)]",
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

          <div className="mb-4 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="text-[11px] text-[var(--text-subtle)] uppercase tracking-wide">
                  Active Model
                </div>
                {modelOptions.length > 0 && (
                  <div className="mt-2">{renderModelSelector()}</div>
                )}
                <div
                  className={clsx(
                    "mt-2 text-xs",
                    selectedModelReady
                      ? "text-[var(--text-secondary)]"
                      : "text-amber-400",
                  )}
                >
                  {selectedModelReady
                    ? "Loaded and ready"
                    : "Open Models and load a Base model"}
                </div>
              </div>
              {onOpenModelManager && (
                <div className="shrink-0">
                  <button
                    onClick={handleOpenModels}
                    className="btn btn-secondary text-xs"
                  >
                    <Settings2 className="w-4 h-4" />
                    Models
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-4">
            {/* Voice Reference Section */}
            <div className="p-3 rounded-lg bg-[var(--bg-surface-1)] border border-[var(--border-muted)]">
              <div className="flex items-center gap-2 mb-3">
                <Users className="w-4 h-4 text-[var(--text-muted)]" />
                <span className="text-xs font-medium text-[var(--text-primary)]">
                  Voice Reference
                </span>
                {isVoiceReady && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-surface-3)] text-[var(--text-secondary)] border border-[var(--border-muted)]">
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
              <label className="block text-xs text-[var(--text-muted)] font-medium mb-2">
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
                  <span className="text-xs text-[var(--text-subtle)]">
                    {text.length}
                  </span>
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
                      "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]",
                    downloadState === "success" &&
                      "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]",
                    downloadState === "error" &&
                      "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]",
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
                Load a Base model to clone voices
              </p>
            )}

            {selectedModelReady && !isVoiceReady && (
              <p className="text-xs text-[var(--text-secondary)]">
                Upload, record, or select a saved voice sample to get started
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
                <div className="p-3 rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)]">
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
      </div>

      <SpeechHistoryPanel
        route="voice-cloning"
        title="Voice Cloning History"
        emptyMessage="No saved voice-cloning generations yet."
        latestRecord={latestRecord}
        desktopHeightClassName="xl:h-[calc(100dvh-11.75rem)]"
      />
    </div>
  );
}
