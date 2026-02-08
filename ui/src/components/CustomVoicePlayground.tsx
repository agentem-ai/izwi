import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Volume2,
  Square,
  Download,
  RotateCcw,
  ChevronDown,
  Loader2,
  MessageSquare,
} from "lucide-react";
import { api, TTSGenerationStats } from "../api";
import { SPEAKERS } from "../types";
import { GenerationStats } from "./GenerationStats";
import clsx from "clsx";

interface CustomVoicePlaygroundProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

export function CustomVoicePlayground({
  selectedModel,
  onModelRequired,
}: CustomVoicePlaygroundProps) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("Vivian");
  const [instruct, setInstruct] = useState("");
  const [showSpeakerSelect, setShowSpeakerSelect] = useState(false);
  const [showInstruct, setShowInstruct] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [generationStats, setGenerationStats] =
    useState<TTSGenerationStats | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const selectedSpeaker = SPEAKERS.find((s) => s.id === speaker);

  const handleGenerate = async () => {
    if (!selectedModel) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      setError("Please enter some text");
      return;
    }

    try {
      setGenerating(true);
      setError(null);

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }
      setGenerationStats(null);

      const result = await api.generateTTSWithStats({
        text: text.trim(),
        model_id: selectedModel,
        max_tokens: 0,
        speaker: speaker,
        voice_description: instruct.trim() || undefined,
      });

      const url = URL.createObjectURL(result.audioBlob);
      setAudioUrl(url);
      setGenerationStats(result.stats);

      setTimeout(() => {
        audioRef.current?.play();
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

  const handleDownload = () => {
    if (audioUrl) {
      const a = document.createElement("a");
      a.href = audioUrl;
      a.download = `izwi-${speaker.toLowerCase()}-${Date.now()}.wav`;
      a.click();
    }
  };

  const handleReset = () => {
    setText("");
    setInstruct("");
    setError(null);
    setGenerationStats(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    textareaRef.current?.focus();
  };

  return (
    <div className="card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded bg-[#1a1a1a] border border-[#2a2a2a]">
            <Volume2 className="w-5 h-5 text-gray-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">Text to Speech</h2>
          </div>
        </div>

        {/* Speaker selector */}
        <div className="relative">
          <button
            onClick={() => setShowSpeakerSelect(!showSpeakerSelect)}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#1a1a1a] border border-[#2a2a2a] hover:bg-[#1f1f1f] text-sm"
          >
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-[10px] font-medium text-white">
              {speaker.charAt(0)}
            </div>
            <span className="text-white">
              {selectedSpeaker?.name || speaker}
            </span>
            <ChevronDown
              className={clsx(
                "w-3.5 h-3.5 text-gray-500 transition-transform",
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
                className="absolute right-0 mt-1 w-56 sm:w-64 max-h-80 overflow-y-auto p-1 rounded bg-[#1a1a1a] border border-[#2a2a2a] shadow-xl z-50"
              >
                {SPEAKERS.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setSpeaker(s.id);
                      setShowSpeakerSelect(false);
                    }}
                    className={clsx(
                      "w-full px-3 py-2 rounded text-left transition-colors flex items-center gap-3",
                      speaker === s.id ? "bg-white/10" : "hover:bg-[#2a2a2a]",
                    )}
                  >
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-xs font-medium text-white flex-shrink-0">
                      {s.name.charAt(0)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className={clsx(
                          "text-sm font-medium",
                          speaker === s.id ? "text-white" : "text-gray-300",
                        )}
                      >
                        {s.name}
                      </div>
                      <div className="text-[10px] text-gray-500 truncate">
                        {s.description}
                      </div>
                    </div>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#2a2a2a] text-gray-500">
                      {s.language}
                    </span>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
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
            <span className="text-xs text-gray-600">{text.length}</span>
          </div>
        </div>

        {/* Instruct toggle */}
        <button
          onClick={() => setShowInstruct(!showInstruct)}
          className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
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

        <AnimatePresence>
          {showInstruct && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="p-3 rounded-lg bg-[#161616] border border-[#2a2a2a]">
                <label className="block text-xs text-gray-500 mb-1.5">
                  Speaking Style Instructions
                </label>
                <input
                  type="text"
                  value={instruct}
                  onChange={(e) => setInstruct(e.target.value)}
                  placeholder="e.g., 'Speak with excitement' or 'Very calm and soothing'"
                  className="input text-sm"
                />
                <p className="text-[10px] text-gray-400 mt-1.5">
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

        {/* Actions */}
        <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
          <button
            onClick={handleGenerate}
            disabled={generating || !selectedModel}
            className={clsx(
              "btn flex-1 min-h-[44px]",
              generating ? "btn-secondary" : "btn-primary",
            )}
          >
            {generating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating...
              </>
            ) : (
              "Generate"
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

        {!selectedModel && (
          <p className="text-xs text-gray-400">
            Load a CustomVoice model to generate speech
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
