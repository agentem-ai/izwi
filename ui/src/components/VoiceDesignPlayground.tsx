import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Wand2,
  Square,
  Download,
  RotateCcw,
  ChevronDown,
  Loader2,
  Globe,
} from "lucide-react";
import { api } from "../api";
import { LANGUAGES, VOICE_DESIGN_PRESETS } from "../types";
import clsx from "clsx";

interface VoiceDesignPlaygroundProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

export function VoiceDesignPlayground({
  selectedModel,
  onModelRequired,
}: VoiceDesignPlaygroundProps) {
  const [text, setText] = useState("");
  const [voiceDescription, setVoiceDescription] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [showLanguageSelect, setShowLanguageSelect] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleGenerate = async () => {
    if (!selectedModel) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      setError("Please enter some text to synthesize");
      return;
    }

    if (!voiceDescription.trim()) {
      setError("Please describe the voice you want to create");
      return;
    }

    try {
      setGenerating(true);
      setError(null);

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      const blob = await api.generateTTS({
        text: text.trim(),
        model_id: selectedModel,
        max_tokens: 0,
        voice_description: voiceDescription.trim(),
      });

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

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
      a.download = `izwi-voice-design-${Date.now()}.wav`;
      a.click();
    }
  };

  const handleReset = () => {
    setText("");
    setVoiceDescription("");
    setError(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    textareaRef.current?.focus();
  };

  const handlePresetSelect = (description: string) => {
    setVoiceDescription(description);
    setShowPresets(false);
  };

  return (
    <div className="card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded bg-[#1a1a1a] border border-[#2a2a2a]">
            <Wand2 className="w-5 h-5 text-gray-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">Voice Design</h2>
          </div>
        </div>

        {/* Language selector */}
        <div className="relative">
          <button
            onClick={() => setShowLanguageSelect(!showLanguageSelect)}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#1a1a1a] border border-[#2a2a2a] hover:bg-[#1f1f1f] text-sm"
          >
            <Globe className="w-3.5 h-3.5 text-gray-500" />
            <span className="text-white">
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
                className="absolute right-0 mt-1 w-44 max-h-64 overflow-y-auto p-1 rounded bg-[#1a1a1a] border border-[#2a2a2a] shadow-xl z-50"
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

      <div className="space-y-4">
        {/* Voice Description */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs text-gray-500 font-medium">
              Voice Description
            </label>
            <button
              onClick={() => setShowPresets(!showPresets)}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              {showPresets ? "Hide" : "Show"} presets
            </button>
          </div>

          <AnimatePresence>
            {showPresets && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-3 overflow-hidden"
              >
                <div className="grid grid-cols-2 gap-2 p-3 rounded-lg bg-[#161616] border border-[#2a2a2a]">
                  {VOICE_DESIGN_PRESETS.map((preset) => (
                    <button
                      key={preset.name}
                      onClick={() => handlePresetSelect(preset.description)}
                      className="p-2 rounded bg-[#1a1a1a] hover:bg-[#1f1f1f] border border-[#2a2a2a] text-left transition-colors"
                    >
                      <div className="text-xs font-medium text-white mb-1">
                        {preset.name}
                      </div>
                      <div className="text-[10px] text-gray-500 line-clamp-2">
                        {preset.description}
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <textarea
            value={voiceDescription}
            onChange={(e) => setVoiceDescription(e.target.value)}
            placeholder="Describe the voice you want to create... (e.g., 'A warm, friendly female voice with a slight British accent, speaking in a calm and reassuring tone')"
            rows={3}
            className="textarea text-sm"
          />
          <p className="text-[10px] text-gray-400 mt-1.5">
            Describe voice characteristics like gender, age, tone, emotion,
            accent, and speaking style
          </p>
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
              placeholder="Enter the text you want to synthesize..."
              rows={4}
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
        <div className="flex items-center gap-2">
          <button
            onClick={handleGenerate}
            disabled={generating || !selectedModel}
            className={clsx(
              "btn flex-1",
              generating ? "btn-secondary" : "btn-primary",
            )}
          >
            {generating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Designing Voice...
              </>
            ) : (
              <>
                <Wand2 className="w-4 h-4" />
                Generate
              </>
            )}
          </button>

          {audioUrl && (
            <>
              <button onClick={handleStop} className="btn btn-secondary">
                <Square className="w-4 h-4" />
              </button>
              <button onClick={handleDownload} className="btn btn-secondary">
                <Download className="w-4 h-4" />
              </button>
              <button onClick={handleReset} className="btn btn-ghost">
                <RotateCcw className="w-4 h-4" />
              </button>
            </>
          )}
        </div>

        {!selectedModel && (
          <p className="text-xs text-gray-400">
            Load a VoiceDesign model to create unique voices
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
            className="mt-4 p-3 rounded bg-[#1a1a1a] border border-[#2a2a2a]"
          >
            <audio ref={audioRef} src={audioUrl} className="w-full" controls />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
