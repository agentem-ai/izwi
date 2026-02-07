import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Square,
  Download,
  RotateCcw,
  ChevronDown,
  Volume2,
  Loader2,
  User,
} from "lucide-react";
import { api } from "../api";
import { VoiceClone } from "./VoiceClone";
import clsx from "clsx";

interface VoicePlaygroundProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

const SPEAKERS = [
  { id: "Vivian", name: "Vivian" },
  { id: "Serena", name: "Serena" },
  { id: "Ryan", name: "Ryan" },
  { id: "Aiden", name: "Aiden" },
  { id: "Dylan", name: "Dylan" },
  { id: "Eric", name: "Eric" },
  { id: "Sohee", name: "Sohee" },
  { id: "Ono_anna", name: "Anna" },
  { id: "Uncle_fu", name: "Uncle Fu" },
];

const SAMPLE_TEXTS = [
  "Hello! Welcome to Izwi, a text-to-speech engine powered by Qwen3-TTS.",
  "The quick brown fox jumps over the lazy dog.",
  "In a world where technology evolves rapidly, the ability to generate natural-sounding speech has become increasingly important.",
];

export function VoicePlayground({
  selectedModel,
  onModelRequired,
}: VoicePlaygroundProps) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("Vivian");
  const [showSpeakerSelect, setShowSpeakerSelect] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [useVoiceClone, setUseVoiceClone] = useState(false);
  const [voiceCloneAudio, setVoiceCloneAudio] = useState<string | null>(null);
  const [voiceCloneTranscript, setVoiceCloneTranscript] = useState<
    string | null
  >(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isBaseModel = selectedModel?.includes("Base") || false;

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

      console.log(
        "[VoicePlayground] Generating TTS - useVoiceClone:",
        useVoiceClone,
        "hasAudio:",
        !!voiceCloneAudio,
        "hasTranscript:",
        !!voiceCloneTranscript,
      );

      const request = {
        text: text.trim(),
        max_tokens: 0,
        speaker: useVoiceClone ? undefined : speaker,
        reference_audio: useVoiceClone
          ? voiceCloneAudio || undefined
          : undefined,
        reference_text: useVoiceClone
          ? voiceCloneTranscript || undefined
          : undefined,
      };

      console.log("[VoicePlayground] API request:", {
        ...request,
        reference_audio: request.reference_audio
          ? `[${request.reference_audio.length} chars]`
          : undefined,
      });

      const blob = await api.generateTTS(request);

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
      a.download = `izwi-${speaker.toLowerCase()}-${Date.now()}.wav`;
      a.click();
    }
  };

  const handleReset = () => {
    setText("");
    setError(null);
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
        <div>
          <h2 className="text-sm font-medium text-white">Voice Playground</h2>
          <p className="text-xs text-gray-600 mt-0.5">
            Generate speech from text
          </p>
        </div>

        {/* Speaker selector */}
        <div className="relative">
          <button
            onClick={() => setShowSpeakerSelect(!showSpeakerSelect)}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#1a1a1a] border border-[#2a2a2a] hover:bg-[#1f1f1f] text-sm"
          >
            <Volume2 className="w-3.5 h-3.5 text-gray-500" />
            <span className="text-white">{speaker}</span>
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
                className="absolute right-0 mt-1 w-40 p-1 rounded bg-[#1a1a1a] border border-[#2a2a2a] shadow-xl z-50"
              >
                {SPEAKERS.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setSpeaker(s.id);
                      setShowSpeakerSelect(false);
                    }}
                    className={clsx(
                      "w-full px-2 py-1.5 rounded text-left text-sm transition-colors",
                      speaker === s.id
                        ? "bg-white/10 text-white"
                        : "hover:bg-[#2a2a2a] text-gray-400",
                    )}
                  >
                    {s.name}
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

        {/* Sample texts */}
        <div className="flex flex-wrap gap-2">
          {SAMPLE_TEXTS.map((sample, i) => (
            <button
              key={i}
              onClick={() => setText(sample)}
              className="text-xs px-2 py-1 rounded bg-[#1a1a1a] hover:bg-[#1f1f1f] text-gray-500 hover:text-gray-300 border border-[#2a2a2a]"
            >
              Sample {i + 1}
            </button>
          ))}
        </div>

        {/* Voice Cloning Section (for Base models) */}
        {isBaseModel && (
          <div className="p-3 rounded-lg bg-[#161616] border border-[#2a2a2a]">
            <div className="flex items-center gap-2 mb-3">
              <User className="w-4 h-4 text-gray-500" />
              <span className="text-xs font-medium text-white">
                Voice Cloning
              </span>
              <span className="text-xs text-gray-600">
                (Required for Base models)
              </span>
            </div>
            <VoiceClone
              onVoiceCloneReady={(audio, transcript) => {
                console.log(
                  "[VoicePlayground] Voice clone ready - audio length:",
                  audio?.length,
                  "transcript:",
                  transcript,
                );
                setVoiceCloneAudio(audio);
                setVoiceCloneTranscript(transcript);
                setUseVoiceClone(true);
              }}
              onClear={() => {
                console.log("[VoicePlayground] Voice clone cleared");
                setVoiceCloneAudio(null);
                setVoiceCloneTranscript(null);
                setUseVoiceClone(false);
              }}
            />
          </div>
        )}

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
                Generating...
              </>
            ) : (
              "Generate"
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
          <p className="text-xs text-gray-600">
            Load a model to generate speech
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
