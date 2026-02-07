import { useState, useRef } from "react";
import {
  Play,
  Square,
  Download,
  Loader2,
  Volume2,
  Settings,
} from "lucide-react";
import { api } from "../api";

interface TTSPanelProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

export function TTSPanel({ selectedModel, onModelRequired }: TTSPanelProps) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [speed, setSpeed] = useState(1.0);
  const [generating, setGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);

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

      // Clear previous audio
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      const blob = await api.generateTTS({
        text: text.trim(),
        model_id: selectedModel,
        max_tokens: 0,
        speaker: speaker || undefined,
        temperature,
        speed,
      });

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      // Auto-play
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
      a.download = "speech.wav";
      a.click();
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Volume2 className="w-5 h-5" />
          Text to Speech
        </h2>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className={`p-2 rounded-lg transition-colors ${
            showSettings
              ? "bg-primary-600 text-white"
              : "text-gray-400 hover:text-white hover:bg-gray-800"
          }`}
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="mb-6 p-4 bg-gray-800/50 rounded-lg space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Speaker ID
            </label>
            <input
              type="text"
              value={speaker}
              onChange={(e) => setSpeaker(e.target.value)}
              placeholder="Optional speaker identifier"
              className="input"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Temperature: {temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1.5"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Speed: {speed.toFixed(1)}x
              </label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      )}

      {/* Text Input */}
      <div className="mb-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to synthesize..."
          rows={6}
          className="textarea"
          disabled={generating}
        />
        <div className="flex justify-between items-center mt-2">
          <span className="text-sm text-gray-500">
            {text.length} characters
          </span>
          {!selectedModel && (
            <span className="text-sm text-yellow-500">
              Load a model to generate speech
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded-lg text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleGenerate}
          disabled={generating || !selectedModel}
          className="btn btn-primary flex items-center gap-2"
        >
          {generating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
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
          </>
        )}
      </div>

      {/* Audio Player */}
      {audioUrl && (
        <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
          <audio ref={audioRef} src={audioUrl} controls className="w-full" />
        </div>
      )}

      {/* Sample Texts */}
      <div className="mt-6 pt-6 border-t border-gray-800">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Sample Texts</h3>
        <div className="space-y-2">
          {[
            "Hello! Welcome to Izwi, a text-to-speech engine powered by Qwen3-TTS.",
            "The quick brown fox jumps over the lazy dog.",
            "In a world where technology evolves rapidly, the ability to generate natural-sounding speech has become increasingly important.",
          ].map((sample, i) => (
            <button
              key={i}
              onClick={() => setText(sample)}
              className="w-full text-left p-3 bg-gray-800/30 hover:bg-gray-800/60 rounded-lg text-sm text-gray-300 transition-colors"
            >
              {sample}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
