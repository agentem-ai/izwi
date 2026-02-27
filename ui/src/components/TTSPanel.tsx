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
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

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
    <div className="rounded-xl border bg-card text-card-foreground shadow-sm p-4 sm:p-5 flex flex-col h-full">
      <div className="flex items-center justify-between mb-6 border-b pb-4">
        <h2 className="text-lg font-semibold tracking-tight flex items-center gap-2">
          <Volume2 className="w-5 h-5 text-muted-foreground" />
          Text to Speech
        </h2>
        <Button
          variant={showSettings ? "secondary" : "ghost"}
          size="icon"
          onClick={() => setShowSettings(!showSettings)}
          className={cn("h-8 w-8", showSettings ? "bg-accent" : "")}
        >
          <Settings className="w-4 h-4" />
        </Button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="mb-6 p-4 bg-muted/30 border rounded-lg space-y-4 shadow-inner">
          <div>
            <label className="block text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
              Speaker ID
            </label>
            <input
              type="text"
              value={speaker}
              onChange={(e) => setSpeaker(e.target.value)}
              placeholder="Optional speaker identifier"
              className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
                Temperature: {temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1.5"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-primary h-1.5 bg-muted rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
                Speed: {speed.toFixed(1)}x
              </label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-full accent-primary h-1.5 bg-muted rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
              />
            </div>
          </div>
        </div>
      )}

      {/* Text Input */}
      <div className="mb-4 flex-1 flex flex-col min-h-0">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to synthesize..."
          className="flex min-h-[120px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 flex-1 resize-none"
          disabled={generating}
        />
        <div className="flex justify-between items-center mt-2">
          <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
            {text.length} characters
          </span>
          {!selectedModel && (
            <span className="text-xs font-medium text-amber-500">
              Load a model to generate speech
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-xs font-medium">
          {error}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 pt-2">
        <Button
          onClick={handleGenerate}
          disabled={generating || !selectedModel}
          className="gap-2"
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
        </Button>

        {audioUrl && (
          <>
            <Button
              onClick={handleStop}
              variant="secondary"
              size="icon"
              title="Stop playback"
            >
              <Square className="w-4 h-4" />
            </Button>
            <Button
              onClick={handleDownload}
              variant="secondary"
              size="icon"
              title="Download audio"
            >
              <Download className="w-4 h-4" />
            </Button>
          </>
        )}
      </div>

      {/* Audio Player */}
      {audioUrl && (
        <div className="mt-4 p-3 bg-muted/30 border rounded-lg shadow-inner">
          <audio
            ref={audioRef}
            src={audioUrl}
            controls
            className="w-full h-10"
          />
        </div>
      )}
    </div>
  );
}
