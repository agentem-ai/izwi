import {
  Download,
  Play,
  Square,
  Loader2,
  Check,
  HardDrive,
} from "lucide-react";
import { ModelInfo } from "../api";

interface ModelCardProps {
  model: ModelInfo;
  isSelected: boolean;
  onDownload: () => void;
  onLoad: () => void;
  onUnload: () => void;
  onSelect: () => void;
  compact?: boolean;
}

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  "Qwen3-TTS-12Hz-0.6B-Base": "0.6B Base",
  "Qwen3-TTS-12Hz-0.6B-CustomVoice": "0.6B CustomVoice",
  "Qwen3-TTS-12Hz-1.7B-Base": "1.7B Base",
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": "1.7B CustomVoice",
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "1.7B VoiceDesign",
  "Qwen3-TTS-Tokenizer-12Hz": "Tokenizer 12Hz",
};

const formatBytes = (bytes: number | null): string => {
  if (bytes === null) return "Unknown size";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(0)} MB`;
};

export function ModelCard({
  model,
  isSelected,
  onDownload,
  onLoad,
  onUnload,
  onSelect,
  compact = false,
}: ModelCardProps) {
  const displayName = MODEL_DISPLAY_NAMES[model.variant] || model.variant;

  const statusColors = {
    not_downloaded: "bg-gray-600",
    downloading: "bg-amber-500",
    downloaded: "bg-gray-500",
    loading: "bg-amber-500",
    ready: "bg-emerald-500",
    error: "bg-red-500",
  };

  const statusLabels = {
    not_downloaded: "Not Downloaded",
    downloading: "Downloading...",
    downloaded: "Downloaded",
    loading: "Loading...",
    ready: "Ready",
    error: "Error",
  };

  const isLoading =
    model.status === "downloading" || model.status === "loading";

  if (compact) {
    return (
      <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${statusColors[model.status]}`}
          />
          <span className="text-sm text-gray-300">{displayName}</span>
        </div>
        {model.status === "not_downloaded" && (
          <button
            onClick={onDownload}
            className="btn btn-secondary text-xs py-1 px-2"
          >
            <Download className="w-3 h-3" />
          </button>
        )}
        {model.status === "downloaded" && (
          <button
            onClick={onLoad}
            className="btn btn-primary text-xs py-1 px-2"
          >
            <Play className="w-3 h-3" />
          </button>
        )}
        {isLoading && (
          <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />
        )}
        {model.status === "ready" && (
          <Check className="w-4 h-4 text-green-500" />
        )}
      </div>
    );
  }

  return (
    <div
      className={`p-4 rounded-lg border transition-all cursor-pointer ${
        isSelected
          ? "bg-[var(--bg-surface-3)] border-[var(--border-strong)]"
          : "bg-gray-800/50 border-gray-700 hover:border-gray-600"
      }`}
      onClick={() => model.status === "ready" && onSelect()}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-white">{displayName}</h3>
            {isSelected && (
              <span className="text-xs bg-[var(--accent-solid)] text-[var(--text-on-accent)] px-2 py-0.5 rounded">
                Active
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 mt-1">
            <span
              className={`text-xs px-2 py-0.5 rounded ${statusColors[model.status]} text-white`}
            >
              {statusLabels[model.status]}
            </span>
            {model.size_bytes && (
              <span className="text-xs text-gray-500 flex items-center gap-1">
                <HardDrive className="w-3 h-3" />
                {formatBytes(model.size_bytes)}
              </span>
            )}
          </div>
          {model.download_progress !== null &&
            model.status === "downloading" && (
              <div className="mt-2">
                <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-yellow-500 transition-all"
                    style={{ width: `${model.download_progress}%` }}
                  />
                </div>
              </div>
            )}
        </div>

        <div className="flex items-center gap-2 ml-4">
          {model.status === "not_downloaded" && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDownload();
              }}
              className="btn btn-secondary text-sm py-1.5"
            >
              <Download className="w-4 h-4 mr-1" />
              Download
            </button>
          )}
          {model.status === "downloaded" && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onLoad();
              }}
              className="btn btn-primary text-sm py-1.5"
            >
              <Play className="w-4 h-4 mr-1" />
              Load
            </button>
          )}
          {model.status === "ready" && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onUnload();
              }}
              className="btn btn-danger text-sm py-1.5"
            >
              <Square className="w-4 h-4 mr-1" />
              Unload
            </button>
          )}
          {isLoading && (
            <Loader2 className="w-5 h-5 text-yellow-500 animate-spin" />
          )}
        </div>
      </div>
    </div>
  );
}
