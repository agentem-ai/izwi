import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Download,
  Play,
  Square,
  Trash2,
  ChevronRight,
  Loader2,
  X,
} from "lucide-react";
import { ModelInfo } from "../api";
import { withQwen3Prefix } from "../utils/modelDisplay";
import clsx from "clsx";

function parseSize(sizeStr: string): number {
  const match = sizeStr.match(/^([\d.]+)\s*(GB|MB|KB|B)?$/i);
  if (!match) return 0;
  const value = parseFloat(match[1]);
  const unit = (match[2] || "B").toUpperCase();
  const multipliers: Record<string, number> = {
    B: 1,
    KB: 1024,
    MB: 1024 * 1024,
    GB: 1024 * 1024 * 1024,
  };
  return value * (multipliers[unit] || 1);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

interface ModelManagerProps {
  models: ModelInfo[];
  selectedModel: string | null;
  onDownload: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  modelFilter?: (variant: string) => boolean;
  isModelDisabled?: (variant: string) => boolean;
  disabledModelLabel?: string;
  emptyStateTitle?: string;
  emptyStateDescription?: string;
}

const MODEL_DETAILS: Record<
  string,
  {
    shortName: string;
    fullName: string;
    description: string;
    features: string[];
    size: string;
    quantization?: string;
  }
> = {
  "Qwen3-TTS-12Hz-0.6B-Base": {
    shortName: "0.6B Base",
    fullName: "Qwen3-TTS 12Hz 0.6B Base Model",
    description: "Voice cloning with reference audio samples",
    features: ["Voice cloning", "Reference audio required", "Fast inference"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-4bit": {
    shortName: "0.6B Base 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 4-bit)",
    description:
      "Quantized base model for lower VRAM without losing cloning support",
    features: ["Voice cloning", "Reference audio", "4-bit quantized"],
    size: "1.6 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-8bit": {
    shortName: "0.6B Base 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 8-bit)",
    description:
      "8-bit MLX weights for better quality while staying memory friendly",
    features: ["Voice cloning", "Reference audio", "8-bit quantized"],
    size: "1.9 GB",
    quantization: "8-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-bf16": {
    shortName: "0.6B Base BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX bf16)",
    description: "BF16 MLX weights for highest fidelity base voices",
    features: ["Voice cloning", "Reference audio", "bf16 weights"],
    size: "2.3 GB",
    quantization: "bf16",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
    shortName: "0.6B Custom",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice Model",
    description: "Pre-trained with 9 built-in voice profiles",
    features: ["9 built-in voices", "No reference needed", "Fast generation"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit": {
    shortName: "0.6B Custom 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 4-bit)",
    description: "Quantized CustomVoice for laptops with tight memory",
    features: ["9 built-in voices", "4-bit quantized", "Fast inference"],
    size: "1.6 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit": {
    shortName: "0.6B Custom 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 8-bit)",
    description:
      "Balanced 8-bit CustomVoice for better quality with reduced VRAM",
    features: ["9 built-in voices", "8-bit quantized", "Balanced quality"],
    size: "1.8 GB",
    quantization: "8-bit",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16": {
    shortName: "0.6B Custom BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX bf16)",
    description: "BF16 precision for premium CustomVoice rendering",
    features: ["9 built-in voices", "bf16 weights", "Best fidelity"],
    size: "2.3 GB",
    quantization: "bf16",
  },
  "Qwen3-TTS-12Hz-1.7B-Base": {
    shortName: "1.7B Base",
    fullName: "Qwen3-TTS 12Hz 1.7B Base Model",
    description: "Higher quality voice cloning capabilities",
    features: [
      "Superior voice cloning",
      "Reference audio required",
      "Best quality",
    ],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
    shortName: "1.7B Custom",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice Model",
    description: "Premium quality with 9 built-in voices",
    features: ["9 built-in voices", "Highest quality", "Natural prosody"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit": {
    shortName: "1.7B Design 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 4-bit)",
    description: "Quantized VoiceDesign for creative voices on 16GB devices",
    features: ["Text-to-voice", "Creative control", "4-bit quantized"],
    size: "2.2 GB",
    quantization: "4-bit",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit": {
    shortName: "1.7B Design 8-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 8-bit)",
    description: "8-bit VoiceDesign balancing VRAM usage and quality",
    features: ["Text-to-voice", "Creative control", "8-bit quantized"],
    size: "2.9 GB",
    quantization: "8-bit",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16": {
    shortName: "1.7B Design BF16",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX bf16)",
    description: "BF16 VoiceDesign with best timbre quality",
    features: ["Text-to-voice", "Creative control", "bf16 weights"],
    size: "4.2 GB",
    quantization: "bf16",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
    shortName: "1.7B Design",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign Model",
    description: "Generate voices from text descriptions",
    features: ["Text-to-voice", "Creative control", "Unique voices"],
    size: "4.2 GB",
  },
  "LFM2-Audio-1.5B": {
    shortName: "LFM2 1.5B",
    fullName: "LFM2-Audio 1.5B by Liquid AI",
    description:
      "End-to-end audio foundation model for TTS, ASR, and audio chat",
    features: ["TTS", "ASR", "Audio-to-audio chat", "4 voice styles"],
    size: "3.0 GB",
  },
  "Qwen3-0.6B-4bit": {
    shortName: "Qwen3 Chat 0.6B",
    fullName: "Qwen3 0.6B (MLX 4-bit)",
    description: "Compact text-to-text chat model optimized for local inference",
    features: ["Text chat", "4-bit quantized", "Fast response"],
    size: "0.9 GB",
    quantization: "4-bit",
  },
  "Qwen3-ASR-0.6B": {
    shortName: "ASR 0.6B",
    fullName: "Qwen3-ASR 0.6B",
    description: "Fast speech-to-text model supporting 52 languages",
    features: ["52 languages", "Language detection", "Fast inference"],
    size: "1.8 GB",
  },
  "Qwen3-ASR-0.6B-4bit": {
    shortName: "ASR 0.6B 4-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 4-bit)",
    description: "Lightweight ASR for real-time transcription on smaller GPUs",
    features: ["52 languages", "4-bit quantized", "Fast inference"],
    size: "0.7 GB",
    quantization: "4-bit",
  },
  "Qwen3-ASR-0.6B-8bit": {
    shortName: "ASR 0.6B 8-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 8-bit)",
    description: "8-bit ASR with higher accuracy and modest footprint",
    features: ["52 languages", "8-bit quantized", "Balanced quality"],
    size: "0.9 GB",
    quantization: "8-bit",
  },
  "Qwen3-ASR-0.6B-bf16": {
    shortName: "ASR 0.6B BF16",
    fullName: "Qwen3-ASR 0.6B (MLX bf16)",
    description: "BF16 precision ASR for top accuracy",
    features: ["52 languages", "bf16 weights", "Highest fidelity"],
    size: "1.5 GB",
    quantization: "bf16",
  },
  "Qwen3-ASR-1.7B": {
    shortName: "ASR 1.7B",
    fullName: "Qwen3-ASR 1.7B",
    description: "High-quality speech-to-text model supporting 52 languages",
    features: [
      "52 languages",
      "Language detection",
      "State-of-the-art accuracy",
    ],
    size: "4.4 GB",
  },
  "Qwen3-ASR-1.7B-4bit": {
    shortName: "ASR 1.7B 4-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 4-bit)",
    description: "Quantized 1.7B ASR for RTX 4090 / M3 workloads",
    features: ["52 languages", "4-bit quantized", "Large-context"],
    size: "1.5 GB",
    quantization: "4-bit",
  },
  "Qwen3-ASR-1.7B-8bit": {
    shortName: "ASR 1.7B 8-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 8-bit)",
    description: "8-bit ASR for high fidelity transcripts on Apple Silicon",
    features: ["52 languages", "8-bit quantized", "Large-context"],
    size: "2.3 GB",
    quantization: "8-bit",
  },
  "Qwen3-ASR-1.7B-bf16": {
    shortName: "ASR 1.7B BF16",
    fullName: "Qwen3-ASR 1.7B (MLX bf16)",
    description: "BF16 ASR providing maximum quality and accuracy",
    features: ["52 languages", "bf16 weights", "Large-context"],
    size: "3.8 GB",
    quantization: "bf16",
  },
  "Voxtral-Mini-4B-Realtime-2602": {
    shortName: "Voxtral 4B",
    fullName: "Voxtral Mini 4B Realtime",
    description:
      "Realtime streaming ASR model from Mistral AI with high-quality transcription",
    features: [
      "Realtime streaming",
      "High-quality transcription",
      "Multilingual support",
      "Causal attention for streaming",
    ],
    size: "~8 GB",
  },
};

export function ModelManager({
  models,
  selectedModel,
  onDownload,
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onCancelDownload,
  downloadProgress,
  modelFilter,
  isModelDisabled,
  disabledModelLabel,
  emptyStateTitle,
  emptyStateDescription,
}: ModelManagerProps) {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [pendingDeleteVariant, setPendingDeleteVariant] = useState<string | null>(
    null,
  );
  const ttsModels = models
    .filter((m) => !m.variant.includes("Tokenizer"))
    .filter((m) => (modelFilter ? modelFilter(m.variant) : true))
    .sort((a, b) => {
      // Sort by size (smallest to largest)
      const sizeA = parseSize(MODEL_DETAILS[a.variant]?.size || "0");
      const sizeB = parseSize(MODEL_DETAILS[b.variant]?.size || "0");
      if (sizeA !== sizeB) {
        return sizeA - sizeB;
      }
      // If sizes are equal, sort by name
      return a.variant.localeCompare(b.variant);
    });

  if (ttsModels.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <div className="w-12 h-12 rounded-full bg-[#1a1a1a] flex items-center justify-center mb-3">
          <Download className="w-5 h-5 text-gray-500" />
        </div>
        <h3 className="text-sm font-medium text-gray-300 mb-1">
          {emptyStateTitle || "No Models Available"}
        </h3>
        <p className="text-xs text-gray-600 max-w-[200px]">
          {emptyStateDescription || "Download models to get started"}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {ttsModels.map((model) => {
        const details = MODEL_DETAILS[model.variant] || {
          shortName: model.variant,
          fullName: model.variant,
          description: "",
          features: [],
          size: "",
        };
        const displayName = withQwen3Prefix(details.shortName, model.variant);

        const isSelected = selectedModel === model.variant;
        const isExpanded = expandedModel === model.variant;
        const isDownloading = model.status === "downloading";
        const isLoading = model.status === "loading";
        const isReady = model.status === "ready";
        const isDownloaded = model.status === "downloaded";
        const isDisabled = isModelDisabled?.(model.variant) ?? false;
        const progressValue = downloadProgress[model.variant];
        const progress = progressValue?.percent ?? model.download_progress ?? 0;

        return (
          <div
            key={model.variant}
            className={clsx(
              "border rounded-lg transition-colors",
              isDisabled && "opacity-50",
              isSelected
                ? "border-white/20 bg-[#1a1a1a]"
                : "border-[#2a2a2a] bg-[#161616]",
            )}
          >
            {/* Main card */}
            <div
              className={clsx(
                "p-3",
                isDisabled ? "cursor-not-allowed" : "cursor-pointer",
                !isDisabled && !isExpanded && "hover:bg-[#1a1a1a]",
              )}
              aria-disabled={isDisabled}
              onClick={() => {
                if (isDisabled) {
                  return;
                }
                if (isReady && !isSelected) {
                  onSelect(model.variant);
                }
                setExpandedModel(isExpanded ? null : model.variant);
              }}
            >
              <div className="flex items-center gap-3">
                {/* Status indicator */}
                <div className="flex-shrink-0">
                  {isDownloading ? (
                    <div className="relative w-8 h-8">
                      <svg className="w-8 h-8 transform -rotate-90">
                        <circle
                          cx="16"
                          cy="16"
                          r="14"
                          fill="none"
                          stroke="#2a2a2a"
                          strokeWidth="2"
                        />
                        <circle
                          cx="16"
                          cy="16"
                          r="14"
                          fill="none"
                          stroke="#ffffff"
                          strokeWidth="2"
                          strokeDasharray={`${2 * Math.PI * 14}`}
                          strokeDashoffset={`${2 * Math.PI * 14 * (1 - progress / 100)}`}
                          strokeLinecap="round"
                          className="transition-all duration-300"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center text-[10px] text-white font-medium">
                        {Math.round(progress)}
                      </div>
                    </div>
                  ) : isLoading ? (
                    <Loader2 className="w-5 h-5 text-white animate-spin" />
                  ) : (
                    <div
                      className={clsx(
                        "w-2 h-2 rounded-full",
                        isReady && "bg-green-500",
                        isDownloaded && "bg-blue-500",
                        model.status === "not_downloaded" && "bg-gray-600",
                      )}
                    />
                  )}
                </div>

                {/* Model info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white">
                      {displayName}
                    </span>
                    {isDisabled && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-amber-500/20 text-amber-300 rounded">
                        {disabledModelLabel || "DISABLED"}
                      </span>
                    )}
                    {isSelected && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-white/10 text-white rounded">
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    {details.size}
                    {isDownloading && progressValue && (
                      <>
                        {" "}
                        • {progress.toFixed(0)}% (
                        {formatBytes(progressValue.downloadedBytes)} /{" "}
                        {formatBytes(progressValue.totalBytes)})
                      </>
                    )}
                    {isLoading && " • Loading..."}
                  </div>
                </div>

                {/* Expand icon */}
                <ChevronRight
                  className={clsx(
                    "w-4 h-4 text-gray-500 transition-transform flex-shrink-0",
                    isExpanded && "rotate-90",
                  )}
                />
              </div>

              {/* Progress bar */}
              {isDownloading && (
                <div className="mt-2 h-1 bg-[#1f1f1f] rounded-sm overflow-hidden">
                  <div
                    className="h-full bg-white rounded-sm transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </div>

            {/* Expanded details */}
            <AnimatePresence>
              {isExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden border-t border-[#2a2a2a]"
                >
                  <div className="p-3 space-y-3">
                    {/* Full name */}
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Model</div>
                      <div className="text-sm text-white font-mono">
                        {details.fullName}
                      </div>
                    </div>

                    {/* Description */}
                    <div>
                      <div className="text-xs text-gray-500 mb-1">
                        Description
                      </div>
                      <div className="text-sm text-gray-300">
                        {details.description}
                      </div>
                    </div>

                    {/* Features */}
                    {details.features.length > 0 && (
                      <div>
                        <div className="text-xs text-gray-500 mb-1">
                          Features
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {details.features.map((feature, i) => (
                            <span
                              key={i}
                              className="text-xs px-2 py-1 bg-[#1f1f1f] text-gray-400 rounded"
                            >
                              {feature}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex items-center gap-2 pt-2">
                      {isDisabled && (
                        <div className="text-xs text-amber-300">
                          This model is unavailable in this view.
                        </div>
                      )}

                      {isDownloading && onCancelDownload && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onCancelDownload(model.variant);
                          }}
                          className="btn btn-danger text-sm flex-1"
                          disabled={isDisabled}
                        >
                          <X className="w-4 h-4" />
                          Cancel Download
                        </button>
                      )}

                      {model.status === "not_downloaded" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDownload(model.variant);
                          }}
                          className="btn btn-primary text-sm flex-1"
                          disabled={isDisabled}
                        >
                          <Download className="w-4 h-4" />
                          Download
                        </button>
                      )}

                      {isDownloaded && (
                        <>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onLoad(model.variant);
                            }}
                            className="btn btn-primary text-sm flex-1"
                            disabled={isDisabled}
                          >
                            <Play className="w-4 h-4" />
                            Load
                          </button>
                          {pendingDeleteVariant === model.variant ? (
                            <div className="flex items-center gap-1">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                  onDelete(model.variant);
                                }}
                                className="btn btn-danger text-sm"
                                disabled={isDisabled}
                              >
                                <Check className="w-4 h-4" />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                }}
                                className="btn btn-secondary text-sm"
                                disabled={isDisabled}
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>
                          ) : (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setPendingDeleteVariant(model.variant);
                              }}
                              className="btn btn-danger text-sm"
                              disabled={isDisabled}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </>
                      )}

                      {isReady && (
                        <>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onUnload(model.variant);
                            }}
                            className="btn btn-secondary text-sm flex-1"
                            disabled={isDisabled}
                          >
                            <Square className="w-4 h-4" />
                            Unload
                          </button>
                          {pendingDeleteVariant === model.variant ? (
                            <div className="flex items-center gap-1">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                  onDelete(model.variant);
                                }}
                                className="btn btn-danger text-sm"
                                disabled={isDisabled}
                              >
                                <Check className="w-4 h-4" />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPendingDeleteVariant(null);
                                }}
                                className="btn btn-secondary text-sm"
                                disabled={isDisabled}
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>
                          ) : (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setPendingDeleteVariant(model.variant);
                              }}
                              className="btn btn-danger text-sm"
                              disabled={isDisabled}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}
